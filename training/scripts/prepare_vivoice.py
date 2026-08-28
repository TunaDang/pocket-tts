"""Prepare capleaf/viVoice (1,017h Vietnamese, 24 kHz) into the training manifest schema.

    python -m training.scripts.prepare_vivoice --align-devices 5,6

viVoice is YouTube speech curated for multi-speaker TTS: one short utterance per
parquet row, audio embedded as 24 kHz wav bytes, speaker identified by the
channel handle. Same shape as prepare_vieneu.py, so the two manifests
concatenate directly, and the extraction copies audio bytes verbatim -- viVoice
is already at Mimi's rate, and measurably full-band (its energy at 8-9 kHz sits
+0.1 dB relative to 7-8 kHz, i.e. no resampling cliff).

Why the duration floor: the DataLoader cuts an utterance at a word boundary
with MIN_CUT_SEC of audio on both sides, so anything under 2s can never be cut
and falls back to a voice prompt drawn from a random window of the SAME file --
for a short clip that prompt IS the target, which trains copying rather than
synthesis. viVoice is short-form (median 3.3s, 26% under 2s), so unlike
HiFiTTS-2 or VieNeu the floor actually binds here. At the 2s default it costs
8.8% of the hours to remove 25.9% of the rows.
"""

import json
import logging
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Annotated

import pyarrow.parquet as pq
import typer
from tqdm import tqdm

from .prepare_data import align
from .prepare_vieneu import fit_tokenizer, normalize_text, wav_duration

logger = logging.getLogger("prepare_vivoice")

app = typer.Typer(pretty_exceptions_show_locals=False)

REPO = "capleaf/viVoice"
ALIGN_MODEL = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
# The loader needs MIN_CUT_SEC (1.0s) of audio on each side of a cut.
MIN_CUTTABLE_SEC = 2.0

_SAFE = re.compile(r"[^A-Za-z0-9._@+-]")


def safe_dirname(name: str) -> str:
    """A channel handle reduced to a filesystem-safe directory name.

    Channel names come from YouTube and are attacker-influenced in principle:
    they may carry separators, whitespace, or unicode that would escape the
    audio root. Names that reduce to nothing (or to a dot entry) fall back to a
    hash so two distinct channels never collide into one directory.
    """
    safe = _SAFE.sub("_", name.strip())[:64].strip("._")
    return safe or f"ch_{zlib.crc32(name.encode()):08x}"


@app.command()
def main(
    manifests_out: Annotated[str, typer.Option()] = "data/vivoice",
    audio_out: Annotated[str, typer.Option()] = "data/vivoice/audio",
    snapshot_dir: Annotated[str, typer.Option()] = "data/downloads/vivoice",
    min_duration: Annotated[
        float, typer.Option(help="drop clips shorter than this (see module docstring)")
    ] = MIN_CUTTABLE_SEC,
    valid_frac: Annotated[float, typer.Option()] = 0.004,
    vocab_size: Annotated[int, typer.Option()] = 4000,
    align_devices: Annotated[str, typer.Option(help="physical GPU ids, one shard each")] = "5,6",
    skip_align: Annotated[bool, typer.Option()] = False,
    tokenizer: Annotated[
        bool,
        typer.Option(
            help="fit a tokenizer on THIS corpus alone (usually you want the "
            "combined one instead, so this defaults off)"
        ),
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    out_dir, audio_root = Path(manifests_out), Path(audio_out)
    for d in (out_dir, audio_root):
        d.mkdir(parents=True, exist_ok=True)

    shards = sorted(Path(snapshot_dir).glob("data/*.parquet"))
    assert shards, f"no parquet shards under {snapshot_dir}/data"
    logger.info(f"{len(shards)} shards")

    train_m, valid_m = out_dir / "train.jsonl", out_dir / "valid.jsonl"
    tmp_t, tmp_v = train_m.with_suffix(".jsonl.tmp"), valid_m.with_suffix(".jsonl.tmp")
    kept, kept_h, dropped, speakers = Counter(), Counter(), Counter(), set()
    with open(tmp_t, "w") as ftr, open(tmp_v, "w") as fev:
        for shard in tqdm(shards, unit="shard", desc="Extract"):
            stem = shard.stem
            for batch_i, batch in enumerate(pq.ParquetFile(shard).iter_batches(batch_size=512)):
                for row_i, row in enumerate(batch.to_pylist()):
                    audio = row["audio"]
                    raw = audio["bytes"] if isinstance(audio, dict) else audio
                    transcript = normalize_text(row["text"] or "")
                    duration = wav_duration(raw)
                    if not transcript:
                        dropped["empty transcript"] += 1
                        continue
                    if duration is None:
                        dropped["unreadable wav header"] += 1
                        continue
                    if duration < min_duration:
                        dropped[f"shorter than {min_duration}s"] += 1
                        continue
                    channel = row["channel"] or "unknown"
                    speakers.add(channel)
                    uid = f"{stem}_{batch_i:04d}_{row_i:04d}"
                    path = audio_root / safe_dirname(channel) / f"{uid}.wav"
                    if not (path.exists() and path.stat().st_size == len(raw)):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        part = path.with_suffix(".wav.part")
                        part.write_bytes(raw)
                        part.rename(path)
                    rec = {
                        "path": str(path),
                        "start": 0.0,
                        "duration": round(duration, 4),
                        "transcript": transcript,
                        "speaker": channel,
                    }
                    held = (zlib.crc32(uid.encode()) % 1_000_000) < valid_frac * 1_000_000
                    (fev if held else ftr).write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept[held] += 1
                    kept_h[held] += duration / 3600
    tmp_t.rename(train_m)
    tmp_v.rename(valid_m)
    for label, n in dropped.most_common():
        logger.info(f"  dropped {n}: {label}")
    logger.info(
        f"train.jsonl: {kept[False]} utts (~{kept_h[False]:.1f}h); "
        f"valid.jsonl: {kept[True]} utts (~{kept_h[True]:.1f}h); {len(speakers)} channels"
    )
    if tokenizer:
        fit_tokenizer(out_dir / "tokenizer", train_m, vocab_size)
    if skip_align:
        logger.info(f"Done (no alignment). Manifests in {out_dir.resolve()}")
        return
    devices = [int(d) for d in align_devices.split(",") if d.strip()]
    align(train_m, out_dir / "train_aligned.jsonl", len(devices), ALIGN_MODEL, "train", devices)
    align(valid_m, out_dir / "valid_aligned.jsonl", 1, ALIGN_MODEL, "valid", devices[:1])
    logger.info(f"Done. {(out_dir / 'train_aligned.jsonl').resolve()}")


if __name__ == "__main__":
    app()
