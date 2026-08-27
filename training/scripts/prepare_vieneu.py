"""One-shot data preparation for Vietnamese: download pnnbao-ump/VieNeu-TTS-140h,
unpack it into per-utterance wavs, build manifests, fit a tokenizer, and attach
word alignments, leaving `train_aligned.jsonl` + `valid_aligned.jsonl` and
`tokenizer.model` in `data/vieneu_140h/`, ready for `training/train.py`.

    python -m training.scripts.prepare_vieneu --align-devices 5,6

VieNeu-TTS-140h is 140.7h / 74,858 utterances of 24 kHz Vietnamese speech with
orthographic transcripts (`text`) and an IPA rendering (`phonemized_text`).
We keep the orthographic text: it is what the aligner's CTC vocabulary spells,
what the sentencepiece tokenizer is fitted on, and what a user types at
inference. The IPA column is carried through the manifest but unused.

Unlike HiFiTTS-2 there are no chapter files to seek into: the dataset ships one
short utterance per row with the audio embedded as WAV bytes, so each row is
written out verbatim (no decode, no resample) and its manifest entry is the
whole file, `start: 0.0`. The manifest schema is otherwise identical to
prepare_data.py's, which is what lets align_data.py and the DataLoader read it
unchanged.

The script is resumable -- the download, the extracted wavs, the tokenizer and
the alignment shards are all skipped on a re-run.
"""

import json
import logging
import subprocess
import sys
import unicodedata
import zlib
from collections import Counter
from pathlib import Path
from typing import Annotated

import huggingface_hub
import pyarrow as pa
import typer
from tqdm import tqdm

from .prepare_data import align

logger = logging.getLogger("prepare_vieneu")

app = typer.Typer(pretty_exceptions_show_locals=False)

REPO = "pnnbao-ump/VieNeu-TTS-140h"
# The aligner's CTC vocabulary is lowercase Vietnamese; see align_data.py.
ALIGN_MODEL = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"


def wav_duration(raw: bytes) -> float | None:
    """Seconds of a canonical PCM RIFF/WAVE blob, read from its header alone.

    The dataset's own `duration` column agrees with this to the millisecond on
    every row we checked, but the header is what the DataLoader will actually
    seek against, so it is the authority.
    """
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None
    pos, byte_rate = 12, None
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        size = int.from_bytes(raw[pos + 4 : pos + 8], "little")
        body = raw[pos + 8 : pos + 8 + size]
        if cid == b"fmt " and len(body) >= 16:
            byte_rate = int.from_bytes(body[8:12], "little")
        elif cid == b"data":
            if not byte_rate:
                return None  # data before fmt: not something we should guess at
            # A truncated blob's header still claims the full size; bytes present win.
            return min(size, len(raw) - pos - 8) / byte_rate
        pos += 8 + size + (size & 1)  # chunks are word-aligned
    return None


def normalize_text(text: str) -> str:
    """NFC + collapsed whitespace.

    Vietnamese has two spellings for every toned vowel (precomposed and base +
    combining mark). The rows we sampled are all precomposed already, but a
    decomposed row would tokenize as different pieces from its precomposed
    twin and align through a different path, so this is pinned rather than
    assumed.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def is_valid_split(uid: str, valid_frac: float) -> bool:
    """Hold out an utterance by a hash of its id: stable across runs and
    machines (unlike `hash()`, which is salted per process), and independent of
    row order, so re-running with more shards keeps the earlier split."""
    return (zlib.crc32(uid.encode()) % 1_000_000) < valid_frac * 1_000_000


def read_shard(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "rb") as src:
        return pa.ipc.open_stream(src).read_all()


def extract(
    snapshot: Path, audio_root: Path, out_dir: Path, valid_frac: float
) -> tuple[Path, Path]:
    """Write every row's audio to `audio_root/<speaker>/<id>.wav` and both manifests.

    Audio bytes are copied out of the arrow shard untouched -- they are already
    the 24 kHz mono PCM wav that Mimi wants, so decoding and re-encoding them
    would only cost fidelity and time.
    """
    shards = sorted(snapshot.glob("data-*-of-*.arrow"))
    assert shards, f"no arrow shards under {snapshot}"
    train_manifest = out_dir / "train.jsonl"
    valid_manifest = out_dir / "valid.jsonl"
    seen: set[str] = set()
    written, written_h, skipped = Counter(), Counter(), Counter()
    speakers: set[str] = set()
    tmp_train = train_manifest.with_suffix(".jsonl.tmp")
    tmp_valid = valid_manifest.with_suffix(".jsonl.tmp")
    with open(tmp_train, "w") as ftr, open(tmp_valid, "w") as fev:
        for shard in tqdm(shards, unit="shard", desc="Extract"):
            for row in read_shard(shard).to_pylist():
                uid, raw = row["_id"], row["audio"]["bytes"]
                assert uid not in seen, f"duplicate utterance id {uid}"
                seen.add(uid)
                transcript = normalize_text(row["text"])
                duration = wav_duration(raw)
                if not transcript or duration is None or duration <= 0:
                    skipped["unusable row"] += 1
                    continue
                speaker = row["speaker"]
                speakers.add(speaker)
                path = audio_root / speaker / f"{uid}.wav"
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
                    "speaker": speaker,
                    "gender": row["gender"],
                    "phonemized_text": row["phonemized_text"],
                }
                held_out = is_valid_split(uid, valid_frac)
                (fev if held_out else ftr).write(json.dumps(rec, ensure_ascii=False) + "\n")
                written[held_out] += 1
                written_h[held_out] += duration / 3600
    tmp_train.rename(train_manifest)
    tmp_valid.rename(valid_manifest)
    for label, n in skipped.most_common():
        logger.info(f"  skipped {n} rows: {label}")
    logger.info(
        f"train.jsonl: {written[False]} utterances (~{written_h[False]:.1f}h); "
        f"valid.jsonl: {written[True]} utterances (~{written_h[True]:.1f}h); "
        f"{len(speakers)} speakers"
    )
    assert written[True] > 0, "valid split is empty -- raise --valid-frac"
    return train_manifest, valid_manifest


def fit_tokenizer(prefix: Path, manifest: Path, vocab_size: int) -> Path:
    """Fit sentencepiece on the training transcripts (skipped if already there).

    Vocabulary size stays at train_tokenizer.py's 4000 so the released config's
    lookup_table.n_bins needs no override -- only the path does.
    """
    model = prefix.with_suffix(".model")
    if model.exists():
        logger.info(f"tokenizer {model.resolve()} exists, skipping")
        return model
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.scripts.train_tokenizer",
            str(prefix),
            str(manifest),
            "--vocab-size",
            str(vocab_size),
        ],
        check=True,
    )
    return model


@app.command()
def main(
    manifests_out: Annotated[str, typer.Option(help="where manifests/tokenizer are written")] = (
        "data/vieneu_140h"
    ),
    audio_out: Annotated[
        str, typer.Option(help="where the unpacked per-utterance wavs go (~24 GB)")
    ] = "data/vieneu_140h/audio",
    snapshot_dir: Annotated[
        str, typer.Option(help="where the arrow shards are downloaded (~24 GB)")
    ] = "data/downloads/vieneu140h",
    valid_frac: Annotated[
        float, typer.Option(help="fraction of utterances held out for validation")
    ] = 0.01,
    vocab_size: Annotated[
        int, typer.Option(help="the model config's lookup_table.n_bins must equal this")
    ] = 4000,
    align_devices: Annotated[
        str,
        typer.Option(
            help="comma-separated physical GPU ids for forced alignment, one shard each "
            "(e.g. '5,6'). This box is shared -- never widen it without checking who owns what."
        ),
    ] = "5,6",
    skip_align: Annotated[bool, typer.Option(help="stop after manifests + tokenizer")] = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    out_dir, audio_root = Path(manifests_out), Path(audio_out)
    for d in (out_dir, audio_root):
        d.mkdir(parents=True, exist_ok=True)

    snapshot = Path(
        huggingface_hub.snapshot_download(REPO, repo_type="dataset", local_dir=snapshot_dir)
    )
    train_m, valid_m = extract(snapshot, audio_root, out_dir, valid_frac)
    tokenizer = fit_tokenizer(out_dir / "tokenizer", train_m, vocab_size)

    if skip_align:
        logger.info(f"Done (no alignment). Manifests in {out_dir.resolve()}")
        return
    devices = [int(d) for d in align_devices.split(",") if d.strip()]
    train_a = out_dir / "train_aligned.jsonl"
    valid_a = out_dir / "valid_aligned.jsonl"
    align(train_m, train_a, len(devices), ALIGN_MODEL, "training manifest", devices)
    align(valid_m, valid_a, 1, ALIGN_MODEL, "valid manifest", devices[:1])
    logger.info(
        f"Done. Training on {train_a.resolve()} and {valid_a.resolve()}, "
        f"tokenizer {tokenizer.resolve()}, audio in {audio_root.resolve()}"
    )


if __name__ == "__main__":
    app()
