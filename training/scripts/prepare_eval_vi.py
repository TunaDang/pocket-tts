"""Build an EXTERNAL Vietnamese evaluation set from FLEURS (google/fleurs, vi_vn).

    python -m training.scripts.prepare_eval_vi

Why external: the valid split that prepare_vieneu.py carves out holds out
utterances, but their neighbouring sentences -- same document, same speaker,
same synthesis pipeline -- stay in train. That measures very little. FLEURS
shares no speaker, domain, or pipeline with VieNeu, so it measures
generalization rather than recall.

What this writes is a TEXT set: `text` (cased and punctuated, the register the
model is trained on) to feed the TTS, and `reference` (FLEURS' own lowercased,
unpunctuated transcription) to score ASR output against for WER. Sentences are
deduplicated -- FLEURS records each sentence from several speakers -- and the
result is balanced by speaker gender, because vi_vn's dev and test splits are
entirely male. Drawing the balance from FLEURS' *train* split is sound here
precisely because we never train on FLEURS: all three splits are equally
held out for us.

Deliberately NOT a validation-loss set. FLEURS audio is 16 kHz and Mimi runs at
24 kHz, so a loss computed on it would partly be measuring the missing octave
rather than the model. For an external loss set, Common Voice vi
(fsicoli/common_voice_17_0, CC0) ships 48 kHz mp3 that downsamples to 24 kHz
with the band intact, and its test split alone carries 121 distinct speakers.
"""

import csv
import json
import logging
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Annotated

import huggingface_hub
import typer

logger = logging.getLogger("prepare_eval_vi")

app = typer.Typer(pretty_exceptions_show_locals=False)

REPO = "google/fleurs"
LANG = "vi_vn"
# id, filename, raw_transcription, transcription, chars, n_samples, gender
COLS = 7
FLEURS_SR = 16000


@app.command()
def main(
    out: Annotated[Path, typer.Option()] = Path("data/eval_vi/fleurs_vi.jsonl"),
    per_gender: Annotated[int, typer.Option(help="sentences kept per gender")] = 250,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []
    for split in ("dev", "test", "train"):
        path = huggingface_hub.hf_hub_download(
            REPO, f"data/{LANG}/{split}.tsv", repo_type="dataset"
        )
        with open(path) as f:
            for r in csv.reader(f, delimiter="\t"):
                if len(r) < COLS:
                    continue  # a handful of vi_vn rows are truncated upstream
                rows.append(
                    {
                        "id": f"fleurs_{split}_{r[0]}",
                        "split": split,
                        "text": unicodedata.normalize("NFC", r[2]).strip(),
                        "reference": unicodedata.normalize("NFC", r[3]).strip(),
                        "duration": int(r[5]) / FLEURS_SR,
                        "gender": r[6],
                        # Kept so the recording can be pulled for voice-prompt or
                        # speaker-similarity work; this file itself is text-only.
                        "audio": f"data/{LANG}/audio/{split}.tar.gz::{r[1]}",
                    }
                )
    seen: set[str] = set()
    uniq = [d for d in rows if not (d["text"] in seen or seen.add(d["text"]))]

    rng = random.Random(seed)
    picked = []
    for gender in ("MALE", "FEMALE"):
        pool = [d for d in uniq if d["gender"] == gender]
        if len(pool) < per_gender:
            logger.warning(f"only {len(pool)} {gender} sentences available, wanted {per_gender}")
        picked += rng.sample(pool, min(per_gender, len(pool)))
    rng.shuffle(picked)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for d in picked:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info(
        f"{out}: {len(picked)} sentences from a {len(uniq)}-sentence pool "
        f"({len(rows)} recordings) | gender {dict(Counter(d['gender'] for d in picked))} "
        f"| source splits {dict(Counter(d['split'] for d in picked))}"
    )


if __name__ == "__main__":
    app()
