"""Quality-check a training manifest before spending GPU-hours on it.

    python -m training.scripts.check_manifest data/vieneu_140h/train_aligned.jsonl

Reports the things that actually predict a bad TTS run, loudest first. Nothing
here is corpus-specific, so it reads a HiFiTTS-2 manifest and a bring-your-own
one the same way, which is the point -- the numbers are only meaningful next to
another corpus you already trust.

The checks, and why each one is here:

- speaking rate: characters per second of aligned speech. A transcript that
  does not match its audio shows up here and almost nowhere else, because the
  text is plausible and the audio is clean -- only their ratio is wrong. The
  tails are what matter, not the mean.
- alignment: the DataLoader cuts each utterance at a word boundary to split
  voice prompt from target. Utterances whose alignment yields no eligible cut
  fall back to a random-window prompt, which is a weaker training signal, so
  their share is worth knowing.
- trailing silence: the loader trims to the last word plus TRAIL_SEC. Long
  trailing silence that the alignment did NOT catch is what teaches a model to
  emit silence instead of EOS.
- duplicates and speaker balance: a corpus that is 40% one voice, or that
  repeats one sentence hundreds of times, trains a model that is good at
  exactly that.
- audio: clipping, DC offset, near-silent files, and level spread, on a random
  sample. Mimi encodes whatever it is given.
"""

import json
import logging
import random
import re
import statistics
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import numpy as np
import sphn
import typer
from tqdm import tqdm

logger = logging.getLogger("check_manifest")

app = typer.Typer(pretty_exceptions_show_locals=False)

# Mirrors training.dataloader.DataLoader, which is what these numbers predict.
MIN_CUT_SEC = 1.0
TRAIL_SEC = 0.2


def pct(part: int, whole: int) -> str:
    return f"{part} ({part / max(whole, 1):.2%})"


def quantiles(xs: list[float], qs=(0.001, 0.01, 0.5, 0.99, 0.999)) -> str:
    s = sorted(xs)
    return "  ".join(f"p{q * 100:g}={s[min(int(q * len(s)), len(s) - 1)]:.2f}" for q in qs)


def eligible_cuts(words: list, duration: float, window: float) -> int:
    """How many word boundaries the DataLoader could cut this utterance at."""
    n = 0
    for i in range(1, len(words)):
        prev, cur = words[i - 1], words[i]
        if prev.get("end") is None or cur.get("start") is None:
            continue
        cut = 0.5 * (prev["end"] + cur["start"])
        if cut >= MIN_CUT_SEC and duration - cut >= MIN_CUT_SEC and prev["end"] < window:
            n += 1
    return n


def audio_stats(path: str, start: float, duration: float) -> dict | None:
    try:
        wav, sr = sphn.read(path, start_sec=start or None, duration_sec=duration if start else None)
    except Exception:  # noqa: BLE001 -- an unreadable file is itself the finding
        return None
    x = wav.mean(axis=0).astype(np.float32)
    if x.size == 0:
        return {"empty": True, "sr": sr}
    peak = float(np.abs(x).max())
    # A frame is "silent" at -50 dBFS, well under any real speech floor.
    frame = max(1, int(0.02 * sr))
    n = (x.size // frame) * frame
    rms_f = np.sqrt((x[:n].reshape(-1, frame) ** 2).mean(axis=1) + 1e-12)
    quiet = rms_f < 10 ** (-50 / 20)
    lead = int(np.argmax(~quiet)) * frame / sr if (~quiet).any() else x.size / sr
    tail = int(np.argmax(~quiet[::-1])) * frame / sr if (~quiet).any() else x.size / sr
    return {
        "sr": sr,
        "channels": wav.shape[0],
        "peak": peak,
        "clipped": float((np.abs(x) >= 0.999).mean()),
        "rms_db": float(20 * np.log10(np.sqrt((x**2).mean()) + 1e-12)),
        "dc": float(x.mean()),
        "lead_sil": lead,
        "tail_sil": tail,
        "silent_frac": float(quiet.mean()),
    }


@app.command()
def main(
    manifest: Annotated[Path, typer.Argument()],
    audio_sample: Annotated[
        int, typer.Option(help="random utterances to decode for the audio checks (0 skips)")
    ] = 300,
    voice_prompt_sec: Annotated[
        float, typer.Option(help="must match the run's data.max_voice_prompt_sec")
    ] = 5.0,
    seed: Annotated[int, typer.Option()] = 0,
    workers: Annotated[int, typer.Option()] = 16,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = [json.loads(line) for line in open(manifest)]
    n = len(rows)
    total_h = sum(r["duration"] for r in rows) / 3600
    print(f"\n=== {manifest}  |  {n} utterances, {total_h:.1f}h ===\n")

    # --- corpus shape -----------------------------------------------------
    durs = [r["duration"] for r in rows]
    print("DURATION (s)  ", quantiles(durs), f"  mean={statistics.mean(durs):.2f}")
    short = sum(d < 2 * MIN_CUT_SEC for d in durs)
    print(f"  under {2 * MIN_CUT_SEC:.0f}s (never cuttable): {pct(short, n)}")

    # --- speakers ---------------------------------------------------------
    spk = Counter(r.get("speaker", "?") for r in rows)
    spk_h = Counter()
    for r in rows:
        spk_h[r.get("speaker", "?")] += r["duration"] / 3600
    top = spk_h.most_common(5)
    print(f"\nSPEAKERS       {len(spk)} distinct")
    print(
        f"  top 5 share: {sum(h for _, h in top) / total_h:.1%}  "
        + ", ".join(f"{s}={h:.1f}h" for s, h in top)
    )
    # Voice families: the prefix before a trailing _NNNN chapter/segment index.
    fam = Counter()
    for r in rows:
        fam[re.sub(r"_\d+$", "", r.get("speaker", "?"))] += r["duration"] / 3600
    if len(fam) < len(spk):
        print(
            f"  {len(fam)} voice families once a trailing _NNNN index is stripped: "
            + ", ".join(f"{f}={h:.0f}h" for f, h in fam.most_common(8))
        )

    # --- text -------------------------------------------------------------
    texts = [r["transcript"] for r in rows]
    dup = Counter(texts)
    dup_n = sum(c for t, c in dup.items() if c > 1) - sum(1 for c in dup.values() if c > 1)
    print(f"\nTEXT           {len(dup)} distinct transcripts, {pct(dup_n, n)} are repeats")
    for t, c in dup.most_common(3):
        if c > 1:
            print(f"    x{c}: {t[:90]}")
    chars = Counter(c for t in texts for c in t)
    letters = {c for c in chars if c.isalpha()}
    ascii_only = sum(
        1 for t in texts for w in t.split() if w.isascii() and any(x.isalpha() for x in w)
    )
    words_total = sum(len(t.split()) for t in texts)
    print(
        f"  {len(letters)} distinct letters; {pct(ascii_only, words_total)} of words are pure ASCII"
    )
    odd = {c: k for c, k in chars.items() if not c.isalnum() and not c.isspace()}
    print(f"  punctuation/symbols: {dict(sorted(odd.items(), key=lambda kv: -kv[1])[:14])}")
    nfc = sum(unicodedata.normalize("NFC", t) != t for t in texts)
    if nfc:
        print(f"  !! {pct(nfc, n)} transcripts are not NFC-normalized")

    # --- speaking rate ----------------------------------------------------
    print("\nSPEAKING RATE (chars per second of audio)")
    rates = [(len(r["transcript"]) / r["duration"], i) for i, r in enumerate(rows)]
    print("  ", quantiles([x for x, _ in rates]))
    lo, hi = (
        sorted(x for x, _ in rates)[int(0.001 * n)],
        sorted(x for x, _ in rates)[int(0.999 * n)],
    )
    outliers = [i for x, i in rates if x < lo or x > hi]
    print(
        f"  outside p0.1-p99.9 ({lo:.1f}-{hi:.1f} c/s): {pct(len(outliers), n)}"
        "   <- transcript/audio mismatches hide here"
    )
    for i in outliers[:3]:
        r = rows[i]
        print(
            f"    {len(r['transcript']) / r['duration']:5.1f} c/s  {r['duration']:5.2f}s  "
            f"{r['transcript'][:80]}"
        )

    # --- alignment --------------------------------------------------------
    aligned = [r for r in rows if r.get("words")]
    print(f"\nALIGNMENT      {pct(len(aligned), n)} of rows carry word timings")
    if aligned:
        wt = sum(len(r["words"]) for r in aligned)
        untimed = sum(1 for r in aligned for w in r["words"] if w.get("start") is None)
        print(f"  words without a timestamp: {pct(untimed, wt)}")
        nocut = sum(
            1 for r in aligned if eligible_cuts(r["words"], r["duration"], voice_prompt_sec) == 0
        )
        print(
            f"  no eligible cut (loader falls back to a random prompt): {pct(nocut, len(aligned))}"
        )
        cov, tail = [], []
        for r in aligned:
            ends = [w["end"] for w in r["words"] if w.get("end") is not None]
            starts = [w["start"] for w in r["words"] if w.get("start") is not None]
            if ends and starts:
                cov.append((max(ends) - min(starts)) / r["duration"])
                tail.append(r["duration"] - max(ends))
        print(f"  speech coverage of file: {quantiles(cov)}")
        print(f"  trailing silence (s):    {quantiles(tail)}")
        print(
            f"    over 1s past the last word: {pct(sum(t > 1 for t in tail), len(tail))}"
            f"  (loader trims to +{TRAIL_SEC}s, so this is handled)"
        )

    # --- audio ------------------------------------------------------------
    if audio_sample > 0:
        rng = random.Random(seed)
        pick = rng.sample(rows, min(audio_sample, n))
        print(f"\nAUDIO          decoding {len(pick)} random utterances")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            stats = list(
                tqdm(
                    pool.map(
                        lambda r: audio_stats(r["path"], r.get("start", 0.0), r["duration"]), pick
                    ),
                    total=len(pick),
                    leave=False,
                )
            )
        bad = sum(s is None for s in stats)
        ok = [s for s in stats if s and not s.get("empty")]
        if bad:
            print(f"  !! {pct(bad, len(pick))} failed to decode")
        if not ok:
            return
        print(
            f"  sample rates: {dict(Counter(s['sr'] for s in ok))}  "
            f"channels: {dict(Counter(s['channels'] for s in ok))}"
        )
        print(f"  RMS dBFS:     {quantiles([s['rms_db'] for s in ok])}")
        clip = sum(s["clipped"] > 1e-4 for s in ok)
        print(
            f"  clipping >0.01% of samples: {pct(clip, len(ok))}   "
            f"peak p99={sorted(s['peak'] for s in ok)[int(0.99 * len(ok)) - 1]:.3f}"
        )
        dc = sum(abs(s["dc"]) > 1e-3 for s in ok)
        print(f"  DC offset >0.001: {pct(dc, len(ok))}")
        print(f"  silent fraction of file: {quantiles([s['silent_frac'] for s in ok])}")
        print(f"  leading silence (s):     {quantiles([s['lead_sil'] for s in ok])}")
        print(f"  trailing silence (s):    {quantiles([s['tail_sil'] for s in ok])}")
    print()


if __name__ == "__main__":
    app()
