"""FLEURS Vietnamese TTS eval: intelligibility (WER/CER) on an EXTERNAL test set.

    python -m training.eval.fleurs_vi runs/vi_finetune_language \
        --checkpoint runs/vi_finetune_language/keep/checkpoint_00027500.pt --limit 200

For each sentence in data/eval_vi/fleurs_vi.jsonl: clone a fixed voice prompt,
synthesize the sentence, transcribe it with a Vietnamese ASR, and score the
transcript against FLEURS' own normalized reference with jiwer.

Why this and not the run's own valid loss: the valid split holds out utterances
whose document- and speaker-neighbours are in train, and flow_diag is a
denoising error, not a perceptual one. FLEURS shares no speaker, domain or
pipeline with the training corpus, and WER answers the question the loss
cannot -- can a listener recover the words.

The voice prompt is drawn from a manifest (the run's own valid split by
default) and assigned round-robin from a small fixed pool, so the prompt is
identical for every checkpoint compared. Two evals of two checkpoints then
differ only by the model, which is the whole point of running it twice.

Vietnamese normalization is NFC + casefold + punctuation strip, applied to both
sides. FLEURS' `reference` column already ships in that form; this makes the
hypothesis match it rather than trusting the ASR's punctuation.
"""

import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path

import jiwer
import sphn
import torch

from training.eval.librispeech import MIN_FRAMES, latents_to_wav, load_mono, load_run
from training.eval.vi_itn import canon_hypothesis, canon_reference

logger = logging.getLogger("eval_fleurs_vi")

# Vietnamese-finetuned Whisper. Plain whisper-large-v3 also works (--asr) but
# scores several points worse on Vietnamese, which would flatter the TTS.
DEFAULT_ASR = "vinai/PhoWhisper-medium"
DEFAULT_EVAL = "data/eval_vi/fleurs_vi.jsonl"

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

# Sentences whose INPUT carries written forms the model must verbalize: digits
# and ALL-CAPS abbreviations. FLEURS' reference keeps the written form -- its
# normalizer only strips punctuation, so "10:00 - 11:00" becomes the four
# tokens "10 00 11 00" -- while a correct model says "mười giờ ..." and the ASR
# transcribes what it hears. Every correctly spoken word then scores as an
# error. On vi_vn this hits 108 of 500 sentences and inflates corpus WER by
# roughly half, so the two populations are reported apart rather than pooled
# into one misleading number.
_WRITTEN = re.compile(r"\d|\b[A-ZĐ]{2,}\b")


def has_written_form(text: str) -> bool:
    return bool(_WRITTEN.search(text))


def build_vi_transcriber(asr_name: str, device):
    """Vietnamese Whisper decoding, hardened against the looping failure.

    librispeech.py's transcriber asks for timestamps, which whisper needs for
    long-form audio but which makes it hallucinate on short clips: it emits the
    same clause over and over and never predicts an end timestamp. Three of the
    first sixteen items came back at 7-25 words per second, against the 3-5 that
    Vietnamese speech actually runs at, and those three alone moved corpus WER
    from 20% to 96%. So: no timestamps, language pinned to Vietnamese rather
    than detected per clip, greedy, and a token budget scaled to the clip's
    duration so a loop is truncated instead of counted.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(asr_name)
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(asr_name, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    def transcribe(wavs: list) -> list[str]:
        inputs = processor(
            wavs, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        feats = inputs.input_features.to(device, torch.float16)
        # ~4 words/s at ~2 tokens/word, tripled for headroom: generous for real
        # speech, still a hard ceiling on a loop.
        budget = int(max(len(w) for w in wavs) / 16000 * 24) + 16
        with torch.no_grad():
            ids = model.generate(
                feats,
                attention_mask=inputs.attention_mask.to(device),
                language="vi",
                task="transcribe",
                return_timestamps=False,
                do_sample=False,
                num_beams=1,
                max_new_tokens=min(budget, 440),
            )
        return processor.batch_decode(ids, skip_special_tokens=True)

    return transcribe


def normalize_vi(text: str) -> str:
    """NFC, casefold, drop punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(_PUNCT.sub(" ", text).casefold().split())


def load_prompts(manifest: Path, n: int, min_sec: float, seed: int) -> list[str]:
    """`n` voice-prompt paths, one per distinct speaker, deterministic."""
    import random

    by_speaker: dict[str, list[dict]] = {}
    with open(manifest) as f:
        for line in f:
            d = json.loads(line)
            if d["duration"] >= min_sec:
                by_speaker.setdefault(d.get("speaker", "?"), []).append(d)
    rng = random.Random(seed)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    return [rng.choice(by_speaker[s])["path"] for s in speakers[:n]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--checkpoint", default=None, help="defaults to the run's latest")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--eval-jsonl", default=DEFAULT_EVAL)
    p.add_argument("--prompts-from", default="data/vieneu_140h/valid_aligned.jsonl")
    p.add_argument("--num-prompts", type=int, default=4, help="voices cycled over the sentences")
    p.add_argument("--voice-sec", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--asr", default=DEFAULT_ASR)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--temp", type=float, default=0.3)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--n-steps", type=int, default=1)
    p.add_argument("--eos-threshold", type=float, default=-1.0)
    p.add_argument("--max-sec", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-itn",
        action="store_true",
        help="skip inverse text normalization of the written-form subset -- those sentences then "
        "keep their verbalization mismatch and their WER is not meaningful (see training/eval/vi_itn.py)",
    )
    p.add_argument("--save-audio", default=None)
    p.add_argument("--out", default=None, help="results json (default: <run_dir>/fleurs_vi.json)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")

    device = torch.device(args.device)
    items = [json.loads(line) for line in open(args.eval_jsonl)]
    if args.limit:
        items = items[: args.limit]
    prompts = load_prompts(Path(args.prompts_from), args.num_prompts, args.voice_sec, args.seed)
    assert prompts, f"no voice prompts >= {args.voice_sec}s in {args.prompts_from}"
    logger.info(f"{len(items)} sentences, {len(prompts)} voice prompts, asr={args.asr}")

    model, mimi, step = load_run(
        args.run_dir, device, use_ema=args.use_ema, checkpoint=args.checkpoint
    )
    torch.manual_seed(args.seed)
    import sentencepiece as spm

    from training.args import load_args
    from training.modules.builders import load_model_config

    cfg = load_model_config(
        str(load_args(args.run_dir / "args.yaml").model_config),
        load_args(args.run_dir / "args.yaml").model_overrides,
    )
    sp = spm.SentencePieceProcessor(model_file=str(cfg.flow_lm.lookup_table.tokenizer_path))

    max_frames = int(args.max_sec * mimi.frame_rate)
    records = []
    voice_cache: dict[str, torch.Tensor] = {}
    for start in range(0, len(items), args.batch_size):
        chunk = items[start : start + args.batch_size]
        tokens = [torch.tensor(sp.encode(c["text"]), dtype=torch.long) for c in chunk]
        with torch.no_grad():
            latents = []
            for i, _c in enumerate(chunk):
                path = prompts[(start + i) % len(prompts)]
                if path not in voice_cache:
                    wav = load_mono(path, mimi.sample_rate)[
                        : int(args.voice_sec * mimi.sample_rate)
                    ]
                    voice_cache[path] = mimi.encode_to_latent(wav[None, None].to(device))[0]
                latents.append(voice_cache[path])
            outs = model.generate(
                tokens,
                latents,
                max_frames=max_frames,
                temp=args.temp,
                n_steps=args.n_steps,
                cfg_coef=args.cfg,
                eos_threshold=args.eos_threshold,
            )
        for i, (item, lat) in enumerate(zip(chunk, outs, strict=True)):
            rec = {
                "id": item["id"],
                "written_form": has_written_form(item["text"]),
                "raw": item["text"],
                "ref": normalize_vi(item["reference"]),
                "hyp": "",
                "silent": 0,
                "no_eos": int(lat.shape[0] >= max_frames),
                "gen_sec": float(lat.shape[0] / mimi.frame_rate),
            }
            if lat.shape[0] < MIN_FRAMES:
                rec["silent"] = 1
                records.append(rec)
                continue
            with torch.no_grad():
                audio = latents_to_wav(mimi, lat, device)
            rec["_wav"] = audio.cpu().numpy()
            if args.save_audio:
                Path(args.save_audio).mkdir(parents=True, exist_ok=True)
                sphn.write_wav(
                    f"{args.save_audio}/{start + i:04d}_{item['id']}.wav",
                    rec["_wav"],
                    int(mimi.sample_rate),
                )
            records.append(rec)
        logger.info(f"generated {min(start + args.batch_size, len(items))}/{len(items)}")

    transcribe = build_vi_transcriber(args.asr, device)
    live = [r for r in records if "_wav" in r]
    for s in range(0, len(live), args.batch_size):
        batch = live[s : s + args.batch_size]
        wavs = [
            sphn.resample(
                r.pop("_wav"), src_sample_rate=int(mimi.sample_rate), dst_sample_rate=16000
            )
            for r in batch
        ]
        for r, hyp in zip(batch, transcribe(wavs), strict=True):
            r["hyp"] = normalize_vi(hyp)
        logger.info(f"transcribed {min(s + args.batch_size, len(live))}/{len(live)}")

    refs = [r["ref"] for r in records]
    hyps = [r["hyp"] for r in records]
    clean = [r for r in records if not r["written_form"]]
    written = [r for r in records if r["written_form"]]

    def corpus(rs, ref_key="ref", hyp_key="hyp"):
        return jiwer.wer([r[ref_key] for r in rs], [r[hyp_key] for r in rs]) if rs else None

    # Canonicalize the written-form subset so any valid reading of a number,
    # date, time or spelled abbreviation scores as correct.
    if written and not args.no_itn:
        logger.info(f"inverse-normalizing {len(written)} written-form sentences")
        for r in written:
            r["ref_itn"] = canon_reference(r["raw"])
            r["hyp_itn"] = canon_hypothesis(r["hyp"])
    pooled_ref = [r["ref"] for r in clean] + [r.get("ref_itn", r["ref"]) for r in written]
    pooled_hyp = [r["hyp"] for r in clean] + [r.get("hyp_itn", r["hyp"]) for r in written]

    out = {
        "step": step,
        "checkpoint": str(args.checkpoint or "latest"),
        "asr": args.asr,
        "num_items": len(records),
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        # The headline number: sentences with nothing to verbalize, so a word
        # error is a real one.
        "wer_clean": corpus(clean),
        "cer_clean": jiwer.cer([r["ref"] for r in clean], [r["hyp"] for r in clean])
        if clean
        else None,
        "n_clean": len(clean),
        "wer_written": corpus(written),
        "wer_written_itn": corpus(written, "ref_itn", "hyp_itn")
        if written and not args.no_itn
        else None,
        "n_written": len(written),
        # The fair whole-corpus number: clean sentences as-is, written-form
        # sentences after inverse normalization.
        "wer_overall": jiwer.wer(pooled_ref, pooled_hyp),
        "silent": sum(r["silent"] for r in records),
        "no_eos": sum(r["no_eos"] for r in records),
        "mean_gen_sec": sum(r["gen_sec"] for r in records) / max(len(records), 1),
        "settings": {
            k: getattr(args, k) for k in ("temp", "cfg", "n_steps", "eos_threshold", "seed")
        },
        "items": records,
    }
    dest = Path(args.out or (args.run_dir / "fleurs_vi.json"))
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    fmt = lambda v: f"{v:.2%}" if v is not None else "n/a"  # noqa: E731
    logger.info(
        f"step {step}: WER {fmt(out['wer_overall'])} overall "
        f"({fmt(out['wer_clean'])} on {out['n_clean']} clean, "
        f"{fmt(out['wer_written_itn'])} on {out['n_written']} written-form after ITN, "
        f"was {fmt(out['wer_written'])} before)  CER {out['cer']:.2%}  "
        f"silent {out['silent']}  no_eos {out['no_eos']}  -> {dest}"
    )


if __name__ == "__main__":
    main()
