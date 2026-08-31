"""Canonicalize Vietnamese so any valid reading of a written form scores as correct.

A TTS given "10:00 – 11:00 … MDT" says it aloud; the ASR transcribes what it
hears; the reference keeps the written form. Every correctly spoken word then
counts as an error. Worse, there is no single correct reading — "2020" is
validly `hai nghìn không trăm hai mươi` or `hai không hai không` — so no choice
of reference string makes exact-match WER fair.

The fix is to canonicalize the hypothesis back to written form rather than to
pick a winner among readings. NeMo's Vietnamese inverse text normalizer maps
every valid reading of a number, date, time, or percentage onto the same written
token, which is exactly the equivalence exact match is missing.

NeMo does not cover letter-spelled abbreviations, and a timezone like MDT is
read out letter by letter, so `collapse_letters` folds a run of Vietnamese
letter names back into the string it spells.

On the 108 written-form sentences of the FLEURS vi_vn set this took the 1,061h
model from 28.98% to 19.49% WER, and individual items from 133% to 17% — while
leaving genuinely garbled generations high, which is the behaviour you want from
a scorer.
"""

import re
import unicodedata
from functools import lru_cache

# Vietnamese letter names, as ASR tends to spell them out.
LETTER_NAMES = {
    "a": "a",
    "bê": "b",
    "bờ": "b",
    "xê": "c",
    "cê": "c",
    "sê": "c",
    "cờ": "c",
    "dê": "d",
    "dờ": "d",
    "di": "d",
    "đê": "đ",
    "đờ": "đ",
    "e": "e",
    "ép": "f",
    "épxì": "f",
    "phờ": "f",
    "giê": "g",
    "gờ": "g",
    "hát": "h",
    "hắt": "h",
    "hờ": "h",
    "i": "i",
    "ca": "k",
    "ka": "k",
    "elờ": "l",
    "lờ": "l",
    "emmờ": "m",
    "mờ": "m",
    "em": "m",
    "ennờ": "n",
    "nờ": "n",
    "en": "n",
    "o": "o",
    "ô": "o",
    "pê": "p",
    "pờ": "p",
    "quy": "q",
    "cu": "q",
    "erờ": "r",
    "rờ": "r",
    "ét": "s",
    "étxì": "s",
    "sờ": "s",
    "tê": "t",
    "tờ": "t",
    "ti": "t",
    "u": "u",
    "vê": "v",
    "vờ": "v",
    "ích": "x",
    "íchxì": "x",
    "xờ": "x",
    "dét": "z",
    "giét": "z",
    "y": "y",
}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


@lru_cache(maxsize=1)
def _normalizer():
    """Built once — the WFST grammars take ~30s to compile."""
    from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

    return InverseNormalizer(lang="vi")


def collapse_letters(text: str, min_run: int = 2) -> str:
    """Fold a run of >=min_run letter names into the abbreviation it spells.

    A single letter name is left alone: "em" and "ca" are ordinary Vietnamese
    words far more often than they are the letters M and K.
    """
    out: list[str] = []
    run: list[str] = []
    raw: list[str] = []

    def flush() -> None:
        if len(run) >= min_run:
            out.append("".join(LETTER_NAMES[t] for t in run))
        else:
            out.extend(raw)
        run.clear()
        raw.clear()

    for token in text.split():
        key = token.replace("-", "").casefold()
        if key in LETTER_NAMES:
            run.append(key)
            raw.append(token)
        else:
            flush()
            out.append(token)
    flush()
    return " ".join(out)


def strip_case_punct(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", unicodedata.normalize("NFC", text)).casefold().split())


def canon_hypothesis(text: str) -> str:
    """ASR output -> written form, so it can be compared against a written reference."""
    try:
        text = _normalizer().inverse_normalize(text, verbose=False)
    except Exception:  # noqa: BLE001 -- a grammar miss must not lose the utterance
        pass
    return strip_case_punct(collapse_letters(text))


def canon_reference(text: str) -> str:
    """The written reference needs only casing and punctuation flattened."""
    return strip_case_punct(text)
