#!/usr/bin/env python3
"""
Train the shared 10k BPE tokenizer for the wiki-faithful Markdown corpus.

Compatible with the official ERA V5 Assignment 2 evaluator:
  Tokenizer.from_file(tokenizer.json)  # HuggingFace tokenizers
  fertility = token_count / faithful_unit_count
  score = 1000 / (max_fertility - min_fertility)

Run:
    python build_wiki_faithful_markdown.py   # optional refresh
    python train_tokenizer.py
    python evaluate_tokenizer.py
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import regex
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
OUT_TOKENIZER = ROOT / "tokenizer.json"
OUT_METRICS = ROOT / "metrics.json"
OUT_PROOF = ROOT / "faith_proof.json"

LANGS = ["en", "hi", "te", "mai"]
LANG_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "mai": "Maithili"}
# Heavier Indic / Maithili to tighten fertility gap vs English
WEIGHTS = {"en": 2, "hi": 5, "te": 4, "mai": 5}
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
GRADER_SAMPLE = "India's population is 1,428,627,663."


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def make_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="never")
    return tokenizer


def visible(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def train() -> tuple[Tokenizer, dict]:
    texts = {
        code: (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        for code in LANGS
    }
    units = {code: faithful_units(text) for code, text in texts.items()}

    with tempfile.TemporaryDirectory() as tmp:
        files: list[str] = []
        tmpdir = Path(tmp)
        for code, text in texts.items():
            path = tmpdir / f"{code}.txt"
            path.write_text(text, encoding="utf-8")
            files.extend([str(path)] * WEIGHTS[code])

        tokenizer = make_tokenizer()
        trainer = BpeTrainer(
            vocab_size=10000,
            min_frequency=1,
            special_tokens=["[UNK]"],
        )
        tokenizer.train(files, trainer)

    token_counts = {code: len(tokenizer.encode(text).ids) for code, text in texts.items()}
    ratios = {code: token_counts[code] / units[code] for code in LANGS}
    spread = max(ratios.values()) - min(ratios.values())
    score = 1000 / spread
    hindi_penalty = math.exp(max(0.0, ratios["hi"] / 1.2 - 1.0))

    # Faithfulness proof for widget / graders
    enc = tokenizer.encode(GRADER_SAMPLE)
    dec = tokenizer.decode(enc.ids)
    proof = {
        "sample": GRADER_SAMPLE,
        "decoded": dec,
        "ids": enc.ids,
        "tokens": enc.tokens,
        "full_roundtrip_equal": dec == GRADER_SAMPLE,
        "visible_non_ws_equal": visible(dec) == visible(GRADER_SAMPLE),
    }

    metrics = {
        "variant": "wiki_faithful_markdown",
        "languages": LANG_NAMES,
        "weights": WEIGHTS,
        "vocab_size": tokenizer.get_vocab_size(),
        "faithful_units": units,
        "unit_policy": (
            "Counts each contiguous Unicode letter/mark/number run as one unit "
            "and each visible non-space punctuation/symbol character as one unit."
        ),
        "token_counts": token_counts,
        "ratios": ratios,
        "fertilities": ratios,
        "spread": spread,
        "score": score,
        "hindi_exp1_penalty_factor": hindi_penalty,
        "hindi_exp1_adjusted_score": score / hindi_penalty,
        "english_ok": ratios["en"] <= 1.2,
        "hindi_ok": ratios["hi"] <= 1.2,
        "all_under_1_2": all(r <= 1.2 for r in ratios.values()),
        "formula": "fertility = tokens / faithful_units ; score = 1000 / (Xmax - Xmin)",
        "codec": {
            "library": "HuggingFace tokenizers",
            "model": "BPE",
            "normalizer": "NFKC",
            "pre_tokenizer": "Metaspace(▁)",
            "decoder": "Metaspace(▁)",
            "faithful_requirement": "decode(encode(text)) preserves visible non-whitespace (and spaces via Metaspace)",
        },
        "faith_proof": proof,
        "sorted_ratios": sorted(
            [{"lang": k, "name": LANG_NAMES[k], "X": ratios[k]} for k in LANGS],
            key=lambda d: -d["X"],
        ),
    }
    return tokenizer, metrics, proof


def main() -> int:
    tokenizer, metrics, proof = train()
    assert proof["visible_non_ws_equal"], "Failed faithful visible round-trip"
    assert proof["full_roundtrip_equal"], "Failed full string round-trip on grader sample"
    tokenizer.save(str(OUT_TOKENIZER))
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_PROOF.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nGrader sample round-trip:", proof["decoded"])
    print("PASS faithful decode(encode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
