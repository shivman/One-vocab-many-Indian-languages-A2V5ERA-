#!/usr/bin/env python3
"""
8-language BPE tokenizer — trained from scratch (no HuggingFace / SentencePiece).

Core Wikipedia India: English, Hindi, Telugu, Tamil, Bengali, Marathi
Uttarakhand Pahari (PahariLI corpus): Garhwali, Kumaoni

Target: vocab ≤ 10_000, English fertility ≤ 1.2,
maximize score = 1000 / (Xmax - Xmin) across all 8 languages.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT

# 8 languages
LANGS = ["en", "hi", "te", "ta", "bn", "mr", "gbm", "kfy"]
LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "bn": "Bengali",
    "mr": "Marathi",
    "gbm": "Garhwali",
    "kfy": "Kumaoni",
}
LANG_SCRIPT = {
    "en": "Latin",
    "hi": "Devanagari",
    "te": "Telugu",
    "ta": "Tamil",
    "bn": "Bengali",
    "mr": "Devanagari",
    "gbm": "Devanagari",
    "kfy": "Devanagari",
}
LANG_SOURCE = {
    "en": "Wikipedia India",
    "hi": "Wikipedia India",
    "te": "Wikipedia India",
    "ta": "Wikipedia India",
    "bn": "Wikipedia India",
    "mr": "Wikipedia India",
    "gbm": "PahariLI corpus (no Wikipedia edition)",
    "kfy": "PahariLI corpus (no Wikipedia edition)",
}
CORE4 = ["en", "hi", "te", "ta"]  # original assignment languages
VOCAB_SIZE = 10_000

# Tuned for tight fertility cluster with EN ≤ 1.2 (8-language mix)
LANG_WEIGHT = {
    "en": 9.0,
    "hi": 0.8,
    "te": 0.9,
    "ta": 0.9,
    "bn": 0.95,
    "mr": 0.95,
    "gbm": 0.65,
    "kfy": 0.65,
}

WORD_RE = re.compile(r"\w+", re.UNICODE)


def load_text(lang: str) -> str:
    return (DATA / f"india_{lang}.txt").read_text(encoding="utf-8")


def split_units(word: str) -> list[str]:
    return list(word)


class BPETokenizer:
    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}

    def _add_token(self, tok: str) -> int:
        if tok in self.token_to_id:
            return self.token_to_id[tok]
        tid = len(self.token_to_id)
        self.token_to_id[tok] = tid
        self.id_to_token[tid] = tok
        return tid

    def train(self, word_freqs: dict[str, float], vocab_size: int = VOCAB_SIZE, log_every: int = 500) -> None:
        chars: set[str] = set()
        for w in word_freqs:
            chars.update(split_units(w))
        for ch in sorted(chars):
            self._add_token(ch)
        for special in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            self._add_token(special)

        corpus: dict[tuple[str, ...], float] = {}
        for w, freq in word_freqs.items():
            units = tuple(split_units(w))
            if units:
                corpus[units] = corpus.get(units, 0.0) + freq

        def count_pairs(corp: dict[tuple[str, ...], float]) -> Counter:
            pairs: Counter = Counter()
            for seq, freq in corp.items():
                for i in range(len(seq) - 1):
                    pairs[(seq[i], seq[i + 1])] += freq
            return pairs

        def apply_merge(corp: dict[tuple[str, ...], float], a: str, b: str) -> dict[tuple[str, ...], float]:
            merged = a + b
            new_corp: dict[tuple[str, ...], float] = {}
            for seq, freq in corp.items():
                out: list[str] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                key = tuple(out)
                new_corp[key] = new_corp.get(key, 0.0) + freq
            return new_corp

        t0 = time.time()
        step = 0
        while len(self.token_to_id) < vocab_size:
            pairs = count_pairs(corpus)
            if not pairs:
                break
            (a, b), _ = pairs.most_common(1)[0]
            merged = a + b
            if merged in self.token_to_id:
                found = False
                for (pa, pb), _cnt in pairs.most_common(30):
                    if pa + pb not in self.token_to_id:
                        a, b = pa, pb
                        merged = a + b
                        found = True
                        break
                if not found:
                    break

            self.merges.append((a, b))
            self.merge_ranks[(a, b)] = len(self.merges) - 1
            self._add_token(merged)
            corpus = apply_merge(corpus, a, b)
            step += 1
            if step % log_every == 0:
                print(
                    f"  merge {step}: vocab={len(self.token_to_id)} "
                    f"pair=({a!r}+{b!r}) elapsed={time.time()-t0:.1f}s"
                )

        print(
            f"Training done: vocab={len(self.token_to_id)} merges={len(self.merges)} "
            f"in {time.time()-t0:.1f}s"
        )

    def encode_word(self, word: str) -> list[str]:
        tokens = split_units(word)
        if not tokens:
            return []
        while True:
            best_rank = None
            best_i = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i is None:
                break
            a, b = tokens[best_i], tokens[best_i + 1]
            tokens = tokens[:best_i] + [a + b] + tokens[best_i + 2 :]
        return tokens

    def encode_text(self, text: str) -> list[str]:
        out: list[str] = []
        for w in WORD_RE.findall(text):
            for p in self.encode_word(w):
                if p in self.token_to_id:
                    out.append(p)
                else:
                    for ch in p:
                        out.append(ch if ch in self.token_to_id else "<UNK>")
        return out

    def fertility(self, text: str) -> dict:
        words = WORD_RE.findall(text)
        tokens = self.encode_text(text)
        n_words = len(words)
        n_tokens = len(tokens)
        unk = sum(1 for t in tokens if t == "<UNK>")
        return {
            "words": n_words,
            "unique_words": len(set(words)),
            "tokens": n_tokens,
            "fertility": (n_tokens / n_words) if n_words else 0.0,
            "unk": unk,
        }

    def to_json(self) -> dict:
        return {
            "model": "bpe",
            "vocab_size": len(self.token_to_id),
            "languages": LANGS,
            "language_names": LANG_NAMES,
            "language_scripts": LANG_SCRIPT,
            "language_sources": LANG_SOURCE,
            "token_to_id": self.token_to_id,
            "merges": [f"{a} {b}" for a, b in self.merges],
            "special_tokens": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            "version": "2.0",
            "notes": (
                "From-scratch 8-language BPE. Wikipedia India (en/hi/te/ta/bn/mr) + "
                "PahariLI Garhwali/Kumaoni (no Wikipedia editions)."
            ),
        }


def build_weighted_freqs() -> dict[str, float]:
    freqs: dict[str, float] = defaultdict(float)
    stats = {}
    for lang in LANGS:
        text = load_text(lang)
        words = WORD_RE.findall(text)
        c = Counter(words)
        w = LANG_WEIGHT[lang]
        total = sum(c.values()) or 1
        for word, cnt in c.items():
            freqs[word] += (cnt / total) * w * 100_000
        stats[lang] = {
            "name": LANG_NAMES[lang],
            "words": len(words),
            "unique": len(c),
            "weight": w,
            "source": LANG_SOURCE[lang],
            "script": LANG_SCRIPT[lang],
        }
    print("Corpus stats:", json.dumps(stats, ensure_ascii=False, indent=2))
    return dict(freqs)


def score_from(results: dict, langs: list[str]) -> dict:
    fertilities = [results[l]["fertility"] for l in langs]
    xmin, xmax = min(fertilities), max(fertilities)
    gap = xmax - xmin
    score = 1000.0 / gap if gap > 1e-12 else float("inf")
    return {
        "langs": langs,
        "Xmin": xmin,
        "Xmax": xmax,
        "gap": gap,
        "score": score,
        "english_ok": results["en"]["fertility"] <= 1.2,
        "unk_total": sum(results[l]["unk"] for l in langs),
    }


def main():
    print("Building weighted word frequencies (8 languages)...")
    freqs = build_weighted_freqs()
    print(f"Unique word types in training mix: {len(freqs):,}")

    tok = BPETokenizer()
    print(f"Training BPE to vocab_size={VOCAB_SIZE}...")
    tok.train(freqs, vocab_size=VOCAB_SIZE)

    results = {}
    print("\n=== Fertility evaluation ===")
    for lang in LANGS:
        text = load_text(lang)
        m = tok.fertility(text)
        results[lang] = m
        print(
            f"  {LANG_NAMES[lang]:10s}  X={m['fertility']:.6f}  "
            f"tokens={m['tokens']:,}  words={m['words']:,}  unk={m['unk']}"
        )

    score8 = score_from(results, LANGS)
    score4 = score_from(results, CORE4)
    score6 = score_from(results, ["en", "hi", "te", "ta", "bn", "mr"])

    print(f"\n8-lang  score={score8['score']:.2f}  gap={score8['gap']:.6f}  EN_ok={score8['english_ok']}")
    print(f"6-wiki  score={score6['score']:.2f}  gap={score6['gap']:.6f}")
    print(f"4-core  score={score4['score']:.2f}  gap={score4['gap']:.6f}")

    payload = tok.to_json()
    payload["metrics"] = {
        "per_language": {
            lang: {
                "name": LANG_NAMES[lang],
                "script": LANG_SCRIPT[lang],
                "source": LANG_SOURCE[lang],
                **results[lang],
                "ratio_label": f"X{i+1}",
            }
            for i, lang in enumerate(LANGS)
        },
        "sorted_ratios": sorted(
            [{"lang": l, "name": LANG_NAMES[l], "X": results[l]["fertility"]} for l in LANGS],
            key=lambda d: -d["X"],
        ),
        "score_8lang": score8,
        "score_6wiki": score6,
        "score_4core": score4,
        # Primary reported score = all 8 (the ambitious build)
        "Xmin": score8["Xmin"],
        "Xmax": score8["Xmax"],
        "gap": score8["gap"],
        "score": score8["score"],
        "english_ok": score8["english_ok"],
        "unk_total": score8["unk_total"],
        "vocab_size": len(tok.token_to_id),
        "formula": "X = total_BPE_tokens / total_words ; score = 1000 / (Xmax - Xmin)",
        "word_regex": r"\w+",
        "training_weights": LANG_WEIGHT,
        "core4": CORE4,
    }

    out_path = OUT / "tokenizer.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")

    (OUT / "metrics.json").write_text(
        json.dumps(payload["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
