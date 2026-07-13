# India Wiki BPE — Faithful Markdown Fertility

Shared **10,000**-token BPE for India Wikipedia in **English · Hindi · Telugu · Maithili**.

Built to pass the official ERA V5 Assignment 2 grader:

```text
decode(encode(text)) preserves visible non-whitespace characters
fertility = tokens / faithful_units
score = 1000 / (Xmax − Xmin)
```

Tokenizer file is **HuggingFace `tokenizers`** format (`Tokenizer.from_file` works).

---

## Results (this submission)

| Language | Tokens | Faithful units | Fertility X |
|----------|-------:|---------------:|------------:|
| Telugu | 23,792 | 36,292 | **0.6556** |
| English | 120,569 | 186,367 | **0.6469** |
| Maithili | 3,407 | 5,808 | **0.5866** |
| Hindi | 47,311 | 88,359 | **0.5354** |

```text
Spread = 0.12013
Raw score ≈ 8324
Hindi penalty = 1.0  (HI ≤ 1.2)
Hindi-adjusted ≈ 8324
```

All four fertilities **≤ 1.2**.  
Grader sample `India's population is 1,428,627,663.` → **exact** `decode(encode(...))` match.

---

## Live / download

| File | Role |
|------|------|
| `index.html` | Netlify widget — score, bars, faithful proof |
| `tokenizer.json` | HF BPE + Metaspace (grader-loadable) |
| `metrics.json` | Official-style metrics |
| `faith_proof.json` | Round-trip evidence for the grader sample |
| `evaluate_tokenizer.py` | Re-score locally |
| `train_tokenizer.py` | Retrain |
| `build_wiki_faithful_markdown.py` | Refresh Wikipedia → faithful Markdown |
| `corpus/*.faithful.txt` | Snapshot corpus used for metrics |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install tokenizers regex requests beautifulsoup4 lxml markdownify
python evaluate_tokenizer.py
python -m http.server 8080
```

---

## Training choices

- Model: HuggingFace **BPE**
- Vocab: **10,000**
- Normalizer: **NFKC**
- Pre-tokenizer / decoder: **Metaspace (`▁`)**
- Weights: `{"en": 2, "hi": 5, "te": 4, "mai": 5}` (Indic/Maithili upweighted to shrink the gap)

Metaspace (not ByteLevel) keeps punctuation & spaces while avoiding UTF-8 byte blow-ups on Indic scripts.

---

## Faithfulness

```text
decode(encode(text)) must keep the same non-whitespace characters as text
```

Stripping apostrophes / commas / URL chars invalidates fertility claims. This tokenizer round-trips the official sample including spaces.

---

## Axiom submit

1. Deploy this folder to Netlify  
2. Widget URL → site root  
3. Tokenizer URL → `https://YOUR-SITE.netlify.app/tokenizer.json`  
4. Test both in Incognito  

Resubmission window: open until Saturday (no late penalty applied in that window).
