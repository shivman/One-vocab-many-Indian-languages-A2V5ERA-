# India BPE Fairness

**One shared vocabulary. Eight Indian languages. Measure fairness by fertility gap.**

A from-scratch **Byte-Pair Encoding (BPE)** tokenizer — no HuggingFace, no SentencePiece — trained so English and Indic languages stay close in how finely words are split.

Live idea: if a shared vocab is unfair, some languages shatter into many tokens while others stay whole. We measure that with **fertility** and score how tight the gap is.

---

## Try it

| Artifact | What it is |
|----------|------------|
| `index.html` | Interactive widget — Core-4 / All-8 modes, compare, live encode, download |
| `tokenizer.json` | Full vocab + ordered merge rules (public download for verification) |
| `metrics.json` | Fertility table + scores |
| `scripts/train_bpe.py` | Trainer (pure Python) |
| `LEARNING.md` | Concepts, design choices, and how to read the results |

```bash
python3 -m http.server 8080
# open http://localhost:8080
```

---

## Results

| Evaluation scope | Score `1000 / (Xmax − Xmin)` | Gap |
|------------------|-----------------------------:|----:|
| **Core-4** (EN · HI · TE · TA) | **~18,896** | 0.053 |
| **All-8** (+ BN · MR · Garhwali · Kumaoni) | **~15,498** | 0.065 |

| Language | Fertility X | Source |
|----------|------------:|--------|
| English | **1.112** (≤ 1.2) | Wikipedia · India |
| Bengali | 1.103 | Wikipedia · India |
| Tamil | 1.083 | Wikipedia · India |
| Telugu | 1.061 | Wikipedia · India |
| Hindi | 1.059 | Wikipedia · India |
| Garhwali | 1.050 | PahariLI corpus |
| Marathi | 1.048 | Wikipedia · India |
| Kumaoni | 1.048 | PahariLI corpus |

- Shared vocab size: **10,000**
- UNK on eval text: **0**
- Formula: `X = BPE_tokens / words` (Unicode `\w+`)

**Why two scores?**  
Core-4 is the clean baseline. All-8 is the harder fairness stress test — including Garhwali & Kumaoni, which have **no Wikipedia language edition**.

---

## What’s inside `tokenizer.json`

- `token_to_id` — shared vocabulary  
- `merges` — ordered BPE merge rules  
- `metrics` — fertilities + Core-4 / All-8 scores  
- Enough for anyone to re-encode text and verify claims  

After Netlify deploy, the public file URL looks like:

`https://YOUR-SITE.netlify.app/tokenizer.json`

---

## Retrain

```bash
python3 scripts/train_bpe.py
```

Language weights and data live under `data/` and `scripts/train_bpe.py`.

---

## Design in one paragraph

Start from characters → repeatedly merge the most frequent adjacent pair → stop at 10k tokens. Word frequencies are **normalized per language**, then reweighted so English fertility stays ≤ 1.2 while Indic + Pahari fertilities stay close — maximizing `1000 / (Xmax − Xmin)`.

---

## License

Use freely for learning and demos. Wikipedia text © respective contributors; Garhwali/Kumaoni sentences from the [PahariLI](https://github.com/rachana2010/PahariLI) corpus.
