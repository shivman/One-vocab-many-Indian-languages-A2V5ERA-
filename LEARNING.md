# Learning guide — India BPE Fairness

This note is for **you**: what we built, why it matters, and how to explain it without jargon overload.

---

## 1. The problem

One tokenizer shared across languages can be **unfair**.

If English words stay almost whole (`fertility ≈ 1.1`) but an Indic word breaks into many pieces (`fertility ≈ 2+`), the model spends more capacity on that language for the same meaning. Multilingual NLP cares about this.

**Fertility**

```
X = (total BPE tokens) / (total words)
```

Words counted with Unicode `\w+`.

**Fairness score**

```
score = 1000 / (Xmax − Xmin)
```

Smaller gap between languages → higher score.

**Hard gates we respected**

- Shared vocab ≤ 10,000  
- English fertility ≤ 1.2  
- No UNK tokens on evaluation text  

---

## 2. What we built

| Piece | Role |
|-------|------|
| From-scratch BPE | Character seed → greedy pair merges → 10k vocab |
| Language weights | Balance merges so fertilities cluster |
| Widget | Core-4 / All-8 / Compare / Takeaways + live encode |
| `tokenizer.json` | Downloadable proof for anyone to re-run |

---

## 3. Eight languages

| Code | Language | Why |
|------|----------|-----|
| en | English | Baseline; gate ≤ 1.2 |
| hi | Hindi | Major Indic · Devanagari |
| te | Telugu | Different script family |
| ta | Tamil | Large Wikipedia page · different script |
| bn | Bengali | Extra Indic · Bengali script |
| mr | Marathi | Devanagari but different morphology |
| gbm | **Garhwali** | Uttarakhand · **no Wikipedia edition** |
| kfy | **Kumaoni** | Uttarakhand · **no Wikipedia edition** |

Garhwali & Kumaoni text comes from the open **PahariLI** research corpus (not Wikipedia). That is intentional: low-resource languages are the real fairness stress test.

---

## 4. Core-4 vs All-8

| Scope | Languages | Typical score | Meaning |
|-------|-----------|--------------:|---------|
| Core-4 | EN HI TE TA | ~18.9k | Cleaner gap · stronger number |
| All-8 | + BN MR GBM KFY | ~15.5k | Broader coverage · wider gap · lower score |

**Takeaway:** expanding languages usually **hurts the score** but **improves the story**. Report both.

---

## 5. How BPE works (our implementation)

1. Split every training word into characters  
2. Count adjacent pairs (weighted by language-balanced word frequency)  
3. Merge the winning pair into one new token; save the rule  
4. Repeat until vocab ≈ 10,000  
5. To encode: start from characters, apply merges in the order they were learned  

No HuggingFace `tokenizers` library. No SentencePiece. Pure Python.

---

## 6. The secret sauce — language weights

Raw page sizes differ a lot (Tamil ≫ Telugu). Naive BPE chases the biggest corpus.

We:

1. Normalize frequencies **inside each language**  
2. Multiply by a **language weight**  
3. Boost English enough that `X_en ≤ 1.2` while others stay nearby  

That shrinks `(Xmax − Xmin)` → score rises.

---

## 7. How to talk about this (30 seconds)

> I trained a shared 10k BPE from scratch across eight Indian languages, including Garhwali and Kumaoni which don’t have Wikipedia editions. Fairness is measured as fertility gap; English stays under 1.2 with zero UNKs. The widget compares a 4-language baseline against the full 8-language stress test, and ships a downloadable tokenizer.json.

---

## 8. Commands

```bash
# preview widget
python3 -m http.server 8080

# retrain
python3 scripts/train_bpe.py
```

---

## 9. Honest limitations

- `\w+` word counting ignores punctuation strategy debates  
- Garhwali/Kumaoni are not Wikipedia India pages — label the source  
- BPE is greedy; Unigram LM / WordPiece optimize differently  
- Score can be gamed by deleting hard text — we didn’t; claims are reproducible from `tokenizer.json`  

---

*When you forget why Core-4 score > All-8 score, re-read section 4.*
