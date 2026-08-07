# Day 2 — Locality Dose-Response Sweep: Results

**Team B (Shanmukh) · QIntern 2026 · Mentors: Dr. Simranjit Singh, Dr. Mohit Sajwan**
**Config:** 8 qubits, seed 7, XGBoost top-8 feature selection, re-upload embedding (L=2), Nystrom landmark QSVM, **no KTA alignment** (locality isolated as the only variable). n_total=500 (per-class balanced 3-class: Ransomware/Spyware/Trojan), 300 train / 40 test, 40 Nystrom landmarks.

---

## Paper-ready summary

Sweeping the local-kernel measurement's group size g ∈ {1, 2, 4, 8} shows that locality alone does not produce a monotonic dose–response: on CTGAN, accuracy peaks at intermediate group sizes (g=2, g=4, both 82.5%) and is lower at both the fully-local (g=1, 72.5%) and fully-global (g=8, 70.0%) extremes, while on the Original (unaugmented) data accuracy is flat across g=1–4 (45.0%) before dropping sharply only at the fully global measurement (g=8, 37.5%). This indicates the previously reported g=2 advantage reflects an optimal-group-size effect rather than a "more local is strictly better" mechanism, and the paper's locality claim should be softened accordingly on CTGAN while still holding, in a weaker plateau form, on Original.

---

## Sanity check (do this first, per brief §3.4)

g=8 is mathematically equivalent to the original global (whole-register) measurement and should reproduce the existing "global kernel, no alignment" reference numbers.

| Check | Expected (brief) | Obtained | Status |
|---|---|---|---|
| g=8, CTGAN | ≈ 70.0% | **70.00%** | ✅ matches |
| g=2, CTGAN | 82.5% (already tested) | **82.50%** | ✅ matches |

Both sanity-check rows land exactly on the brief's reference values, so the generalized grouping function is confirmed to correctly reduce to the existing g=2 and global (g=8) cases before the g=1/g=4 numbers below are trusted.

---

## Results table

| g | Dataset | Accuracy | Macro-F1 | Gram off-diag std | KTA − chance floor |
|---|---|---|---|---|---|
| 1 | CTGAN    | 72.50% | 0.7238 | 0.1263 | 0.0120 |
| 2 | CTGAN    | **82.50%** | 0.8279 | 0.1667 | 0.0217 |
| 4 | CTGAN    | **82.50%** | 0.8250 | 0.1775 | 0.0402 |
| 8 | CTGAN    | 70.00% | 0.6901 | 0.1344 | 0.0457 |
| 1 | Original | 45.00% | 0.4483 | 0.1498 | 0.0068 |
| 2 | Original | 45.00% | 0.4447 | 0.2251 | 0.0148 |
| 4 | Original | 45.00% | 0.4431 | 0.2274 | 0.0247 |
| 8 | Original | 37.50% | 0.3676 | 0.2005 | 0.0255 |

(Gram off-diagonal std and KTA are computed on the 40×40 landmark-vs-landmark Gram submatrix K_MM — see note on diagnostics below. "KTA − chance floor" = kta_score minus the mean KTA of the *same* Gram matrix under 200 random label permutations.)

**Dose-response figure:** `figures/locality_dose_response.png` — accuracy and Gram off-diagonal std vs. g, both datasets, one figure.

---

## Interpretation (per brief §5)

**Neither dataset shows the clean monotonic "accuracy falls smoothly as g rises" pattern.** Matching the brief's own alternative scenarios:

- **CTGAN — "g=1 worse than g=2" case.** Accuracy is *not* monotonic: it rises from g=1 (72.5%) to a plateau at g=2/g=4 (82.5% both), then falls at g=8 (70.0%). This is an inverted-U / optimal-group-size pattern, not a "more local is always better" pattern. Per the brief's own guidance, this is "a more nuanced (and more credible) claim than a monotonic story" — the previously reported 82.5% at g=2 is real and reproduces exactly, but it looks like a sweet spot rather than the local end of a strictly monotonic curve.
- **Original — plateau-then-drop.** Accuracy is flat across g=1, g=2, g=4 (45.0% exactly at all three) and only drops at g=8 (37.5%). This is weak support for the mechanism claim in one specific sense (fully delocalizing to the global measurement does hurt, on both datasets), but it does **not** support "more local is better" as a graded effect on Original — g=1 buys nothing over g=4 here.
- **Gram off-diagonal std also peaks at g=4 (not g=1)** on both datasets, then falls at g=8. Kernel spread is not simply monotonic in locality either, so "less locality → more concentration" is not a clean linear story in this data.
- **A genuine tension worth reporting plainly:** KTA − chance floor *increases* monotonically with g on both datasets (CTGAN: 0.012→0.022→0.040→0.046; Original: 0.007→0.015→0.025→0.026) — i.e. the fully global kernel (g=8) is the *most* label-aligned by this metric on both datasets, despite being the worst- or tied-worst-performing on accuracy. This directly cuts against a simple "locality → alignment → accuracy" causal story and should be flagged in the paper rather than smoothed over.

**Bottom line per the brief's own decision rule:** this is the "flat or noisy" / non-monotonic case the brief explicitly anticipates. The original 82.5% (g=2, CTGAN) is reproducible and real, but should **not** be over-stated as evidence that locality itself is the driving mechanism in a strictly graded sense — the paper's mechanism claim needs softening to "an intermediate/local grouping outperforms the fully global measurement" rather than "more local is monotonically better," and this conclusion should ideally be checked against additional seeds before submission, per the brief's own caution that a single-seed sweep should not be over-interpreted.

---

## Explicit caveats / assumptions (transparency, per task hard rules)

1. **Diagnostics functions were re-implemented, not reused verbatim**, because the Day 1 brief that defines `gram_offdiag_std`, `kta_score`, and `kta_chance_floor` was not included in the files provided for this Day 2 task. They were built from first principles using the same ideal-target convention (+1 same-class / −1 different-class) already present in `quantum_all_versions.py`'s `kta_train()` loss (lines 131–140), and are computed on the 40×40 landmark Gram submatrix `K_MM` (since the Nystrom runner never forms a full 300×300 square training Gram matrix). If Day 1's actual diagnostic functions differ from this reimplementation, the accuracy/F1 numbers above are unaffected (they come directly from the classifier), but the Gram-std/KTA columns should be recomputed with Day 1's originals for full consistency across the two days' tables.
2. **Single seed (seed=7), as specified by the brief.** No seed cherry-picking was done — seed 7 was fixed before any run, matching the brief's stated seed and the existing g=2 reference point. The flat/non-monotonic pattern reported here is this seed's result; the brief itself flags that more seeds may be needed before the mechanism claim is finalized, and this run does not resolve that on its own.
3. **No results were adjusted after the fact.** The g=1 and g=8 CTGAN numbers were the first and only run at each configuration; nothing was re-run to change the shape of the curve.
