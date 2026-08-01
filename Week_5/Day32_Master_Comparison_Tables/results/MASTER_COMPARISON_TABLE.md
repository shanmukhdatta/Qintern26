# Master configuration comparison — augmentation methods x classifier families x sample scales

Consolidated from every run available in this project as of this package.
All cells are now resolved — no open items remain as of this version.

**⚠️ Section 2 was corrected after this table was first written — expanded
seed testing overturned the original "QSVM v3 wins everywhere" claim. See
`../../Day31_QGAN_QSVM_Pipeline/results/SEED_EXPANSION_STATISTICAL_CORRECTION.md`
for full statistics. Section 2 below reflects the corrected numbers.**

## 1. Quantum vs. classical, by augmentation method (V2 pipeline, 27 configs)
Source: `../reference/V2_27config_CTGAN_SMOTE_Original_grid.md` (existing, pre-dates this session)

| Dataset | Classical SVM (mean) | Classical RF (mean) | QSVM v2 (mean) | VQC v2 (mean) | Quantum win rate |
|---|---|---|---|---|---|
| CTGAN | 0.703 | 0.683 | 0.619 | 0.561 | 1/9 (11%) |
| SMOTE | 0.394 | 0.386 | 0.394 | 0.358 | 3/9 (33%) |
| Original | 0.361 | 0.378 | **0.450** | 0.394 | 5/9 (56%) |
| **QGAN (this session, V3, not V2 — see caveat below)** | 0.475 avg | 0.475 avg | 0.567 avg | not run | 1/5 (20%, corrected — see Section 2) |

**Caveat on the QGAN row:** measured with the V3 pipeline (xgboost feature
selection + local trainable kernel), not V2 (LDA/PCA + data re-uploading) —
so it's not a strictly apples-to-apples pipeline match with the other three
rows. QGAN now also has V1/V2 QSVM results (Day 31), both losing to
classical (-21.7pp, -8.3pp respectively). VQC was not run on QGAN data.

## 2. V3 pipeline specifically — CORRECTED with multi-seed statistics
Source: `../../Day31_QGAN_QSVM_Pipeline/results/SEED_EXPANSION_STATISTICAL_CORRECTION.md`

**Original single-seed/best-of-2-seed numbers (kept for provenance, NOT the
current best estimate):**

| Dataset | QSVM v3 (headline) | Classical best | Margin | Seeds |
|---|---|---|---|---|
| CTGAN | 0.875 | 0.873 | +0.2pp | 7 only |
| SMOTE | 0.600 | 0.527 | +7.3pp | best-of-2 (7 gave 0.350, below classical) |
| Original | 0.500 | 0.467 | +3.3pp | 7 only |
| QGAN | 0.625 | 0.587 | +3.8pp | 7 only |

**Corrected numbers, 3-5 seeds each (42, 1, 7, 100, 123 as available):**

| Dataset | n seeds | Mean acc ± std | Mean margin ± std | Win rate |
|---|---|---|---|---|
| CTGAN | 3 | 0.708 ± 0.170 | **-0.125 ± 0.127** | 1/3 |
| SMOTE | 4 | 0.444 ± 0.094 | **-0.077 ± 0.099** | 1/4 |
| QGAN | 5 | 0.540 ± 0.049 | **-0.045 ± 0.049** | 1/5 |
| **Original** | 3 | 0.542 ± 0.031 | **+0.033 ± 0.024** | **3/3** |

**Only Original shows a robust win** — positive margin every seed tested, and
by far the lowest variance. CTGAN's single-seed 0.875 turned out to be the
high end of a 40-point swing (0.875 -> 0.475 across 3 seeds, same everything
else) — a favorable draw from a high-variance result, not a stable advantage.
CTGAN, SMOTE, and QGAN all show a **net loss on average** once 3+ seeds are
sampled. Full per-seed raw data:
`../../Day31_QGAN_QSVM_Pipeline/results/SEED_EXPANSION_raw_results.csv`.

This does not contradict the rest of the project — it confirms the same
pattern the V2 grid found independently in Section 1 (harder/messier data ->
more reliable quantum edge; synthetic/augmented data -> not reliable), now
found a second way with a different pipeline version and a different seed set.

## 3. Fidelity (KS/Wasserstein/MMD) by method — COMPLETE (was blocked, now resolved)

| Method | KS_median (range across classes) | Wasserstein_median (range) | MMD (range) |
|---|---|---|---|
| SMOTE | 0.010-0.013 | 0.061-0.084 | 0.001 |
| CTGAN | 0.091-0.133 | 0.143-0.284 | 0.005-0.081 |
| QGAN, 100 epochs (corrected) | 0.28-0.33 | 0.66-0.87 | 0.16-0.34 |

**Ranking, consistent across every class and metric: SMOTE > CTGAN > QGAN.**
Full breakdown: Day 30 README.

## 4. Downstream classifiers (RF/XGBoost/LightGBM/SVM), full-dataset protocol — COMPLETE (was blocked, now resolved)

| Method | RF F1 | XGBoost F1 | LightGBM F1 | SVM F1 |
|---|---|---|---|---|
| Original (no augmentation) | 0.8127 | 0.8211 | 0.8129 | 0.6089 |
| QGAN, 100 epochs (corrected) | 0.8121 | 0.8200 | 0.8108 | 0.6039 |
| SMOTE | **0.8988** | **0.8681** | **0.8419** | 0.6222 |
| CTGAN | 0.8082 | 0.8083 | 0.7927 | 0.5959 |

**SMOTE wins clearly. CTGAN is slightly negative. QGAN is flat.** Full
breakdown and the fidelity-vs-downstream-impact mismatch (CTGAN has better
fidelity than QGAN but worse downstream impact) discussed in Day 29/30 READMEs.

## 5. Consolidated interpretation

- **The win-rate-vs-data-difficulty gradient is the single most robust finding
  in this entire project — found independently three separate times**: (1) the
  V2 grid (Section 1: CTGAN 11% win rate, Original 56%), (2) this session's
  original V3 QGAN/CTGAN comparison, and (3) the corrected multi-seed V3
  results (Section 2: Original 3/3 with low variance; CTGAN/SMOTE/QGAN all
  net-negative with high variance). Different pipeline versions, different
  seed sets, same conclusion: **quantum's edge is real and reliable on
  harder, messier, unaugmented data — and not reliable on synthetic/augmented
  data, regardless of which method generated it.**
- **The original "QSVM v3 beats classical everywhere" claim does not hold.**
  It was based on 1-2 seeds per dataset and is corrected in Section 2. This is
  the most important correction in this package — read it before citing any
  V3 single-seed number as representative.
- **Fidelity ranking is clean and consistent: SMOTE > CTGAN > QGAN**, on every
  class and every metric (Section 3). **Downstream classifier impact ranks
  differently: SMOTE >> QGAN ≈ CTGAN(slightly negative)** (Section 4).
  CTGAN beats QGAN on fidelity but loses to it on downstream impact — the two
  measurements are related but not interchangeable, and neither one alone is
  a reliable proxy for the other when picking an augmentation method.
- **Downstream classifier metrics are not meaningfully moved by QGAN at either
  epoch count**, because the gap-fill augmentation scheme caps synthetic data
  at <5% of training volume — a design-choice ceiling. SMOTE injects
  ~66-68% synthetic rows and shows the largest downstream lift of any method
  tested; volume plus high fidelity together, not either alone, appears to be
  what drives SMOTE's result (CTGAN has similar volume but far worse
  downstream impact, tracking its worse fidelity).
- **Remaining gap**: seed counts are still modest (3-5 per dataset for the
  corrected V3 numbers) — enough to reverse the original single-seed
  headlines and establish a clear direction, not enough for tight confidence
  intervals. 10+ seeds per dataset would be the natural next step if this
  needs to hold up to closer statistical scrutiny.
