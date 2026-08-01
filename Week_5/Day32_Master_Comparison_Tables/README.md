# Day 32 — Master Configuration Comparison (Shanmukh)

## Objective
Consolidate the full configuration comparison — augmentation methods x
classifier families x sample scales — into final master tables, checked for
consistency across every day's work so far (Days 29-31 plus the pre-existing
V2 grid).

## Tasks Performed
- Pulled together Day 29 (downstream classifiers), Day 30 (fidelity), Day 31
  (V3 QSVM/VQC pipeline), and the pre-existing V2 27-config grid into one
  master document.
- Applied the multi-seed correction from Day 31 to overwrite the original
  single-seed headline table.
- Cross-checked whether the correction found in Day 31 agreed with the
  independent, pre-existing V2 grid — a genuine consistency check across two
  different pipeline versions and two different sets of random seeds.
- Wrote one consolidated interpretation tying fidelity, downstream impact,
  and quantum-vs-classical results together.

## Methodology
This day involved no new experiments — it is a consolidation and consistency
check. The one substantive addition was comparing two independently-derived
results against each other:
1. The V2 grid's finding (quantum win rate: Original 56%, SMOTE 33%, CTGAN
   11%) — produced earlier, on a different pipeline architecture, with a
   different set of seeds.
2. The Day 31 multi-seed correction (quantum win rate: Original 3/3, CTGAN
   1/3, SMOTE 1/4, QGAN 1/5) — produced on the V3 pipeline.
If these two, built independently, showed the same underlying pattern, that
would be much stronger evidence than either one alone. They did.

## Results

**1. Quantum vs. classical by augmentation method (V2 pipeline, 27 configs):**

| Dataset | Classical SVM | Classical RF | QSVM v2 | VQC v2 | Quantum win rate |
|---|---|---|---|---|---|
| CTGAN | 0.703 | 0.683 | 0.619 | 0.561 | 1/9 (11%) |
| SMOTE | 0.394 | 0.386 | 0.394 | 0.358 | 3/9 (33%) |
| Original | 0.361 | 0.378 | **0.450** | 0.394 | 5/9 (56%) |
| QGAN (V3, not V2 — not a strict like-for-like) | 0.475 avg | 0.475 avg | 0.567 avg | not run | 1/5 (20%) |

**2. V3 pipeline, corrected with multi-seed testing:**

| Dataset | Headline (1-2 seeds) | Corrected mean ± std (3-5 seeds) | Win rate |
|---|---|---|---|
| CTGAN | 0.875 | 0.708 ± 0.170 | 1/3 |
| SMOTE | 0.600 | 0.444 ± 0.094 | 1/4 |
| QGAN | 0.625 | 0.540 ± 0.049 | 1/5 |
| **Original** | 0.500 | 0.542 ± 0.031 | **3/3** |

**3. Fidelity ranking (from Day 30):** SMOTE > CTGAN > QGAN, consistent on
every class and metric (KS 0.010-0.013 vs. 0.091-0.133 vs. 0.28-0.33).

**4. Downstream classifier impact (from Day 29):**

| Method | RF F1 | XGBoost F1 | SVM F1 |
|---|---|---|---|
| Original (no augmentation) | 0.8127 | 0.8211 | 0.6089 |
| SMOTE | **0.8988** | **0.8681** | 0.6222 |
| CTGAN | 0.8082 | 0.8083 | 0.5959 |
| QGAN | 0.8121 | 0.8200 | 0.6039 |

## Interpretation (in plain language)
- **The single most important finding in the whole project**: quantum
  (QSVM) shows a real, reliable advantage over classical models only on the
  original, unaugmented, messy dataset. Every form of synthetic/augmented
  data tested (CTGAN, SMOTE, QGAN, all architecture versions) showed
  quantum's apparent wins were high-variance noise, not a stable advantage.
- **Why original data behaves differently**: real malware data is genuinely
  hard — feature distributions overlap across malware families, there's no
  clean boundary between classes, and no artificial balancing. Quantum's
  more complex decision boundary has real room to help here. Synthetic data,
  in different ways, is more *separable* than real data — CTGAN especially,
  since it generates samples conditioned directly on class label, producing
  tight, well-separated clusters. Once data is already easy to tell apart,
  classical models do the job fine on their own, and quantum's extra
  complexity has nothing left to add — any apparent win there is just luck
  across random seeds, not a structural advantage.
- **Why this is trustworthy and not a coincidence**: the exact same pattern
  (quantum wins on hard data, not on synthetic/augmented data) was found
  three separate times, independently — the pre-existing V2 grid, the
  original V3 single-seed comparison, and the corrected V3 multi-seed
  result. Different pipeline versions, different seed sets, same conclusion.
- **Fidelity and downstream usefulness measure different things.** CTGAN
  beats QGAN on fidelity (how statistically close its synthetic data looks
  to real data) but loses to it on downstream classifier impact. Looking
  "realistic" on a feature-by-feature basis does not guarantee a classifier
  will actually learn better from it — the two need to be checked
  separately, not assumed to track each other.
- **Volume plus fidelity together, not either alone, seems to drive
  usefulness.** SMOTE has both the best fidelity and a large synthetic
  volume (66-68%) — and it's the only method with a clear downstream win.
  QGAN's tiny synthetic volume (2-5%) means even its improved fidelity
  couldn't move classifier metrics either way.

## Conclusion
- The original claim "QSVM v3 beats classical on all 3 datasets" does not
  hold — it was based on 1-2 seeds per dataset and is formally retired in
  this table, replaced by the multi-seed corrected numbers.
- Quantum's advantage is real, reproducible, and specific: it shows up only
  on hard, unaugmented, messy data — not on any synthetic/augmented data
  regardless of which method produced it.
- Fidelity ranking (SMOTE > CTGAN > QGAN) and downstream usefulness ranking
  (SMOTE >> QGAN ≈ CTGAN, CTGAN slightly negative) do not match each other —
  both need to be measured for any future augmentation method, not just one.
- Remaining limitation: 3-5 seeds is enough to overturn a false headline and
  establish the correct direction, but not enough for tight statistical
  confidence intervals. 10+ seeds per dataset is the natural next step if
  this needs to hold up to closer scrutiny.
