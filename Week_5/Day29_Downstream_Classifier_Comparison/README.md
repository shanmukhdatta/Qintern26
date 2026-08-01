# Day 29 — Downstream Classifier Comparison (Marco)

## Objective
Run the same set of downstream classifiers (Random Forest, XGBoost, LightGBM,
SVM) on four versions of the training data — Original, SMOTE, CTGAN, and the
now-fixed QGAN — to see which augmentation method actually helps a real
classifier, and to replace Week 3's broken-QGAN numbers with the corrected
result.

## Tasks Performed
- Loaded Original, SMOTE, CTGAN, and QGAN (100-epoch, corrected) training sets.
- Separated real rows from synthetic rows inside the SMOTE and CTGAN files
  (neither file ships with an explicit real/synthetic flag).
- Trained all four classifiers on each dataset version, evaluated on one
  shared, untouched, held-out test split.
- Replaced Week 3's broken (30-epoch) QGAN numbers with the corrected
  (100-epoch, calibrated) QGAN numbers.

## Methodology
- **The real-vs-synthetic split problem:** a plain merge (matching rows back
  to the original data) was tried first to identify which SMOTE/CTGAN rows
  were real. It broke, because this dataset has duplicate rows and a plain
  merge multiplies duplicates instead of matching them 1-to-1. Fixed with a
  hash-based, duplicate-safe matching method instead
  (`../shared_data/code/build_augmented_frames.py`).
- **Fairness check:** all four methods were trained/tested on the exact same
  `RANDOM_STATE` and the exact same held-out split. Confirmed by checking that
  all four methods' "original-only" baseline F1 scores matched to 6 decimal
  places — meaning any difference in the "augmented" results is caused by the
  augmentation method itself, not by accidentally using a different test set.
- **What the real-vs-synthetic split revealed on its own:** SMOTE keeps every
  real row exactly as-is and only adds synthetic rows on top. CTGAN, on the
  other hand, keeps zero real rows — every single CTGAN row is synthetic.
  These are two fundamentally different augmentation philosophies, not a bug
  in either file.

## Results
F1-macro score, Original data vs. Augmented data, per classifier:

| Method | Classifier | F1 (original) | F1 (augmented) | Delta |
|---|---|---|---|---|
| **QGAN** (100ep, corrected) | RandomForest | 0.8127 | 0.8121 | -0.0006 |
| | XGBoost | 0.8211 | 0.8200 | -0.0012 |
| | LightGBM | 0.8129 | 0.8108 | -0.0021 |
| | SVM | 0.6089 | 0.6039 | -0.0050 |
| **SMOTE** | RandomForest | 0.8127 | **0.8988** | **+0.0861** |
| | XGBoost | 0.8211 | **0.8681** | **+0.0470** |
| | LightGBM | 0.8129 | **0.8419** | **+0.0290** |
| | SVM | 0.6089 | 0.6222 | +0.0133 |
| **CTGAN** | RandomForest | 0.8127 | 0.8082 | -0.0045 |
| | XGBoost | 0.8211 | 0.8083 | -0.0128 |
| | LightGBM | 0.8129 | 0.7927 | -0.0202 |
| | SVM | 0.6089 | 0.5959 | -0.0130 |

Full table: `results/MASTER_downstream_4way.csv`. Per-method raw output:
`results/downstream_SMOTE.csv`, `results/downstream_CTGAN.csv`,
`results/downstream_QGAN_100epoch_gapfill_CORRECTED.csv`.

## Interpretation (in plain language)
- **SMOTE clearly helps.** Every classifier got noticeably better, especially
  the tree-based ones (+8.6 points for Random Forest, +4.7 for XGBoost).
  SMOTE works by generating new rows *by interpolating between real
  neighbors* — so the new rows stay realistic almost by definition, and
  there's a lot of them (roughly 66-68% of each minority class's training
  rows end up synthetic).
- **CTGAN actually makes things slightly worse**, not better. This is a
  genuinely useful, counterintuitive result: CTGAN's synthetic data looked
  "clean" and easy to separate in isolated small-sample tests (see Day 31),
  but when blended into the full training set, it introduces enough
  distribution shift that classifiers trained on it generalize *slightly
  worse* to real test data.
- **QGAN is basically flat** — neither helps nor hurts in any meaningful way.
  The reason is simple: QGAN's augmentation scheme only adds synthetic rows
  up to about 2-5% of the affected classes' training volume. Even a perfect
  generator couldn't move the needle by injecting that little extra data —
  the QGAN itself is discussed in detail in Day 33 (its generator never fully
  stabilized during training), but even setting that aside, the *volume* of
  synthetic data it contributes is too small to show up in a classifier's F1
  score either way.
- **Volume matters as much as quality.** SMOTE (66-68% synthetic) and CTGAN
  (62-66% synthetic) inject similar amounts of synthetic data, yet SMOTE
  helps and CTGAN hurts — so volume alone doesn't explain the difference.
  The likely explanation (fidelity — how statistically close the synthetic
  data actually is to real data) is measured directly in Day 30.

## Conclusion
- SMOTE is the clear winner for downstream classifier impact on this dataset.
- CTGAN is mildly harmful when blended into full-scale training, despite
  looking "easy" in small isolated comparisons.
- QGAN's current augmentation volume (2-5%) is too small to move classifier
  metrics in either direction — this is a design-choice ceiling, not a sign
  that the fix failed outright.
- Decision: proceed to Day 30 to measure fidelity directly, to explain *why*
  CTGAN underperforms despite similar synthetic volume to SMOTE.
