# Day 29 — Downstream classifier comparison (Marco)

**Task:** Run RF/XGBoost/LightGBM/SVM on Original, SMOTE, CTGAN-v2, and the
now-fixed QGAN — replacing Week 3's broken-QGAN numbers with the corrected result.

## Status: COMPLETE. All four methods run, identical protocol, identical held-out test set.

### How the comparison was made fair

SMOTE and CTGAN's raw CSVs don't include an explicit "is this row real or
synthetic" flag. A plain merge against the original data to find real rows
**exploded** (this dataset has duplicate rows), so a hash-based duplicate-safe
matching approach was used instead (`../shared_data/code/build_augmented_frames.py`).
This surfaced something worth knowing before reading the results below:
**SMOTE retains every real row exactly** (matched counts: 9,791/10,020/9,487 —
exact to known real class sizes) plus synthetic fill to 29,298/class.
**CTGAN retains zero real rows** — every single CTGAN row is synthetic.
Different augmentation philosophies, not a bug.

The train/test split is **identical across all four methods** — same
`RANDOM_STATE`, same held-out test set (verified: all four methods' "original"
baseline F1 numbers match to 6 decimal places), so any difference in the
"augmented" columns is attributable to the augmentation method, not test-set
variance.

### Full results (F1-macro, Original vs. Augmented)

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

### Honest read

**SMOTE is the clear winner for downstream classifier impact** — a genuine,
substantial F1 lift across every classifier, largest for the tree ensembles
(+8.6pp RF, +4.7pp XGBoost). **CTGAN actually makes things slightly worse**,
not better — a real, somewhat counterintuitive finding: CTGAN's synthetic
data looked "easy" in the isolated small-sample V3 QSVM comparison (Day 31),
but blended into full-dataset training it appears to introduce enough
distribution shift to hurt generalization to real test data. **QGAN is
statistically flat** either way.

**Why the gap between SMOTE and the other two is so large:** synthetic-data
volume, not just quality. SMOTE injects ~66-68% synthetic rows per minority
class into training; QGAN's gap-fill scheme injects <5%; CTGAN sits in
between (~62-66%) but still underperforms SMOTE despite similar volume —
so volume alone doesn't explain CTGAN's negative result; see Day 30 for the
fidelity numbers that likely explain why.

### Note on running this code directly

`code/` is now self-contained (includes `config.py`, `data_prep.py`, `optim.py`,
`qgan_model.py`, `train_qgan.py` alongside `main.py`/`evaluate.py`) so imports
resolve correctly. To actually *execute* `main.py`, `config.py` expects a
sibling `../data/malmem_original_reconstructed.csv` — point it at
`../../shared_data/malmem_original.csv` (same data, different filename/location
in this package) or copy/rename accordingly. Included primarily for code
transparency and audit — not pre-wired as one-command-runnable in this
package layout.
