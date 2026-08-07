# Paired Significance Testing — QSVM v3 vs. Classical SVM

See `Paired_Significance_Testing_Results.md` for the tables, statistics, and
interpretation. Original dataset extended from 3 to 8 seeds; CTGAN included at
8 seeds for comparison.

## Setup
```bash
pip install --break-system-packages pennylane pennylane-lightning xgboost \
    scikit-learn scipy pandas numpy matplotlib
```
Data bundled in `data/`.

## Config
q=8, seeds 0–7, `n_total=500`. QSVM: `max_train=300`, `max_test=40`, 40
Nyström landmarks, alignment subset=10 (fixed circuit — trainable rotation
applied once). Classical: RBF-kernel SVM, `max_train=80`, `max_test=40`. Both
draw the same seeded test-batch subsample, making the comparison paired.

## Run
```bash
cd code
python3 run_paired.py Original 8 0 1 2 3 4 5 6 7
python3 run_paired.py CTGAN 8 0 1 2 3 4 5 6 7
```
Saves incrementally per seed; already-completed seeds are skipped on rerun.

## Files
| File | Purpose |
|---|---|
| `code/run_paired.py` | Paired runner + `paired_tests()` (t-test, Wilcoxon, CI, Cohen's d). |
| `results/paired_original.json` | Per-seed QSVM/classical pairs, Original. |
| `results/paired_ctgan.json` | Per-seed QSVM/classical pairs, CTGAN. |
| `figures/paired_seed_comparison.png` | Bar chart, both datasets, all 8 seeds. |
