# Day 1 — Kernel-Concentration Diagnostics

QTagger+ Consolidation Week, Team B. See `Day1_Results.md` for the results
table and interpretation.

## Setup
```bash
pip install --break-system-packages pennylane pennylane-lightning xgboost \
    scikit-learn scipy pandas numpy matplotlib
```
Data is bundled in `data/` (`malmem_ctgan.csv`, `MalMem2022.csv`).

## Config
8 qubits, seed 7, `n_total=500`, `max_train=300`, `max_test=40`, alignment
subset=10, KTA optimized via COBYLA, 100 permutations for the chance floor.

## Run
```bash
cd code
python3 run_diagnostics.py
```

## Files
| File | Purpose |
|---|---|
| `code/circuit.py` | Local/global kernel circuits with trainable alignment rotation. |
| `code/diagnostics.py` | `gram_offdiag_std`, `kta_score`, `kta_chance_floor`. |
| `code/run_diagnostics.py` | Builds the diagnostic table on the shared v2/v3 test batch. |
| `code/pipeline_v3_full.py` | Project's V3 pipeline (data loading, preprocessing, feature selection). |
| `code/data_original.py` | Loader for the Original/unaugmented dataset. |
| `results/day1_kernel_diagnostics.json` | Raw output backing the results table. |
| `figures/day1_kernel_diagnostics.png` | Gram-std and KTA-above-floor bar charts. |
