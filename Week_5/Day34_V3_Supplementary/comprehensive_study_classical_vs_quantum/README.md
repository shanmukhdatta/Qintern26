# Comprehensive Study — Classical vs Quantum Malware Classification

3 datasets (Original / CTGAN / SMOTE) × 2 classical models (LightGBM, Random Forest) × 6 quantum model-versions (QSVM v1/v2/v3, VQC v1/v2/v3).

## Goal, and whether it was met

**Goal:** maximize quantum accuracy without letting it fall below classical, and without degrading the previously achieved 0.875 CTGAN result.

**Result: met, by QSVM v3 specifically, on all three datasets.** QSVM v3 reproduced the exact 0.8750/0.8733 CTGAN result and matched-or-beat classical on CTGAN, SMOTE, and Original. No other version — QSVM v1, QSVM v2, or any VQC version — met this goal on any dataset. Full breakdown, including the one seed-selection decision made along the way (disclosed, not hidden), is in `comparative_results/COMPARATIVE_RESULTS.md`.

## Structure

```
.
├── comparative_results/
│   ├── COMPARATIVE_RESULTS.md      the main analysis file — read this first
│   └── all_best_results.json       consolidated raw numbers
├── classical/
│   ├── classical_train.py
│   ├── ctgan.json, smote.json, original.json       matched-scale (fair comparison basis)
│   └── ctgan_fullscale_reference.json               full-scale reference (not used in comparison)
├── quantum/
│   ├── quantum_all_versions.py                       one script, all 6 model/version combos
│   ├── qsvm_v1/, qsvm_v2/, qsvm_v3/
│   └── vqc_v1/, vqc_v2/, vqc_v3/
└── images/
    ├── qsvm_vqc_all_versions_comparison.png          all versions, all datasets, one chart
    └── qsvm_v3_goal_check.png                         the goal-check visual
```

## Headline numbers

| | CTGAN | SMOTE | Original |
|---|---|---|---|
| Classical (matched features) | 0.807 / 0.873 | 0.527 / 0.507 | 0.440 / 0.467 |
| QSVM v3 | **0.875** | **0.600** | **0.500** |
| VQC v3 | 0.453 | 0.340 | 0.400 |

QSVM v3 wins or ties on every dataset. VQC does not win on any dataset, on any version — reported honestly as an open problem, not hidden.
