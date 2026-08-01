# QGAN Results — Comparison Tables

Run date: see `logs/training_run.log`. Target label: 4-class malware TYPE
(`Benign` / `Ransomware` / `Spyware` / `Trojan`), derived from the fine-grained
`Category` family names. Balancing target = 10,020 samples/class (size of the
largest malware type, Spyware — already at target, so untouched).

## 1. Class balance before / after

| Class      | Real count | Synthetic added | Final count |
|------------|-----------:|-----------------:|------------:|
| Benign     | 29,298     | 0                 | 29,298      |
| Spyware    | 10,020     | 0                 | 10,020      |
| Ransomware | 9,791      | 229               | 10,020      |
| Trojan     | 9,487      | 533               | 10,020      |

## 2. Fidelity metrics — real vs. QGAN-synthetic (per augmented class)

| Class      | n_real | n_synth | KS (median) | KS (p75) | Wasserstein (median) | Wasserstein (p75) | MMD (RBF) |
|------------|-------:|--------:|-------------:|---------:|----------------------:|-------------------:|----------:|
| Ransomware | 229    | 229     | 0.3886       | 0.6048   | 1.0819                | 7.1222              | 0.1743    |
| Trojan     | 533    | 533     | 0.2889       | 0.5009   | 0.5679                | 3.2582              | 0.3366    |

**Reading this:** KS statistic and Wasserstein distance are computed per feature
then summarized (median / 75th percentile) across all 54 features. Lower is
better fidelity. MMD (RBF kernel, median-heuristic bandwidth) is a single
distribution-level distance; 0 = identical distributions. These scores are
noticeably weaker than typical CTGAN/SMOTE fidelity in the same project
(commonly KS median ~0.15–0.2) — see the README for why, and for tuning
suggestions.

## 3. Downstream classifier comparison — Original vs. QGAN-augmented

Both columns are evaluated on the **same held-out real test split** (25% of
the original data, stratified by class). "Augmented" training data = original
train split + the 762 QGAN-synthetic rows.

| Classifier   | Accuracy (Original) | F1-macro (Original) | Accuracy (QGAN-augmented) | F1-macro (QGAN-augmented) |
|--------------|---------------------:|----------------------:|----------------------------:|-----------------------------:|
| RandomForest | 0.8756               | 0.8127                 | 0.8749                      | 0.8119                       |
| XGBoost      | 0.8814               | 0.8211                 | 0.8807                      | 0.8203                       |
| LightGBM     | 0.8758               | 0.8129                 | 0.8764                      | 0.8137                       |
| SVM          | 0.7440               | 0.6089                 | 0.7422                      | 0.5994                       |

**Reading this:** synthetic rows are only ~2–5% of each affected class's
training data, so the effect is small either way — essentially flat, mildly
negative. Not yet a win over the classical baselines from Week 3.
