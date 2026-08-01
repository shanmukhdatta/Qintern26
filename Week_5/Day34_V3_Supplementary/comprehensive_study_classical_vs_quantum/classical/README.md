# Classical models — LightGBM + Random Forest

Two sets of results, clearly separated:

1. **Matched-scale (ctgan.json, smote.json, original.json)** — trained on the same sample count and same features quantum sees (after variance/correlation filtering, no PCA/LDA/XGBoost restriction). Hyperparameters tuned via RandomizedSearchCV (8 iterations, 3-fold CV, scoring on macro F1). This is the fair comparison basis used in comparative_results/COMPARATIVE_RESULTS.md.

2. **Full-scale reference (ctgan_fullscale_reference.json)** — LightGBM trained on the entire CTGAN dataset (~47,000 rows), no sample-size handicap. Included for honesty: this is what classical can really do when not artificially scale-matched to quantum. NOT used in the quantum comparison, since it isn't a fair test of method vs method, only of data-scale advantage.

File: classical_train.py — the exact script executed for both.
