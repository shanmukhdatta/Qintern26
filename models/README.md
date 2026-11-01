# Trained QTagger+ model artifacts

The Streamlit application loads the **trusted** model saved by the final Kaggle training notebook.

The evaluation files are already present in this folder. The deployed branch now contains the trusted model and evaluation files originally exported from Kaggle:

- `qtagger_v3_1.joblib` **required** (includes label names and all fitted preprocessors)
- `evaluation_metrics.json` recommended
- `evaluation.png` recommended
- `demo_unlabeled_features.csv` recommended

The original Kaggle model `qtagger_v3_1.joblib` (99,653 bytes) and the real evaluation metrics, package versions, model-comparison CSV and five-row feature-input CSV are now committed to this deployment branch. The GitHub Actions real-model smoke test passed. This does **not** independently verify inference through the hosted Streamlit website.

If updating the trusted model in a future training run, first validate it locally, then commit the replacement:

```bash
python scripts/check_model.py --model models/qtagger_v3_1.joblib
git add -f models/qtagger_v3_1.joblib
git add models/evaluation_metrics.json models/evaluation.png models/demo_unlabeled_features.csv
git commit -m "Add verified Kaggle-trained QTagger+ model and evaluation artifacts"
git push
```

GitHub's normal file-size limit is 100 MiB. If larger, use a trusted release asset or private artifact storage and arrange a secure deployment-time download. **Do not load untrusted pickle/joblib files**.

There is **no separate label encoder file** to upload: the notebook's model bundle already stores `classes`, the raw feature schema, all fitted scalers, selected indices, KTA weights, landmarks, Nyström inverse square root and final SVM.
