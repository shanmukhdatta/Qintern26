# Trained QTagger+ model artifacts

The Streamlit application loads the **trusted** model saved by the final Kaggle training notebook.

The evaluation files are already present in this folder. Copy the remaining trusted model file from `qtagger_outputs/malware3/` (or `four_class/`) in the Kaggle ZIP:

- `qtagger_v3_1.joblib` **required** (includes label names and all fitted preprocessors)
- `evaluation_metrics.json` recommended
- `evaluation.png` recommended
- `demo_unlabeled_features.csv` recommended

The real Kaggle evaluation metrics, package versions, model-comparison CSV and five-row input CSV have been committed from the values supplied by the user. The actual trained \`.joblib\` remains missing because the uploaded ZIP could not be extracted through the available connector. Do **not** substitute a dummy model.

If the model is reasonably small and you have verified its origin, copy it here and run:

```bash
python scripts/check_model.py --model models/qtagger_v3_1.joblib
git add -f models/qtagger_v3_1.joblib
git add models/evaluation_metrics.json models/evaluation.png models/demo_unlabeled_features.csv
git commit -m "Add verified Kaggle-trained QTagger+ model and evaluation artifacts"
git push
```

GitHub's normal file-size limit is 100 MiB. If larger, use a trusted release asset or private artifact storage and arrange a secure deployment-time download. **Do not load untrusted pickle/joblib files**.

There is **no separate label encoder file** to upload: the notebook's model bundle already stores `classes`, the raw feature schema, all fitted scalers, selected indices, KTA weights, landmarks, Nyström inverse square root and final SVM.
