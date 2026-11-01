# QTagger+ — Kaggle model → GitHub → Streamlit Community Cloud

This guide deploys the **corrected QSVM v3.1** trained in `notebooks/QTagger_Plus_Final_Training.ipynb`. No paid hosting is required for a small research demo.

## 1. Export your real Kaggle model

After **all notebook cells finish successfully**, Kaggle produces `qtagger_plus_kaggle_artifacts.zip` under `/kaggle/working/qtagger_outputs/`.

Download and unzip it. In `malware3/` (or `four_class/`) locate:

```text
qtagger_v3_1.joblib
evaluation_metrics.json
evaluation.png
demo_unlabeled_features.csv
environment.json
```

The `.joblib` already contains `classes` (the label mapping), raw feature columns, train-fitted preprocessing, quantum weights, landmarks, Nyström transform and trained SVM. **No separate label file is needed.** Never deserialize a model from an untrusted source.

## 2. Put the artifacts into the repository

Use the `qtagger-final-training-notebook` branch containing the Streamlit app, or merge its PR to `main`.

```bash
git clone https://github.com/shanmukhdatta/Qintern26.git
cd Qintern26
git checkout qtagger-final-training-notebook
mkdir -p models
# Copy the trusted files from the Kaggle ZIP into models/
python -m pip install -r requirements.txt
python scripts/check_model.py --model models/qtagger_v3_1.joblib --sample models/demo_unlabeled_features.csv
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

**Important:** The `.gitignore` intentionally excludes `*.joblib` from accidental commits. Only after checking the trusted file:

```bash
git add -f models/qtagger_v3_1.joblib
git add models/evaluation_metrics.json models/evaluation.png models/demo_unlabeled_features.csv
git commit -m "Add verified Kaggle-trained QSVM v3.1 artifacts"
git push origin qtagger-final-training-notebook
```

If the model exceeds GitHub's 100 MiB file limit, store it in trusted external artifact storage instead of committing it. The current app expects a local file; configure a secure startup download before using external storage. Never point the app at an arbitrary user-provided pickle URL.

## 3. Test the website locally

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. Download the demo input CSV, upload it, click **Classify malware families**, then download the predictions.

**Expected:** the model loads, the CSV passes schema checks, predictions match the notebook's saved demo predictions, and the evaluation tab shows the saved metrics.

## 4. Publish with Streamlit Community Cloud

1. Visit **https://share.streamlit.io/** and sign in with GitHub.
2. Select **Create app → Deploy a public app from GitHub**.
3. Repository: `shanmukhdatta/Qintern26`.
4. Branch: `main` after merging, or `qtagger-final-training-notebook` while testing.
5. Main file path: `app.py`.
6. Advanced settings: choose **Python 3.11**, and deploy. Streamlit installs `requirements.txt`.
7. Wait for dependencies and the app to load. Use the real demo CSV to test inference.

**Deployment cannot be marked successful until a real Kaggle model is present and a prediction succeeds.** If you see "trained model has not been installed," add the verified model and redeploy. If you see a package error, compare installed versions with Kaggle's `environment.json` and adjust `requirements.txt`.

## 5. Add the real public URL to README

Streamlit Community Cloud will assign a URL such as `https://YOUR-APP-NAME.streamlit.app/` **after successful deployment**. Replace the README's **Deployment pending** text with your **actual** URL, not this placeholder.

## 6. Troubleshooting

- **Model not found:** Confirm `models/qtagger_v3_1.joblib` exists in the deployed GitHub branch.
- **ImportError / unpickling error:** Use Python 3.11 and compare scikit-learn / NumPy versions against Kaggle `environment.json`. Only use a trusted model.
- **Missing columns:** The CSV must include all `raw_feature_columns`; use the real Kaggle demo CSV as the template.
- **Slow predictions:** Every record needs `N_LANDMARKS` quantum circuit evaluations. Keep demo uploads small; the app limits to 100 rows.
- **No model accuracy displayed:** Include `evaluation_metrics.json` or ensure the joblib's `evaluation` field is populated.
- **Unexpected output labels:** Ensure the model's `task` and `classes` match the dataset used in Kaggle. Do not fabricate a separate label file.

## 7. Model validity

This demo classifies numerical memory-forensics features, not arbitrary malware binaries. Its local quantum similarity is not guaranteed PSD, and the reported results are for the saved test subset. Do not claim a universal quantum advantage or production malware detection.
