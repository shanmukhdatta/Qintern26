# QTagger+ — Kaggle model → GitHub → Streamlit Community Cloud

This guide deploys the **corrected QSVM v3.1** trained in `notebooks/QTagger_Plus_Final_Training.ipynb`. No paid hosting is required for a small research demo.

## Browser-only method: upload the previously trained model

The GitHub repository already includes `app.py`, `requirements.txt`, the preprocessing/inference code, real evaluation metrics and a real sample CSV. **Only the trusted trained model is required for the first deployment.**

1. On your computer, download the original `qtagger_v3_1.joblib` artifact that you uploaded to ChatGPT or exported from Kaggle. Do not rename/retrain or substitute another model.
2. Sign in to GitHub and open [the `models` folder on the deployment branch](https://github.com/shanmukhdatta/Qintern26/tree/qtagger-final-training-notebook/models). Check that the branch selector says `qtagger-final-training-notebook`, **not `main`**.
3. Select **Add file → Upload files** (in the `models` folder), and drag in **`qtagger_v3_1.joblib`**. At the bottom, write commit message `Add trained QSVM v3.1 model` and commit to the **existing deployment branch**.
4. Confirm the file is listed at `models/qtagger_v3_1.joblib`. The file is about 100 KB, so GitHub's browser upload limit should not be a problem. GitHub's browser upload bypasses local `.gitignore` staging behavior.
5. Open [GitHub Actions](https://github.com/shanmukhdatta/Qintern26/actions), inspect the newest **QTagger+ Streamlit checks** run and confirm the real Kaggle model smoke-test step succeeds before publishing. If a step fails, open its logs; do not mark the app validated.
6. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/), click **Create app → Yup, I have an app**, and choose the repository, branch, file path and Python version as described below. Deploy.
7. Open the deployed URL, run the built-in demo with real feature values, and confirm it produces class labels. Only then update the README with the **actual** `.streamlit.app` URL.

**Note:** A model uploaded to an earlier ChatGPT conversation may be visible as a Library listing but not authorized for raw-byte transfer to GitHub. In that case, upload the same trusted model file directly in GitHub using the steps above.

---

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
6. Advanced settings: choose **Python 3.13** (matching Kaggle) if offered, and deploy. Streamlit installs `requirements.txt`.
7. Wait for dependencies and the app to load. Use the real demo CSV to test inference.

**Deployment cannot be marked successful until a real Kaggle model is present and a prediction succeeds.** If you see "trained model has not been installed," add the verified model and redeploy. If you see a package error, compare installed versions with Kaggle's `environment.json` and adjust `requirements.txt`.

## 5. Public Streamlit URL and deployment verification

- **Project-owner-provided URL:** https://qtagger.streamlit.app/
- **GitHub status:** The trusted `models/qtagger_v3_1.joblib` has been committed and the GitHub Actions inference smoke test passed.
- **Runtime status:** Hosting is configured according to the supplied URL, but independent hosted-browser inference has not been confirmed in this documentation update.

Open the hosted application, verify the "Trained model loaded successfully" message, run the built-in five-row Kaggle demo, and confirm it returns predicted labels. If hosted inference fails, inspect Streamlit Cloud logs and this repository's deployment instructions before claiming end-to-end completion.

## 6. Troubleshooting

- **Model not found:** Confirm `models/qtagger_v3_1.joblib` exists in the deployed GitHub branch.
- **ImportError / unpickling error:** Prefer Python 3.13 to match the Kaggle training environment and compare scikit-learn / NumPy versions against Kaggle `environment.json`. Only use a trusted model.
- **Missing columns:** The CSV must include all `raw_feature_columns`; use the real Kaggle demo CSV as the template.
- **Slow predictions:** Every record needs `N_LANDMARKS` quantum circuit evaluations. Keep demo uploads small; the app limits to 100 rows.
- **No model accuracy displayed:** Include `evaluation_metrics.json` or ensure the joblib's `evaluation` field is populated.
- **Unexpected output labels:** Ensure the model's `task` and `classes` match the dataset used in Kaggle. Do not fabricate a separate label file.

## 7. Model validity

This demo classifies numerical memory-forensics features, not arbitrary malware binaries. Its local quantum similarity is not guaranteed PSD, and the reported results are for the saved test subset. Do not claim a universal quantum advantage or production malware detection.
