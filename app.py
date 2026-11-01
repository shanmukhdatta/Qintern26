"""QTagger+ Streamlit Cloud entrypoint. Deploy this file as app.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from qtagger.artifacts import model_path, read_metrics, sample_csv_path
from qtagger.inference import QTaggerPredictor

REPO = "https://github.com/shanmukhdatta/Qintern26"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 100

st.set_page_config(
    page_title="QTagger+ | Quantum Malware Family Classification",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner="Loading trained QSVM and quantum simulator...")
def load_model(path: str, mtime_ns: int) -> QTaggerPredictor:
    return QTaggerPredictor.from_file(path)

st.title("🧬 QTagger+")
st.subheader("Hybrid quantum–classical malware family classification")
st.caption("CIC-MalMem-2022 memory-forensics features · corrected QSVM v3.1 · PennyLane + Nyström + SVM")
st.info(
    "Research demonstration only. This tool classifies pre-extracted numerical memory features; "
    "it does not scan files, execute malware, or replace security software."
)

with st.sidebar:
    st.header("About the project")
    st.markdown("**QIntern 2026 — QTagger+**")
    st.markdown("Quantum machine learning and data augmentation for malware family classification.")
    st.link_button("Source code & research", REPO)
    st.markdown("**Model:** corrected QSVM v3.1")
    st.markdown("**Pipeline:** XGBoost feature selection → angle encoding → trained local quantum readout → Nyström → linear SVM")
    st.caption("Quantum predictions can take several seconds per record on CPU.")

path = model_path()
if path is None:
    st.error("The trained model has not been installed on this deployment.")
    st.markdown(
        "Place the **trusted** Kaggle artifact `qtagger_v3_1.joblib` in "
        "`models/qtagger_v3_1.joblib` and redeploy. Optionally place "
        "`evaluation_metrics.json` and `demo_unlabeled_features.csv` beside it. "
        "See [deployment instructions](https://github.com/shanmukhdatta/Qintern26/blob/main/DEPLOYMENT.md)."
    )
    st.stop()

try:
    predictor = load_model(str(path), path.stat().st_mtime_ns)
except Exception as exc:
    st.error("Could not load the deployed model. Check artifact integrity and Python package versions.")
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

bundle = predictor.bundle
metrics = read_metrics(path, bundle)
st.success("Trained model loaded successfully.")
st.caption(f"Task: {bundle.get('task', 'unknown')} · Classes: {', '.join(predictor.classes)} · Simulator: {predictor.device_name}")
st.caption(f"Expected input: {len(predictor.raw_columns)} numeric memory-feature columns. Label and filename columns are not needed.")

tab_predict, tab_metrics, tab_method = st.tabs(["🔍 Predict from CSV", "📊 Evaluation", "🧪 How it works"])

with tab_predict:
    st.markdown(
        "Upload a **CSV of extracted numeric CIC-MalMem-2022 features**. "
        "Columns may be in any order; all training-time feature names must be present. "
        f"For interactive performance, uploads are limited to {MAX_ROWS} rows and 5 MB."
    )
    sample_path = sample_csv_path(path)
    if sample_path:
        st.download_button(
            "Download genuine demo input CSV",
            data=sample_path.read_bytes(),
            file_name="qtagger_demo_input.csv",
            mime="text/csv",
        )
    else:
        st.warning(
            "No real demo CSV is installed. Export `demo_unlabeled_features.csv` "
            "from your Kaggle notebook and place it beside the model. "
            "Do not use fabricated all-zero feature vectors as real samples."
        )
    upload = st.file_uploader("Upload numeric memory features (.csv)", type=["csv"])
    if upload is not None:
        if upload.size > MAX_UPLOAD_BYTES:
            st.error("CSV exceeds 5 MB. Upload a smaller file.")
        else:
            try:
                frame = pd.read_csv(upload, low_memory=False)
                if len(frame) == 0:
                    st.error("CSV contains no data rows.")
                elif len(frame) > MAX_ROWS:
                    st.error(f"CSV has {len(frame)} rows; maximum is {MAX_ROWS}.")
                else:
                    st.caption(f"Loaded {len(frame)} rows and {len(frame.columns)} columns.")
                    st.dataframe(frame.head(5), use_container_width=True)
                    if st.button("Classify malware families", type="primary"):
                        with st.spinner("Computing quantum similarities and predictions..."):
                            labels = predictor.predict(frame)
                        output = frame.copy()
                        output.insert(0, "predicted_family", labels)
                        st.success(f"Finished classifying {len(output)} records.")
                        st.dataframe(output[["predicted_family"]], use_container_width=True)
                        counts = pd.Series(labels).value_counts().rename_axis("family").reset_index(name="count")
                        st.bar_chart(counts.set_index("family"))
                        st.download_button(
                            "Download predictions CSV",
                            data=output.to_csv(index=False).encode("utf-8"),
                            file_name="qtagger_predictions.csv",
                            mime="text/csv",
                        )
                        st.caption(
                            "Predicted labels are not calibrated probabilities. "
                            "No confidence percentage is reported by this SVM."
                        )
            except (ValueError, KeyError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(f"CSV validation failed: {exc}")
            except Exception as exc:
                st.error("Prediction failed. Check the CSV and model compatibility.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")

with tab_metrics:
    st.markdown("### Held-out evaluation from the Kaggle training run")
    rows = metrics.get("metrics", [])
    if isinstance(rows, list) and rows:
        table = pd.DataFrame(rows)
        st.dataframe(table, use_container_width=True)
        if {"model", "macro_f1"}.issubset(table.columns):
            st.bar_chart(table.set_index("model")["macro_f1"])
    else:
        st.warning("No evaluation metrics found in the deployed model artifacts.")
    left, right = st.columns(2)
    left.metric("Training subset", metrics.get("n_train_subset", "Not recorded"))
    right.metric("Test subset", metrics.get("n_test_subset", "Not recorded"))
    st.caption("Metrics are from a specific held-out sample, not production detection rates.")
    if metrics.get("dataset_sha256"):
        st.code("Dataset SHA-256: " + str(metrics["dataset_sha256"]))
    fig = path.with_name("evaluation.png")
    if fig.is_file():
        st.image(str(fig), caption="Saved Kaggle evaluation chart", use_container_width=True)
    else:
        st.caption("Add evaluation.png beside the model to display the confusion matrix.")

with tab_method:
    st.markdown("""
### Inference pipeline

1. Validate required numerical columns in the uploaded CSV.
2. Impute missing values with training-only medians.
3. Apply the saved variance/correlation column filter and signed-log transform.
4. Apply the saved StandardScaler, eight-feature selection and RobustScaler.
5. Encode features as eight qubit rotation angles in the range [−π, π].
6. Reconstruct the corrected PennyLane circuit using trained KTA weights.
7. Compute local pairwise quantum measurements against training landmarks.
8. Apply the saved Nyström projection and linear SVM to predict the family.

### Scientific and safety limitations

- This is a **research classifier**, not an antivirus or a malware detector for arbitrary executables.
- It requires numerical memory-analysis features with the same schema as the training CSV.
- The local quantum similarity readout is not guaranteed to be a positive-semidefinite kernel.
- Accuracy and Macro-F1 depend on the recorded training/test sample and should not be generalized.
- The application does not execute uploaded files or deserialize user-uploaded models.
- For responsible use, upload only feature data that you are authorized to share.
""")
    st.link_button("Read training notebook", REPO + "/blob/qtagger-final-training-notebook/notebooks/QTagger_Plus_Final_Training.ipynb")
