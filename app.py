"""QTagger+ | responsive research dashboard for the saved Kaggle QSVM v3.1 model.

The Streamlit user interface is deliberately separate from qtagger.inference.
Only CSV feature data may be uploaded. The trained joblib file is trusted,
bundled with the repository and never accepted from website visitors.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from qtagger.artifacts import model_path, read_metrics, sample_csv_path
from qtagger.inference import QTaggerPredictor


ROOT = Path(__file__).resolve().parent
REPO = "https://github.com/shanmukhdatta/Qintern26"
NOTEBOOK = REPO + "/blob/main/notebooks/QTagger_Plus_Final_Training.ipynb"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 100

st.set_page_config(
    page_title="QTagger+ • Quantum Malware Research",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

stylesheet = ROOT / "assets" / "dashboard.css"
if stylesheet.is_file():
    st.markdown("<style>" + stylesheet.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing trained QSVM circuit...")
def load_model(model_file: str, mtime_ns: int) -> QTaggerPredictor:
    """Cache only the trusted local model, not user-uploaded objects."""
    return QTaggerPredictor.from_file(model_file)


def score_text(model_row: dict | None, field: str, decimals: int = 1) -> str:
    value = model_row.get(field) if isinstance(model_row, dict) else None
    return f"{100 * float(value):.{decimals}f}%" if isinstance(value, (int, float)) else "—"


def card(label: str, value: str, note: str, accent: str = "") -> None:
    st.markdown(
        f'<div class="qt-stat {accent}">'
        f'<div class="qt-stat-label">{escape(label)}</div>'
        f'<div class="qt-stat-value">{escape(value)}</div>'
        f'<div class="qt-stat-note">{escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def panel(title: str, body: str) -> None:
    st.markdown(
        f'<div class="qt-panel"><div class="qt-panel-title">{escape(title)}</div>'
        f'<div class="qt-panel-sub">{escape(body)}</div></div>',
        unsafe_allow_html=True,
    )


def clear_result() -> None:
    st.session_state.pop("qtagger_result", None)


def classify(predictor: QTaggerPredictor, frame: pd.DataFrame, source: str) -> None:
    """Run real quantum inference, then keep results across Streamlit reruns."""
    try:
        with st.spinner("Calculating quantum similarities against trained landmarks..."):
            labels = predictor.predict(frame)
        result = frame.copy()
        result.insert(0, "predicted_family", labels)
        st.session_state.qtagger_result = {"data": result, "source": source}
        st.success(f"Prediction complete — {len(result)} records classified.")
    except Exception as exc:
        clear_result()
        st.error("Prediction could not be completed. Check feature schema and model compatibility.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")


def render_results() -> None:
    saved = st.session_state.get("qtagger_result")
    if not saved:
        return
    result = saved["data"]
    st.markdown(
        '<div class="qt-section-header"><div class="qt-section-title">Prediction results</div>'
        f'<div class="qt-section-note">{escape(saved["source"])} · {len(result)} records</div></div>',
        unsafe_allow_html=True,
    )
    counts = result["predicted_family"].value_counts()
    if not counts.empty:
        badge_html = "".join(
            '<span class="qt-result-tag">' + escape(str(name)) + ": " +
            str(int(value)) + "</span>&nbsp;&nbsp;"
            for name, value in counts.items()
        )
        st.markdown(badge_html, unsafe_allow_html=True)
    st.write("")
    table_col, chart_col = st.columns([1.2, 1], gap="large")
    with table_col:
        st.dataframe(
            pd.DataFrame({
                "Record": range(1, len(result) + 1),
                "Predicted family": result["predicted_family"].values,
            }),
            use_container_width=True, hide_index=True,
        )
    with chart_col:
        chart = counts.rename("Records").rename_axis("Malware family")
        st.bar_chart(chart)
    st.download_button(
        "⬇ Download predictions as CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="qtagger_predictions.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_predictions",
    )
    st.caption(
        "Predicted family names are classifier outputs, not detection guarantees. "
        "The saved linear SVM does not provide calibrated confidence probabilities."
    )
    with st.expander("Inspect predictions with all uploaded numerical features"):
        st.dataframe(result.head(100), use_container_width=True, hide_index=True)


# Model and Kaggle evaluation remain the source of truth.
artifact = model_path()
predictor = None
load_error = None
if artifact is not None:
    try:
        predictor = load_model(str(artifact), artifact.stat().st_mtime_ns)
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"

evaluation = read_metrics(artifact, predictor.bundle) if artifact and predictor else {}
rows = evaluation.get("metrics", [])
rows = rows if isinstance(rows, list) else []
qsvm = next((x for x in rows if isinstance(x, dict) and
             "QSVM v3.1" in str(x.get("model", ""))), None)
rf = next((x for x in rows if isinstance(x, dict) and
           str(x.get("model", "")).startswith("RF same 8")), None)

with st.sidebar:
    st.markdown(
        '<div class="qt-side-brand"><div class="qt-side-logo">QTagger<span>+</span></div>'
        '<div class="qt-side-caption">Quantum × Classical · Research Lab</div></div>',
        unsafe_allow_html=True,
    )
    if predictor:
        st.markdown(
            '<div class="qt-status">● Model loaded · QSVM v3.1</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Model currently unavailable")
    st.markdown("##### Workspace")
    st.caption("Explore saved evaluation metrics, then run an example or upload the required numeric memory features.")
    st.link_button("GitHub repository ↗", REPO, use_container_width=True)
    st.link_button("Training notebook ↗", NOTEBOOK, use_container_width=True)
    st.markdown("---")
    st.markdown("##### Deployment details")
    st.caption(f"**Simulator:** {predictor.device_name if predictor else 'Unavailable'}")
    st.caption(f"**Raw numeric features:** {len(predictor.raw_columns) if predictor else 'Unavailable'}")
    st.caption("**Model:** Corrected QSVM v3.1")
    st.caption("QIntern 2026 · CIC-MalMem-2022")

status_label = "MODEL ONLINE" if predictor else "MODEL UNAVAILABLE"
status_class = "live" if predictor else ""
st.markdown(
    f"""
<div class="qt-hero">
  <div class="qt-hero-copy">
    <div class="qt-kicker">QINTERN 2026 / QUANTUM RESEARCH WORKSPACE</div>
    <h1>Malware families.<br><span>Decoded with quantum ML.</span></h1>
    <p>Explore a trained hybrid quantum–classical classifier built on
       CIC-MalMem-2022 memory-forensics features. Review measured performance,
       run a real Kaggle example, or classify your own extracted feature CSV.</p>
    <div class="qt-chip-row">
      <span class="qt-chip {status_class}">{status_label}</span>
      <span class="qt-chip">QSVM v3.1</span>
      <span class="qt-chip">PennyLane + Nyström</span>
      <span class="qt-chip">Research demo</span>
    </div>
  </div>
  <div class="qt-orbit" aria-hidden="true">
    <div class="qt-orbit-ring r1"></div><div class="qt-orbit-ring r2"></div>
    <div class="qt-orbit-ring r3"></div><div class="qt-orbit-core">Q<span style="color:#53dfc6">+</span></div>
    <div class="qt-orbit-node n1"></div><div class="qt-orbit-node n2"></div>
    <div class="qt-orbit-node n3"></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if predictor is None:
    st.error("The trained model is not ready. Predictions are disabled until the trusted Kaggle artifact loads.")
    with st.expander("How to fix"):
        st.write("Ensure models/qtagger_v3_1.joblib exists on the deployed branch and matches the pinned Python dependencies.")
        if load_error:
            st.code(load_error)
        st.markdown(f"[Deployment instructions]({REPO}/blob/main/DEPLOYMENT.md)")
    st.stop()

st.markdown(
    '<div class="qt-section-header"><div class="qt-section-title">Experiment at a glance</div>'
    '<div class="qt-section-note">Saved Kaggle hold-out metrics · not production detection rates</div></div>',
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns(4, gap="small")
with c1:
    card("QSVM accuracy", score_text(qsvm, "accuracy", 2), "Held-out test subset", "teal")
with c2:
    card("QSVM macro-F1", score_text(qsvm, "macro_f1", 2), "Balanced view of class performance", "purple")
with c3:
    card("Evaluation records", str(evaluation.get("n_test_subset", "—")), "Subset from CIC-MalMem-2022", "sky")
with c4:
    card("Malware families", str(len(predictor.classes)), " / ".join(map(str, predictor.classes)), "")

st.markdown(
    '<div class="qt-section-header"><div class="qt-section-title">Your research workbench</div>'
    '<div class="qt-section-note">Start with a real five-record sample or upload your own features</div></div>',
    unsafe_allow_html=True,
)
tab_classify, tab_evaluation, tab_pipeline = st.tabs(
    ["⚡ Classify samples", "📊 Model evaluation", "🧬 Inside the pipeline"]
)

with tab_classify:
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown("#### Try it with real data")
        st.caption("Run the bundled five-row Kaggle sample without setting up a dataset.")
        sample = sample_csv_path(artifact)
        demo_a, demo_b = st.columns(2, gap="small")
        with demo_a:
            if st.button(
                "⚡ Run live 5-row demo", type="primary", use_container_width=True,
                disabled=sample is None, key="run_demo",
            ):
                try:
                    demo_frame = pd.read_csv(sample, low_memory=False).head(5)
                    if demo_frame.empty:
                        st.error("The included Kaggle sample has no data rows.")
                    else:
                        classify(predictor, demo_frame, "Bundled Kaggle sample")
                except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
                    st.error(f"Sample could not be read: {exc}")
        with demo_b:
            if sample is not None:
                st.download_button(
                    "↓ Download sample CSV",
                    data=sample.read_bytes(),
                    file_name="qtagger_demo_input.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_sample",
                )
            else:
                st.warning("Demo CSV is unavailable.")

        st.markdown('<div class="qt-mini-rule"></div>', unsafe_allow_html=True)
        st.markdown("#### Or classify your own CSV")
        st.caption(
            f"Provide up to {MAX_ROWS} rows · 5 MB maximum · "
            f"{len(predictor.raw_columns)} required numeric feature columns."
        )
        uploaded = st.file_uploader(
            "Upload extracted numeric memory features",
            type=["csv"],
            key="feature_csv",
            on_change=clear_result,
            help="Upload numeric features only; never upload executable malware or model files.",
        )
        if uploaded is not None:
            if uploaded.size > MAX_UPLOAD_BYTES:
                st.error("File exceeds 5 MB. Please use a smaller CSV.")
            else:
                try:
                    frame = pd.read_csv(uploaded, low_memory=False, nrows=MAX_ROWS + 1)
                    if frame.empty:
                        st.error("The CSV has no data rows.")
                    elif len(frame) > MAX_ROWS:
                        st.error(f"This CSV has more than {MAX_ROWS} records. Use a smaller batch.")
                    else:
                        missing = [column for column in predictor.raw_columns if column not in frame.columns]
                        if missing:
                            st.error(
                                f"Missing {len(missing)} required feature columns. "
                                "Use the bundled sample to match the expected schema."
                            )
                            with st.expander("Missing column names"):
                                st.code("\n".join(missing))
                        else:
                            st.success(f"Ready to classify {len(frame)} rows · {len(frame.columns)} columns")
                            with st.expander("Preview input data"):
                                st.dataframe(
                                    frame.iloc[:5, :min(8, len(frame.columns))],
                                    use_container_width=True, hide_index=True,
                                )
                                st.caption("Showing the first eight columns. All supplied columns are retained for download.")
                            if st.button("Classify uploaded records →", type="primary",
                                         use_container_width=True, key="run_uploaded"):
                                classify(predictor, frame, "Uploaded CSV")
                except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
                    st.error(f"Could not parse the CSV: {exc}")
    with right:
        st.markdown(
            f"""
<div class="qt-panel">
  <div class="qt-kicker">GUIDED WORKFLOW</div>
  <div class="qt-panel-title" style="margin-top:9px">From features to family labels</div>
  <div class="qt-step"><div class="qt-step-number">01</div>
    <div class="qt-step-text"><strong>Bring a feature CSV</strong>
      Use the genuine Kaggle sample or your own pre-extracted numerical memory features.</div></div>
  <div class="qt-step"><div class="qt-step-number">02</div>
    <div class="qt-step-text"><strong>Apply saved preprocessing</strong>
      The trained pipeline validates {len(predictor.raw_columns)} columns and encodes eight qubit angles.</div></div>
  <div class="qt-step"><div class="qt-step-number">03</div>
    <div class="qt-step-text"><strong>Predict & export</strong>
      View family-level labels and download the results as a CSV.</div></div>
  <div class="qt-mini-rule"></div>
  <div class="qt-small">Circuit simulator: {escape(predictor.device_name)}
    · {predictor.n_qubits} qubits · {len(predictor.classes)} output classes</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="qt-tip"><strong>Important:</strong> This is a research classification '
            'tool for pre-extracted numeric features, not an antivirus scanner. '
            'Do not upload executable malware, personal records, or sensitive datasets.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("View expected feature names"):
            st.code("\n".join(predictor.raw_columns))
    render_results()

with tab_evaluation:
    st.markdown("#### Held-out model comparison")
    st.caption(
        "These are saved evaluation results from the Kaggle training run. "
        "All figures are tied to the recorded subset and should not be interpreted as real-world detection rates."
    )
    if rows:
        valid = [item for item in rows if isinstance(item, dict)]
        table = pd.DataFrame(valid)
        if {"model", "accuracy", "macro_f1"}.issubset(table.columns):
            metric_data = table.set_index("model")[["accuracy", "macro_f1"]].rename(
                columns={"accuracy": "Accuracy", "macro_f1": "Macro-F1"}
            )
            st.bar_chart(metric_data, use_container_width=True)
            styled = table[["model", "accuracy", "macro_f1", "balanced_accuracy"]].copy()
            styled.columns = ["Model", "Accuracy", "Macro-F1", "Balanced accuracy"]
            for field in ("Accuracy", "Macro-F1", "Balanced accuracy"):
                styled[field] = styled[field].map(lambda v: f"{100 * float(v):.2f}%")
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)
        if qsvm and rf and float(rf.get("accuracy", 0)) > float(qsvm.get("accuracy", 0)):
            panel(
                "What the benchmark actually shows",
                "The Random Forest baseline performed better than QSVM v3.1 on this "
                "matched eight-feature held-out subset. The dashboard reports that "
                "difference transparently; the experiment does not establish quantum advantage.",
            )
    else:
        st.warning("No benchmark metrics were found in the saved training artifact.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Training subset", evaluation.get("n_train_subset", "—"))
    m2.metric("Test subset", evaluation.get("n_test_subset", "—"))
    m3.metric("Selected quantum features", predictor.n_qubits)
    if evaluation.get("dataset_sha256"):
        with st.expander("Dataset reproducibility and artifacts"):
            st.code(str(evaluation["dataset_sha256"]))
            st.caption("SHA-256 of the recorded source dataset.")
    figure = artifact.with_name("evaluation.png")
    if figure.is_file():
        st.image(str(figure), caption="Saved Kaggle evaluation figure", use_container_width=True)

with tab_pipeline:
    st.markdown("#### A reproducible hybrid quantum–classical workflow")
    st.caption("The interface loads the saved preprocessing objects and trained model; it does not retrain on your upload.")
    stages = [
        ("01 / DATA", "Validate & clean",
         "Check input column names, impute using training-only medians and apply the saved filtering and signed-log transform."),
        ("02 / FEATURES", "Scale & encode",
         "Apply fitted standard/robust scalers, select eight features and map values to qubit rotation angles."),
        ("03 / QUANTUM", "Local similarity measurement",
         "Run the corrected PennyLane circuit with learned KTA weights against the trained Nyström landmarks."),
        ("04 / CLASSIFIER", "SVM family prediction",
         "Project measured local similarities through the saved Nyström transform and classify with the trained linear SVM."),
    ]
    for idx in range(0, len(stages), 2):
        a, b = st.columns(2, gap="large")
        for col, (eyebrow, title, description) in zip((a, b), stages[idx:idx + 2]):
            with col:
                st.markdown(
                    f'<div class="qt-panel"><div class="qt-kicker">{escape(eyebrow)}</div>'
                    f'<div class="qt-panel-title" style="margin-top:8px">{escape(title)}</div>'
                    f'<div class="qt-panel-sub">{escape(description)}</div></div>',
                    unsafe_allow_html=True,
                )
    st.markdown("#### Scientific boundaries")
    st.markdown(
        "- The tool accepts extracted numerical CIC-MalMem-2022 features, not arbitrary executables.\n"
        "- The quantum local similarity is not guaranteed to be positive semidefinite.\n"
        "- Reported results use a limited held-out subset and are not production detection rates.\n"
        "- No calibrated confidence scores or claims of quantum superiority are supplied.\n"
        "- Uploaded CSV data is processed for the active session; models are never accepted as uploads."
    )
    st.link_button("Explore research and training notebook ↗", NOTEBOOK)

st.markdown(
    '<div class="qt-foot">QTagger+ · QIntern 2026 · Research use only'
    '<br>Pre-extracted memory-feature classification · Not a replacement for security software</div>',
    unsafe_allow_html=True,
)
