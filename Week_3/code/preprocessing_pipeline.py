import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---- knobs -------------------------------------------------------------
IN_PATH        = "C:\\AI\\QTagger+\\malmem_ctgan.csv"
LABEL_COL      = "Family"
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
VAR_THRESH     = 1e-4     
CORR_THRESH    = 0.95     
QUBIT_BUDGET   = 8        
OUT_DIR        = "C:\\AI\\QTagger+\\outputs"
# ------------------------------------------------------------------------

# ---- load & split --------------------------------------------------------
df = pd.read_csv(IN_PATH)
y = df[LABEL_COL]
X = df.drop(columns=[LABEL_COL])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

class CorrelationFilter(BaseEstimator, TransformerMixin):

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.to_drop_ = [c for c in upper.columns if any(upper[c] > self.threshold)]
        self.keep_ = [c for c in X.columns if c not in self.to_drop_]
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X[self.keep_].to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_, dtype=object)

# ---- pipeline definition --------------------------------------------------
pipeline = Pipeline(steps=[
    ("variance_filter", VarianceThreshold(threshold=VAR_THRESH)),
    ("correlation_filter", CorrelationFilter(threshold=CORR_THRESH)),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=QUBIT_BUDGET, random_state=RANDOM_STATE)),
])


pipeline.fit(X_train)

X_train_p = pipeline.transform(X_train)
X_test_p = pipeline.transform(X_test)

# ---- persist ---------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

np.save(f"{OUT_DIR}/X_train_pca.npy", X_train_p)
np.save(f"{OUT_DIR}/X_test_pca.npy", X_test_p)
y_train.to_csv(f"{OUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUT_DIR}/y_test.csv", index=False)

# one artifact for the whole preprocessing pipeline (industry norm)
joblib.dump(pipeline, f"{OUT_DIR}/preprocess_pipeline.joblib")

# ---- recover intermediate diagnostics from fitted steps -------------------
var_step = pipeline.named_steps["variance_filter"]
corr_step = pipeline.named_steps["correlation_filter"]
pca_step = pipeline.named_steps["pca"]

keep_var = X_train.columns[var_step.get_support()].tolist()
dropped_var = [c for c in X.columns if c not in keep_var]
dropped_corr = corr_step.to_drop_
keep_corr = corr_step.keep_

# one artifact for the whole preprocessing pipeline (industry norm)
joblib.dump(pipeline, f"{OUT_DIR}/preprocess_pipeline.joblib")

report = {
    "input_rows": int(len(df)),
    "input_features": int(X.shape[1]),
    "dropped_variance_filter": dropped_var,
    "dropped_correlation_filter": dropped_corr,
    "features_after_filters": keep_corr,
    "n_features_final_before_pca": len(keep_corr),
    "qubit_budget": QUBIT_BUDGET,
    "pca_explained_variance_ratio": pca_step.explained_variance_ratio_.tolist(),
    "pca_cumulative_variance": float(np.sum(pca_step.explained_variance_ratio_)),
    "train_shape": list(X_train_p.shape),
    "test_shape": list(X_test_p.shape),
    "class_balance_train": y_train.value_counts().to_dict(),
    "class_balance_test": y_test.value_counts().to_dict(),
}
with open(f"{OUT_DIR}/pipeline_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))