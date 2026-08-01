"""
evaluate.py
Fidelity metrics (KS, Wasserstein, MMD) comparing real vs QGAN-synthetic samples
per malware family, and a downstream classifier comparison (RF, XGBoost,
LightGBM, SVM) trained on Original-only vs QGAN-augmented data, evaluated on a
held-out real test split.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import config


def mmd_rbf(X: np.ndarray, Y: np.ndarray, max_samples: int = 500) -> float:
    """Maximum Mean Discrepancy with an RBF kernel (median-heuristic bandwidth)."""
    rng = np.random.RandomState(config.RANDOM_STATE)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    Z = np.vstack([X, Y])
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    sigma2 = np.median(d2[d2 > 0]) + 1e-8
    K = np.exp(-d2 / (2 * sigma2))

    n, m = len(X), len(Y)
    Kxx = K[:n, :n]
    Kyy = K[n:, n:]
    Kxy = K[:n, n:]
    return float(Kxx.sum() / (n * n) + Kyy.sum() / (m * m) - 2 * Kxy.sum() / (n * m))


def fidelity_report(real: np.ndarray, synth: np.ndarray, feature_cols: list, class_name: str) -> dict:
    ks_vals, wd_vals = [], []
    for j in range(real.shape[1]):
        ks_vals.append(ks_2samp(real[:, j], synth[:, j]).statistic)
        wd_vals.append(wasserstein_distance(real[:, j], synth[:, j]))
    return {
        "class": class_name,
        "n_real": len(real),
        "n_synth": len(synth),
        "KS_median": float(np.median(ks_vals)),
        "KS_p75": float(np.percentile(ks_vals, 75)),
        "Wasserstein_median": float(np.median(wd_vals)),
        "Wasserstein_p75": float(np.percentile(wd_vals, 75)),
        "MMD": mmd_rbf(real, synth),
    }


SVM_MAX_TRAIN_ROWS = 6000  # RBF-kernel SVC does not scale to tens of thousands of rows in reasonable time


def _make_classifiers():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, eval_metric="mlogloss",
            random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1),
        "SVM": SVC(kernel="rbf", probability=False, random_state=config.RANDOM_STATE),
    }


def _subsample_for_svm(X, y, max_rows=SVM_MAX_TRAIN_ROWS):
    if len(X) <= max_rows:
        return X, y
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = rng.choice(len(X), max_rows, replace=False)
    return X[idx], y[idx]


def downstream_comparison(df_original: pd.DataFrame, df_augmented: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Trains each classifier on (a) Original-only and (b) QGAN-augmented data, and
    evaluates both on the SAME held-out real test split for a fair comparison."""
    le = LabelEncoder()
    le.fit(df_original[config.LABEL_COL])

    X_orig = df_original[feature_cols].to_numpy(dtype=np.float64)
    y_orig = le.transform(df_original[config.LABEL_COL])

    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X_orig, y_orig, test_size=config.DOWNSTREAM_TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y_orig,
    )

    # augmented training set = original TRAIN split + the synthetic-only rows
    synth_only = df_augmented[df_augmented["is_synthetic"] == 1]
    X_synth = synth_only[feature_cols].to_numpy(dtype=np.float64)
    y_synth = le.transform(synth_only[config.LABEL_COL])
    X_train_aug = np.vstack([X_train_orig, X_synth])
    y_train_aug = np.concatenate([y_train_orig, y_synth])

    scaler = StandardScaler().fit(X_train_orig)
    X_test_s = scaler.transform(X_test)

    rows = []
    for name, clf_factory in _make_classifiers().items():
        Xo_fit, yo_fit = scaler.transform(X_train_orig), y_train_orig
        Xa_fit, ya_fit = X_train_aug, y_train_aug
        if name == "SVM":
            Xo_fit, yo_fit = _subsample_for_svm(Xo_fit, yo_fit)

        # Original-only
        clf_o = _make_classifiers()[name]
        clf_o.fit(Xo_fit, yo_fit)
        pred_o = clf_o.predict(X_test_s)

        # QGAN-augmented
        scaler_aug = StandardScaler().fit(X_train_aug)
        Xa_fit_s = scaler_aug.transform(Xa_fit)
        if name == "SVM":
            Xa_fit_s, ya_fit = _subsample_for_svm(Xa_fit_s, ya_fit)
        clf_a = _make_classifiers()[name]
        clf_a.fit(Xa_fit_s, ya_fit)
        pred_a = clf_a.predict(scaler_aug.transform(X_test))

        rows.append({
            "classifier": name,
            "accuracy_original": accuracy_score(y_test, pred_o),
            "f1_macro_original": f1_score(y_test, pred_o, average="macro"),
            "accuracy_qgan_augmented": accuracy_score(y_test, pred_a),
            "f1_macro_qgan_augmented": f1_score(y_test, pred_a, average="macro"),
        })
    return pd.DataFrame(rows)
