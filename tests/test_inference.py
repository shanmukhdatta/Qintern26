"""Lightweight tests using a synthetic in-memory model, never malware binaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from qtagger.inference import ModelCompatibilityError, QTaggerPredictor


@pytest.fixture(scope="module")
def predictor():
    raw = np.array([[1., 2.], [2., 3.], [4., 5.], [5., 6.]])
    ss = StandardScaler().fit(np.log1p(raw))
    z = ss.transform(np.log1p(raw))
    rs = RobustScaler().fit(z)
    svc = SVC(kernel="linear").fit(
        np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        np.array([0, 1, 0, 1]),
    )
    bundle = {
        "format_version": "qtagger-v3.1-1",
        "task": "test",
        "classes": ["Ransomware", "Trojan"],
        "raw_feature_columns": ["a", "b"],
        "fill_values": {"a": 2.0, "b": 3.0},
        "kept_columns": ["a", "b"],
        "standard_scaler": ss,
        "selected_indices": np.array([0, 1]),
        "robust_scaler": rs,
        "quantum_config": {"n_qubits": 2, "layers": 1, "pairs": [(0, 1)]},
        "kta_weights": np.array([0.1, -0.2]),
        "landmarks": np.array([[0.0, 0.0], [0.4, -0.3]]),
        "K_mm_inv_sqrt": np.eye(2),
        "svm": svc,
    }
    return QTaggerPredictor(bundle)


def test_angles_finite_and_bounded(predictor):
    angles = predictor.prepare_features(pd.DataFrame({"b": [3., 4.], "a": [1., 2.]}))
    assert angles.shape == (2, 2)
    assert np.isfinite(angles).all()
    assert (np.abs(angles) <= np.pi + 1e-12).all()


def test_missing_required_column(predictor):
    with pytest.raises(ValueError, match="Missing"):
        predictor.predict(pd.DataFrame({"a": [1.]}))


def test_non_numeric_rejected(predictor):
    with pytest.raises(ValueError, match="non-numeric"):
        predictor.prepare_features(pd.DataFrame({"a": ["malformed"], "b": [3.]}))


def test_imputation_and_extra_columns(predictor):
    data = pd.DataFrame({"a": [np.nan], "b": [3.], "unrelated": ["ignored"]})
    angles = predictor.prepare_features(data)
    assert np.isfinite(angles).all()


def test_predict_labels(predictor):
    labels = predictor.predict(pd.DataFrame({"a": [1., 5.], "b": [2., 6.]}))
    assert len(labels) == 2
    assert set(labels).issubset(set(predictor.classes))


def test_reject_invalid_bundle():
    with pytest.raises(ModelCompatibilityError, match="Missing"):
        QTaggerPredictor({"format_version": "qtagger-v3.1-1"})
