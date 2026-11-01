"""Reconstruct and run the exact QSVM v3.1 Kaggle notebook inference pipeline.

Only load trusted joblib artifacts: joblib/pickle deserialization executes code.
Uploaded user data must be CSV; never deserialize uploaded model files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pennylane as qml


class ModelCompatibilityError(ValueError):
    """A model bundle is incomplete or incompatible with this inference code."""


REQUIRED_KEYS = {
    "format_version", "classes", "raw_feature_columns", "fill_values",
    "kept_columns", "standard_scaler", "selected_indices", "robust_scaler",
    "quantum_config", "kta_weights", "landmarks", "K_mm_inv_sqrt", "svm",
}


class QTaggerPredictor:
    """Trusted local model, train-time transformations and PennyLane circuit."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        if not isinstance(bundle, dict):
            raise ModelCompatibilityError("Expected a dictionary model bundle.")
        missing = REQUIRED_KEYS - bundle.keys()
        if missing:
            raise ModelCompatibilityError(f"Missing model bundle fields: {sorted(missing)}")
        if not str(bundle["format_version"]).startswith("qtagger-v3.1-"):
            raise ModelCompatibilityError("Unsupported bundle format; use the final training notebook.")
        self.bundle = bundle
        self.classes = list(bundle["classes"])
        self.raw_columns = list(bundle["raw_feature_columns"])
        self.kept_columns = list(bundle["kept_columns"])
        self.selected_indices = np.asarray(bundle["selected_indices"], dtype=int)
        self.weights = np.asarray(bundle["kta_weights"], dtype=float)
        self.landmarks = np.asarray(bundle["landmarks"], dtype=float)
        self.inverse_sqrt = np.asarray(bundle["K_mm_inv_sqrt"], dtype=float)
        self.n_qubits = int(bundle["quantum_config"]["n_qubits"])
        self.layers = int(bundle["quantum_config"]["layers"])
        self.pairs = [tuple(p) for p in bundle["quantum_config"]["pairs"]]
        if (not self.classes or not self.raw_columns or not self.kept_columns
                or self.n_qubits < 2 or self.layers < 1):
            raise ModelCompatibilityError("Invalid class, feature or circuit configuration.")
        if len(self.selected_indices) != self.n_qubits or len(self.weights) != self.n_qubits:
            raise ModelCompatibilityError("Selected feature and rotation weight dimensions mismatch.")
        if self.landmarks.ndim != 2 or self.landmarks.shape[1] != self.n_qubits:
            raise ModelCompatibilityError("Landmarks have the wrong number of qubits.")
        m = len(self.landmarks)
        if self.inverse_sqrt.shape != (m, m):
            raise ModelCompatibilityError("Nyström matrix has the wrong dimensions.")
        if not np.isfinite(self.landmarks).all() or not np.isfinite(self.weights).all():
            raise ModelCompatibilityError("Model contains non-finite quantum parameters.")
        if any(a < 0 or b < 0 or a >= self.n_qubits or b >= self.n_qubits or a == b
               for a, b in self.pairs):
            raise ModelCompatibilityError("Invalid measurement pair.")
        self.device_name = "default.qubit"
        try:
            dev = qml.device("lightning.qubit", wires=self.n_qubits)
            self.device_name = "lightning.qubit"
        except Exception:
            dev = qml.device("default.qubit", wires=self.n_qubits)

        nq, depth = self.n_qubits, self.layers

        @qml.qnode(dev)
        def probs(x1: np.ndarray, x2: np.ndarray, w: np.ndarray):
            for _ in range(depth):
                qml.AngleEmbedding(x1, wires=range(nq), rotation="Y")
                for i in range(nq):
                    qml.RY(w[i], wires=i)
                for i in range(nq):
                    qml.CNOT(wires=[i, (i + 1) % nq])
            for _ in range(depth):
                for i in reversed(range(nq)):
                    qml.CNOT(wires=[i, (i + 1) % nq])
                for i in range(nq):
                    qml.RY(-w[i], wires=i)
                qml.adjoint(qml.AngleEmbedding)(x2, wires=range(nq), rotation="Y")
            return qml.probs(wires=range(nq))

        self._probs = probs
        ids = np.arange(2 ** nq)
        self._masks = [
            ((((ids >> (nq - 1 - a)) & 1) == 0) &
             (((ids >> (nq - 1 - b)) & 1) == 0))
            for a, b in self.pairs
        ]
        if not self._masks:
            raise ModelCompatibilityError("No measurement pairs in saved model.")

    @classmethod
    def from_file(cls, path: str | Path) -> "QTaggerPredictor":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        # Never call this on user-uploaded or otherwise untrusted .joblib files.
        return cls(joblib.load(path))

    def prepare_features(self, frame: pd.DataFrame) -> np.ndarray:
        """Validate raw numerical CSV and reproduce all training-time transforms."""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("Provide a CSV containing at least one data row.")
        if frame.columns.duplicated().any():
            raise ValueError("CSV contains duplicate column names.")
        missing = [c for c in self.raw_columns if c not in frame.columns]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required numerical feature columns: {missing[:12]}. "
                "Use the demo CSV generated by the training notebook as a template."
            )
        raw = frame.loc[:, self.raw_columns]
        numeric = raw.apply(pd.to_numeric, errors="coerce")
        invalid = raw.notna() & numeric.isna()
        if invalid.any().any():
            bad = invalid.columns[invalid.any(axis=0)].tolist()
            raise ValueError(f"These columns contain non-numeric values: {bad[:12]}")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        fill = pd.Series(self.bundle["fill_values"], dtype=float)
        numeric = numeric.fillna(fill)
        if numeric.isna().any().any():
            raise ValueError("Missing feature values could not be imputed.")
        x = numeric.loc[:, self.kept_columns]
        x = np.sign(x) * np.log1p(np.abs(x))
        z = self.bundle["standard_scaler"].transform(x)
        z = z[:, self.selected_indices]
        z = self.bundle["robust_scaler"].transform(z)
        z = np.clip(z, -3, 3) / 3 * np.pi
        if z.shape[1] != self.n_qubits or not np.isfinite(z).all():
            raise ModelCompatibilityError("Invalid transformed feature dimensions or values.")
        return np.asarray(z, dtype=float)

    def _local_similarity(self, x: np.ndarray, landmark: np.ndarray) -> float:
        probs = np.asarray(self._probs(x, landmark, self.weights), dtype=float)
        return float(np.mean([probs[mask].sum() for mask in self._masks]))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict coarse malware family labels; no probability is fabricated."""
        angles = self.prepare_features(frame)
        cross = np.empty((len(angles), len(self.landmarks)), dtype=float)
        for i, x in enumerate(angles):
            for j, landmark in enumerate(self.landmarks):
                cross[i, j] = self._local_similarity(x, landmark)
        psi = cross @ self.inverse_sqrt
        indices = np.asarray(self.bundle["svm"].predict(psi), dtype=int)
        if np.any(indices < 0) or np.any(indices >= len(self.classes)):
            raise ModelCompatibilityError("Predicted class index is outside saved label range.")
        return np.asarray(self.classes, dtype=object)[indices]
