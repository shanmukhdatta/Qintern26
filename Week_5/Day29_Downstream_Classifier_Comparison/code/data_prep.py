"""
data_prep.py
Loads CIC-MalMem-2022, isolates the malware-family label (Category), and builds
a per-class bounded representation (Standardize -> PCA(N_QUBITS) -> scale to [-1,1])
that the quantum generator operates in. Also provides the inverse transform back
to the original 54-feature space so generated samples slot into the real schema.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import config


def load_raw(path: str = config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])
    # Derive the 4-class malware TYPE (Benign/Ransomware/Spyware/Trojan) from the
    # fine-grained family name in Category (e.g. "Spyware-Gator" -> "Spyware").
    df[config.TYPE_LABEL_COL] = df["Category"].str.split("-").str[0]
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {config.LABEL_COL, config.BINARY_LABEL_COL}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


class ClassRepresentation:
    """Fits Standardize -> PCA(N_QUBITS) -> [-1,1] scaling for one malware family,
    and can invert generated latent-space points back to the original feature space."""

    def __init__(self, n_components: int = config.N_QUBITS):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
        self.abs_max = None  # per-component max-abs value used for [-1,1] bounding

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_std = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_std)
        self.abs_max = np.maximum(np.abs(X_pca).max(axis=0), 1e-8)
        X_norm = X_pca / self.abs_max
        return np.clip(X_norm, -1.0, 1.0)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        X_pca = X_norm * self.abs_max
        X_std = self.pca.inverse_transform(X_pca)
        X_orig = self.scaler.inverse_transform(X_std)
        return X_orig


def split_by_class(df: pd.DataFrame, feature_cols: list) -> dict:
    """Returns {class_name: X (n_samples, n_features) ndarray} for every value in LABEL_COL."""
    out = {}
    for cls, group in df.groupby(config.LABEL_COL):
        out[cls] = group[feature_cols].to_numpy(dtype=np.float64)
    return out


def classes_to_augment(df: pd.DataFrame) -> list:
    counts = df[config.LABEL_COL].value_counts()
    return [c for c in counts.index if c not in config.EXCLUDE_FROM_AUGMENTATION]


def target_count(df: pd.DataFrame) -> int:
    if config.TARGET_SAMPLES_PER_CLASS is not None:
        return config.TARGET_SAMPLES_PER_CLASS
    counts = df[config.LABEL_COL].value_counts()
    malware_counts = counts.drop(labels=config.EXCLUDE_FROM_AUGMENTATION, errors="ignore")
    return int(malware_counts.max())
