"""
Day 1 diagnostics: kernel-concentration metrics.

gram_offdiag_std : dispersion of the off-diagonal Gram entries (low => concentrated/
                    flat kernel -- the kernel-concentration failure mode).
kta_score         : Kernel-Target Alignment, using the SAME formula already used in
                    the project's alignment training step (ablation.py /
                    pipeline_v3_full.py train_alignment / kernel_target_alignment_train):
                        KTA(K, y) = <K, Y>_F / (||K||_F * ||Y||_F)
                    with Y_ij = +1 if y_i == y_j else -1. No second definition is
                    introduced here -- this is a direct copy of that formula.
kta_chance_floor  : mean KTA under label-permutation (null distribution), so that
                    KTA - chance_floor reports signal above the "no real structure"
                    baseline rather than raw KTA (which is bounded away from 0 even
                    for meaningless kernels once class sizes are unequal).
"""
import numpy as np


def gram_offdiag_std(K: np.ndarray) -> float:
    mask = ~np.eye(K.shape[0], dtype=bool)
    return float(np.std(K[mask]))


def kta_score(K: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y)
    n = len(y)
    Y = np.array([[1.0 if y[i] == y[j] else -1.0 for j in range(n)] for i in range(n)])
    num = np.sum(K * Y)
    den = np.sqrt(np.sum(K * K) * np.sum(Y * Y)) + 1e-9
    return float(num / den)


def kta_chance_floor(K: np.ndarray, y: np.ndarray, n_perm: int = 100, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = []
    for _ in range(n_perm):
        y_shuffled = rng.permutation(y)
        scores.append(kta_score(K, y_shuffled))
    return float(np.mean(scores))
