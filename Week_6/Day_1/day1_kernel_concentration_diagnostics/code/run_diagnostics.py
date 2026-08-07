"""
Day 1 - Kernel-Concentration Diagnostics (QTagger+ Consolidation Week, Team B)

Hooks into the existing v2/v3 kernel-construction and RNG sequence used by the
project's ablation script, so that:
  - the Gram matrix reflects each model's real forward pass, and
  - v2 and v3 are diagnosed on the *same* held-out test batch.

v2 (global kernel, no alignment)  == ablation config "A"
v3 (local kernel,  with alignment) == ablation config "D"

No new training run is performed; this reuses the Gram-matrix machinery the
eval loop already builds, computed on the held-out test batch instead of
against the Nystrom landmarks.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v3_full import load_ctgan, base_filter, project_xgboost, angle_scale
from data_original import load_original
from diagnostics import gram_offdiag_std, kta_score, kta_chance_floor
from circuit import make_circuit as make_circuit_fixed

N_QUBITS = 8
SEED = 7
N_TOTAL = 500
MAX_TRAIN = 300
MAX_TEST = 40
ALIGN_SUBSET = 10
ALIGN_ITERS = 5
N_PERM = 100


def make_circuit(n_qubits, L=2):
    """Delegates to circuit_fixed.make_circuit -- the corrected circuit."""
    return make_circuit_fixed(n_qubits, L=L)


def train_alignment(kernel_fn, X_sub, y_sub, n_qubits, iters=ALIGN_ITERS, seed=SEED):
    """Identical to ablation.train_alignment -- reuses the project's KTA formula."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    n = len(X_sub)
    Y = np.array([[1.0 if y_sub[i] == y_sub[j] else -1.0 for j in range(n)] for i in range(n)])

    def kta_loss(w):
        K = np.array([[kernel_fn(X_sub[i], X_sub[j], w) for j in range(n)] for i in range(n)])
        return -(np.sum(K * Y) / (np.sqrt(np.sum(K * K) * np.sum(Y * Y)) + 1e-9))

    w0 = 0.05 * rng.standard_normal(n_qubits)
    res = minimize(kta_loss, w0, method='COBYLA', options={'maxiter': max(iters, n_qubits + 2), 'rhobeg': 0.3})
    return res.x


def prepare_dataset(dataset_name, seed=SEED, n_qubits=N_QUBITS):
    if dataset_name == 'CTGAN':
        X, y = load_ctgan(N_TOTAL, seed=seed)
    elif dataset_name == 'Original':
        X, y = load_original(N_TOTAL, seed=seed)
    else:
        raise ValueError(dataset_name)
    Xtr_s, Xte_s, ytr_i, yte_i, classes = base_filter(X, y, seed=seed)
    Xtr_p, Xte_p = project_xgboost(Xtr_s, Xte_s, ytr_i, n_qubits, seed=seed)
    Xtr_r, Xte_r = angle_scale(Xtr_p, Xte_p)
    return Xtr_r, Xte_r, ytr_i, yte_i, classes


def get_common_test_batch(Xtr, Xte, ytr, yte, seed=SEED):
    """Replicates the exact RNG sequence of ablation.run_variant up to (and
    including) the test-batch subsample, so v2/v3 diagnostics use the identical
    test batch that the real accuracy eval used -- and the two calls below run
    identically regardless of use_alignment, since the alignment branch comes
    strictly after them."""
    rng = np.random.default_rng(seed)
    if len(Xtr) > MAX_TRAIN:
        idx = rng.choice(len(Xtr), MAX_TRAIN, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > MAX_TEST:
        idx = rng.choice(len(Xte), MAX_TEST, replace=False)
        Xte, yte = Xte[idx], yte[idx]
    return rng, Xtr, Xte, ytr, yte


def build_gram(kernel_fn, w, X_batch):
    n = len(X_batch)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j < i:
                K[i, j] = K[j, i]
            else:
                K[i, j] = kernel_fn(X_batch[i], X_batch[j], w)
    return K


def run_one(model, dataset_name, n_qubits=N_QUBITS, seed=SEED):
    Xtr, Xte, ytr, yte, classes = prepare_dataset(dataset_name, seed=seed, n_qubits=n_qubits)
    global_kernel, local_kernel = make_circuit(n_qubits, L=2)

    rng, Xtr_b, Xte_b, ytr_b, yte_b = get_common_test_batch(Xtr, Xte, ytr, yte, seed=seed)

    if model == 'v2':  # == ablation config A: global kernel, no alignment
        kernel_fn = global_kernel
        w = np.zeros(n_qubits)
    elif model == 'v3':  # == ablation config D: local kernel, with alignment
        kernel_fn = local_kernel
        align_idx = rng.choice(len(Xtr_b), min(ALIGN_SUBSET, len(Xtr_b)), replace=False)
        w = train_alignment(local_kernel, Xtr_b[align_idx], ytr_b[align_idx], n_qubits, seed=seed)
    else:
        raise ValueError(model)

    t0 = time.time()
    K = build_gram(kernel_fn, w, Xte_b)
    build_time = time.time() - t0

    g_std = gram_offdiag_std(K)
    kta = kta_score(K, yte_b)
    floor = kta_chance_floor(K, yte_b, n_perm=N_PERM, seed=0)

    row = {
        'model': 'v2 (global)' if model == 'v2' else 'v3 (local)',
        'dataset': dataset_name,
        'qubits': n_qubits,
        'seed': seed,
        'test_batch_size': len(Xte_b),
        'gram_offdiag_std': g_std,
        'kta': kta,
        'kta_chance_floor': floor,
        'kta_above_floor': kta - floor,
        'gram_build_time_s': round(build_time, 1),
    }
    print(f"[{row['model']:>12} | {dataset_name:>8}] gram_std={g_std:.4f}  "
          f"KTA={kta:.4f}  floor={floor:.4f}  KTA-floor={kta - floor:.4f}  "
          f"(n={row['test_batch_size']}, {build_time:.0f}s)", flush=True)
    return row


if __name__ == '__main__':
    rows = []
    # Original first per brief's priority note (paper's surviving claim),
    # then CTGAN, ordered v2 then v3 for readability.
    for dataset_name in ['Original', 'CTGAN']:
        for model in ['v2', 'v3']:
            rows.append(run_one(model, dataset_name))

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'day1_kernel_diagnostics_fixed.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({'config': {'qubits': N_QUBITS, 'seed': SEED, 'n_total': N_TOTAL,
                           'max_train': MAX_TRAIN, 'max_test': MAX_TEST,
                           'align_subset': ALIGN_SUBSET, 'align_iters': ALIGN_ITERS,
                           'n_perm': N_PERM},
               'rows': rows}, open(out_path, 'w'), indent=2)
    print(f"\nSAVED -> {out_path}", flush=True)
