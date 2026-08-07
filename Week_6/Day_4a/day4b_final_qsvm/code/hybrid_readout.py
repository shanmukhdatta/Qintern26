"""
Day 4b - Hybrid Readout Stacking (QTagger+ Consolidation Week, Team B)

Interception point (per brief Section 3.1): V3's pipeline builds
`psi_train` / `psi_test` -- the Nystrom-projected features from the local,
KTA-"aligned" kernel -- and currently feeds them straight into a linear SVM.
Nothing upstream of that line changes here: same load_ctgan/load_original,
same base_filter, same project_xgboost (q=8), same angle_scale, same
make_local_trainable_circuit + kernel_target_alignment_train, same Nystrom
landmarks (M=40). The only change is what consumes psi_train/psi_test at the
very end.

NOTE on "KTA-aligned": Day 1's diagnostics found that the RY(w)/RY(-w) block
in make_local_trainable_circuit is a no-op (RY(-w) . RY(w) = identity for any
w), so the alignment step does not actually change the kernel. That bug is
untouched here (out of scope for today), but it means "local, KTA-aligned
kernel" in this pipeline is currently equivalent to "local kernel, unaligned."
Readout-head comparisons below are still valid -- they test what happens
downstream of the kernel, which is unaffected by that bug -- but the kernel
itself is not doing what its name implies.
"""
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v3_full import (
    load_ctgan, base_filter, project_xgboost, angle_scale,
    make_local_trainable_circuit, kernel_target_alignment_train,
)
from data_original import load_original

warnings.filterwarnings('ignore')

N_QUBITS = 8
SEED = 7
N_TOTAL = 500
MAX_TRAIN = 300
MAX_TEST = 40
N_LANDMARKS = 40
ALIGN_SUBSET = 10
ALIGN_ITERS = 5
EXTRA_SEEDS = [1, 42]  # for the 3-seed robustness check, matches seeds already used elsewhere in the project


def build_nystrom_features(dataset_name, n_qubits=N_QUBITS, seed=SEED, L=2):
    """Reproduces run_qsvm_v3's pipeline exactly, up to and including
    psi_train/psi_test -- the interception point specified in the brief."""
    if dataset_name == 'CTGAN':
        X, y = load_ctgan(N_TOTAL, seed=seed)
    elif dataset_name == 'Original':
        X, y = load_original(N_TOTAL, seed=seed)
    else:
        raise ValueError(dataset_name)

    Xtr_s, Xte_s, ytr_i, yte_i, classes = base_filter(X, y, seed=seed)
    Xtr_p, Xte_p = project_xgboost(Xtr_s, Xte_s, ytr_i, n_qubits, seed=seed)
    Xtr_r, Xte_r = angle_scale(Xtr_p, Xte_p)

    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = Xtr_r, ytr_i, Xte_r, yte_i
    if len(Xtr) > MAX_TRAIN:
        idx = rng.choice(len(Xtr), MAX_TRAIN, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > MAX_TEST:
        idx = rng.choice(len(Xte), MAX_TEST, replace=False)
        Xte, yte = Xte[idx], yte[idx]

    local_kernel, pairs = make_local_trainable_circuit(n_qubits, L=L)
    align_idx = rng.choice(len(Xtr), min(ALIGN_SUBSET, len(Xtr)), replace=False)
    w_trained, _ = kernel_target_alignment_train(local_kernel, Xtr[align_idx], ytr[align_idx],
                                                  n_qubits, iters=ALIGN_ITERS, seed=seed)

    landmark_idx = rng.choice(len(Xtr), min(N_LANDMARKS, len(Xtr)), replace=False)
    L_pts = Xtr[landmark_idx]

    K_train_land = np.array([[local_kernel(a, b, w_trained) for b in L_pts] for a in Xtr])
    K_test_land = np.array([[local_kernel(a, b, w_trained) for b in L_pts] for a in Xte])
    K_MM = np.array([[local_kernel(a, b, w_trained) for b in L_pts] for a in L_pts]) + 1e-6 * np.eye(len(L_pts))
    evals, evecs = np.linalg.eigh(K_MM)
    evals = np.clip(evals, 1e-8, None)
    K_MM_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    psi_train = K_train_land @ K_MM_inv_sqrt
    psi_test = K_test_land @ K_MM_inv_sqrt

    return psi_train, psi_test, ytr, yte


def make_heads(seed=SEED):
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    return {
        'linear_svm': lambda: SVC(kernel='linear', C=1.0, random_state=seed),
        'mlp': lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=500,
                                      random_state=seed, early_stopping=False),
        'xgboost': lambda: XGBClassifier(n_estimators=100, max_depth=3, random_state=seed,
                                          eval_metric='mlogloss'),
        'logreg': lambda: LogisticRegression(max_iter=1000, random_state=seed),
    }


def run_head(head_name, psi_train, psi_test, ytr, yte, seed=SEED):
    from sklearn.metrics import accuracy_score, f1_score
    clf = make_heads(seed=seed)[head_name]()
    t0 = time.time()
    clf.fit(psi_train, ytr)
    train_pred = clf.predict(psi_train)
    test_pred = clf.predict(psi_test)
    train_acc = accuracy_score(ytr, train_pred)
    test_acc = accuracy_score(yte, test_pred)
    test_f1 = f1_score(yte, test_pred, average='macro')
    return {
        'readout': head_name, 'train_acc': train_acc, 'test_acc': test_acc,
        'macro_f1': test_f1, 'overfit_gap': train_acc - test_acc,
        'time_s': round(time.time() - t0, 1),
    }


def run_dataset(dataset_name, n_qubits=N_QUBITS, seed=SEED):
    psi_train, psi_test, ytr, yte = build_nystrom_features(dataset_name, n_qubits=n_qubits, seed=seed)
    rows = []
    for head_name in ['linear_svm', 'mlp', 'xgboost', 'logreg']:
        r = run_head(head_name, psi_train, psi_test, ytr, yte, seed=seed)
        r['dataset'] = dataset_name
        r['seed'] = seed
        rows.append(r)
        print(f"[{dataset_name:>8} | seed={seed:<3} | {head_name:>11}] "
              f"train_acc={r['train_acc']:.3f} test_acc={r['test_acc']:.3f} "
              f"f1={r['macro_f1']:.3f} overfit_gap={r['overfit_gap']:+.3f} ({r['time_s']}s)", flush=True)
    return rows


if __name__ == '__main__':
    all_rows = []
    for dataset_name in ['Original', 'CTGAN']:  # Original first, per project priority
        all_rows += run_dataset(dataset_name, seed=SEED)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'day4b_hybrid_readout_seed7_FIXED.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({'config': {'qubits': N_QUBITS, 'seed': SEED, 'n_total': N_TOTAL,
                           'max_train': MAX_TRAIN, 'max_test': MAX_TEST,
                           'n_landmarks': N_LANDMARKS, 'align_subset': ALIGN_SUBSET,
                           'align_iters': ALIGN_ITERS},
               'rows': all_rows}, open(out_path, 'w'), indent=2)
    print(f"\nSAVED -> {out_path}", flush=True)
