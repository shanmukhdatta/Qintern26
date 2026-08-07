"""
QSVM V3 pipeline -- multi-seed test.

Runs the full V3 pipeline (XGBoost feature selection -> local trainable
kernel, alignment rotation applied once between forward/adjoint halves ->
Nystrom projection -> linear SVM) across multiple seeds, so a single-seed
result (e.g. seed=7) can be checked against seed variance rather than taken
at face value.

Usage:
    python3 run_seeds.py CTGAN 8 0 1 2 3 4 5 6 7
    python3 run_seeds.py Original 8 0 1 2 3 4 5 6 7
"""
import sys
import os
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v3_full import load_ctgan, base_filter, project_xgboost, angle_scale, run_qsvm_v3
from data_original import load_original


def run_one_seed(dataset, n_qubits, seed, n_total=500, C=1.0):
    if dataset == 'CTGAN':
        X, y = load_ctgan(n_total, seed=seed)
    elif dataset == 'Original':
        X, y = load_original(n_total, seed=seed)
    else:
        raise ValueError(dataset)
    Xtr_s, Xte_s, ytr_i, yte_i, classes = base_filter(X, y, seed=seed)
    Xtr_p, Xte_p = project_xgboost(Xtr_s, Xte_s, ytr_i, n_qubits, seed=seed)
    Xtr_r, Xte_r = angle_scale(Xtr_p, Xte_p)
    t0 = time.time()
    qsvm_r = run_qsvm_v3(Xtr_r, Xte_r, ytr_i, yte_i, n_qubits=n_qubits, C=C, seed=seed)
    qsvm_r['dataset'] = dataset
    qsvm_r['seed'] = seed
    qsvm_r['qubits'] = n_qubits
    qsvm_r['total_time_s'] = round(time.time() - t0, 1)
    print(f"[{dataset:>8} | seed={seed:<3}] acc={qsvm_r['acc']:.3f} f1={qsvm_r['f1_macro']:.3f} "
          f"({qsvm_r['total_time_s']}s)", flush=True)
    return qsvm_r


if __name__ == '__main__':
    dataset = sys.argv[1]
    n_qubits = int(sys.argv[2])
    seeds = [int(s) for s in sys.argv[3:]]

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', f'qsvm_v3_{dataset.lower()}_seeds.json')
    existing = json.load(open(out_path)) if os.path.exists(out_path) else []
    done_seeds = {r['seed'] for r in existing}

    rows = existing
    for seed in seeds:
        if seed in done_seeds:
            print(f"[{dataset:>8} | seed={seed:<3}] already done, skipping", flush=True)
            continue
        r = run_one_seed(dataset, n_qubits, seed)
        rows.append(r)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(rows, open(out_path, 'w'), indent=2)  # save after every seed

    accs = [r['acc'] for r in rows]
    print(f"\n{dataset} q={n_qubits}: {len(rows)} seeds -> mean={np.mean(accs):.3f} std={np.std(accs):.3f} "
          f"min={min(accs):.3f} max={max(accs):.3f}")
    print(f"SAVED -> {out_path}")
