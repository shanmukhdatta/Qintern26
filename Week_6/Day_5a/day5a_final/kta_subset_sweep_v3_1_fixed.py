"""
Day 5a — KTA Alignment Subset-Size Sensitivity Sweep (v3.1 -- BUG FIXED)
Team B (Shanmukh) · QIntern 2026

Re-run of the Day 5a sweep using pipeline_v3_1_fixed.py, which fixes two bugs
found while investigating the first sweep pass (see
Day5a_KTA_Subset_Sweep_Results.md for full detail):

  Bug 1 (circuit): RY(w) immediately followed by RY(-w) on the same wire
  composed to the identity for any w -- the "trainable kernel alignment"
  step had zero effect. Fixed by interleaving RY(w) inside each of the L
  data-encoding layers (standard trainable-quantum-kernel construction),
  verified: kernel now genuinely depends on w, and K(x,x)=1 still holds.

  Bug 2 (RNG coupling): train/test subsampling, the align-subset draw, and
  the landmark draw all shared one rng stream, so varying align_subset size
  silently changed which landmarks got used -- contaminating the very
  comparison this sweep is supposed to make. Fixed with independent,
  seed-derived RNG streams per sampling step; verified landmark_idx is now
  identical across all n.

Same config as the original sweep: n in {5,10,20,50}, CTGAN + Original,
q=8, seed=7, C=1.0, g=2 measurement grouping, xgboost feature selection.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_v3_1_fixed as p

N_QUBITS = 8
SEED = 7
C = 1.0
SUBSET_SIZES = [5, 10, 20, 50]
DATASETS = ["CTGAN", "Original"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(OUT_DIR, "kta_subset_sweep_results_v3_1_fixed.json")

# Old (buggy) reference, kept for explicit before/after comparison in the report
OLD_BUGGY_N10_CTGAN_ACC = 0.875
OLD_BUGGY_N10_CTGAN_F1 = 0.8732974910394264


def kta_score(local_kernel, X, y, w):
    n = len(X)
    Y = np.array([[1.0 if y[i] == y[j] else -1.0 for j in range(n)] for i in range(n)])
    K = np.array([[local_kernel(X[i], X[j], w) for j in range(n)] for i in range(n)])
    num = np.sum(K * Y)
    den = np.sqrt(np.sum(K * K) * np.sum(Y * Y)) + 1e-9
    return float(num / den)


def run_qsvm_v3_expose_w(Xtr, Xte, ytr, yte, n_qubits, align_subset, L=2,
                          n_landmarks=40, max_train=300, max_test=40,
                          align_iters=5, C=1.0, seed=42):
    """Same decoupled-RNG logic as pipeline_v3_1_fixed.run_qsvm_v3, but also
    returns local_kernel/w_trained/the exact test set used, so we can score
    KTA on the full held-out test set afterward."""
    rng_split = np.random.default_rng(seed)
    rng_align = np.random.default_rng(seed + 10_000)
    rng_land = np.random.default_rng(seed + 20_000)

    if len(Xtr) > max_train:
        idx = rng_split.choice(len(Xtr), max_train, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > max_test:
        idx = rng_split.choice(len(Xte), max_test, replace=False)
        Xte, yte = Xte[idx], yte[idx]

    local_kernel, pairs = p.make_local_trainable_circuit(n_qubits, L=L)

    align_idx = rng_align.choice(len(Xtr), min(align_subset, len(Xtr)), replace=False)
    w_trained, align_time = p.kernel_target_alignment_train(
        local_kernel, Xtr[align_idx], ytr[align_idx], n_qubits, iters=align_iters, seed=seed)

    landmark_idx = rng_land.choice(len(Xtr), min(n_landmarks, len(Xtr)), replace=False)
    L_pts = Xtr[landmark_idx]

    t0 = time.time()
    K_train_land = np.array([[local_kernel(a, b, w_trained) for b in L_pts] for a in Xtr])
    K_test_land = np.array([[local_kernel(a, b, w_trained) for b in L_pts] for a in Xte])
    K_MM = np.array([[local_kernel(a, b, w_trained) for b in L_pts] for a in L_pts]) + 1e-6 * np.eye(len(L_pts))
    evals, evecs = np.linalg.eigh(K_MM)
    evals = np.clip(evals, 1e-8, None)
    K_MM_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    psi_train = K_train_land @ K_MM_inv_sqrt
    psi_test = K_test_land @ K_MM_inv_sqrt
    kernel_time = time.time() - t0

    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score
    clf = SVC(kernel='linear', C=C).fit(psi_train, ytr)
    pred = clf.predict(psi_test)
    metrics = {
        "acc": accuracy_score(yte, pred), "f1_macro": f1_score(yte, pred, average='macro'),
        "align_time_s": round(align_time, 1), "kernel_time_s": round(kernel_time, 1),
        "n_train_used": len(Xtr), "n_landmarks": len(L_pts), "C": C,
        "w_trained_norm": float(np.linalg.norm(w_trained)),
    }
    return metrics, local_kernel, w_trained, Xte, yte


def prep_dataset(name, seed):
    if name == "CTGAN":
        X, y = p.load_ctgan(500, seed=seed)
    elif name == "Original":
        X, y = p.load_original(500, seed=seed)
    else:
        raise ValueError(name)
    Xtr_s, Xte_s, ytr_i, yte_i, classes = p.base_filter(X, y, seed=seed)
    Xtr_p, Xte_p = p.project_xgboost(Xtr_s, Xte_s, ytr_i, N_QUBITS, seed=seed)
    Xtr_r, Xte_r = p.angle_scale(Xtr_p, Xte_p)
    return Xtr_r, Xte_r, ytr_i, yte_i, classes


def main():
    all_records = []
    prepped = {}
    for ds in DATASETS:
        print(f"[prep] loading + featurizing {ds} (seed={SEED}, q={N_QUBITS})...", flush=True)
        prepped[ds] = prep_dataset(ds, SEED)
        print(f"[prep] {ds}: Xtr={prepped[ds][0].shape} Xte={prepped[ds][1].shape} "
              f"classes={prepped[ds][4]}", flush=True)

    for ds in DATASETS:
        Xtr_r, Xte_r, ytr_i, yte_i, classes = prepped[ds]
        for n in SUBSET_SIZES:
            t0 = time.time()
            metrics, local_kernel, w_trained, Xte_used, yte_used = run_qsvm_v3_expose_w(
                Xtr_r, Xte_r, ytr_i, yte_i, n_qubits=N_QUBITS, align_subset=n, C=C, seed=SEED)
            kta_full_test = kta_score(local_kernel, Xte_used, yte_used, w_trained)
            elapsed = time.time() - t0
            record = {
                "kta_subset_n": n, "dataset": ds, "qubits": N_QUBITS, "seed": SEED, "C": C,
                "accuracy": metrics["acc"], "macro_f1": metrics["f1_macro"],
                "kta_full_test": kta_full_test,
                "align_time_s": metrics["align_time_s"], "kernel_time_s": metrics["kernel_time_s"],
                "w_trained_norm": metrics["w_trained_norm"],
                "n_train_used": metrics["n_train_used"], "n_test_used": len(yte_used),
                "n_landmarks": metrics["n_landmarks"], "wall_time_s": round(elapsed, 1),
            }
            all_records.append(record)
            print(f"[{ds:>8}] n={n:<3} acc={metrics['acc']:.3f} f1={metrics['f1_macro']:.3f} "
                  f"kta_test={kta_full_test:.4f} w_norm={metrics['w_trained_norm']:.3f} ({elapsed:.0f}s)", flush=True)
            json.dump({"runs": all_records}, open(RESULTS_JSON, "w"), indent=2)

    print(f"\nDone. Results written to {RESULTS_JSON}")
    return all_records


if __name__ == "__main__":
    main()
