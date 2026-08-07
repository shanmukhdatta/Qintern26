"""
Day 2 — Locality Dose-Response Sweep
QTagger+ Consolidation Week, Team B (Shanmukh)

Generalizes V3's local-kernel measurement (previously hardcoded to pairs, g=2)
to an arbitrary non-overlapping group size g, and sweeps g in {1, 2, 4, 8} on
both CTGAN and Original datasets, WITHOUT KTA alignment (isolating the locality
variable only, per the Day 2 brief).

Reuses, unmodified in spirit, the following pieces of the existing V3 pipeline
(quantum_all_versions.py):
  - load_3class()      : dataset loading / 3-class balancing
  - base_filter()       : train/test split, variance + correlation filtering
  - project_xgboost()   : XGBoost top-8 feature selection (V3's projection)
  - run_qsvm_nystrom()  : Nystrom landmark QSVM (same runner V2/V3 already use
                           for the no-KTA case)

New in this script:
  - make_local_circuit_g(n_qubits, g, L)  : generalized grouped-measurement
    kernel (Section 3.2 of the brief). Same re-upload embedding (L=2 passes)
    and same per-group joint |0...0> fidelity formula as the existing g=2
    code, just with the group partition parametrized by g instead of
    hardcoded pairs.
  - gram_offdiag_std, kta_score, kta_chance_floor : diagnostics functions.
    NOTE (explicit assumption): the Day 1 brief that defines these functions
    was not attached to this Day 2 task, so they are re-implemented here from
    first principles, using the same ideal-target convention (+1 same class,
    -1 different class) that the existing kta_train() loss in
    quantum_all_versions.py already uses (lines 131-140) for consistency with
    the rest of the codebase. Diagnostics are computed on the landmark-vs-
    landmark Gram submatrix (K_MM), since the Nystrom runner never forms a
    full square training Gram matrix -- this is a deliberate, stated choice,
    not a hidden one.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import pennylane as qml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 7
N_QUBITS = 8
N_TOTAL = 500
L_REUPLOAD = 2
G_VALUES = [1, 2, 4, 8]
DATASETS = ["ctgan", "original"]
MAX_TRAIN = 300
MAX_TEST = 40
N_LANDMARKS = 40
KTA_CHANCE_TRIALS = 200

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATHS = {
    "ctgan": os.path.join(HERE, "..", "data", "ctgan.csv"),
    "original": os.path.join(HERE, "..", "data", "original.csv"),
}
RESULTS_DIR = os.path.join(HERE, "..", "results")
FIGURES_DIR = os.path.join(HERE, "..", "figures")


# ---------------------------------------------------------------------------
# Data loading / preprocessing (reused from quantum_all_versions.py)
# ---------------------------------------------------------------------------
def load_3class(dataset, n_total, seed=42):
    df = pd.read_csv(DATA_PATHS[dataset])
    if dataset == "ctgan":
        df["fam"] = df["Family"]
        drop_cols = ["Family"]
    elif dataset == "original":
        df = df[df["Category"] != "Benign"].copy()
        df["fam"] = df["Category"].str.split("-").str[0]
        drop_cols = ["Class", "Category", "Filename"]
    else:
        raise ValueError(dataset)
    per_class = n_total // 3
    parts = [
        g.sample(n=per_class, random_state=seed, replace=len(g) < per_class)
        for _, g in df.groupby("fam")
    ]
    sub = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    y = sub["fam"].values
    X = sub.drop(columns=[c for c in drop_cols if c in sub.columns] + ["fam"]).select_dtypes(
        include=[np.number]
    )
    return X, y


def base_filter(X, y, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    variances = Xtr.var()
    keep = variances[variances > 1e-8].index
    Xtr, Xte = Xtr[keep], Xte[keep]
    corr = Xtr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_c = [c for c in upper.columns if any(upper[c] > 0.95)]
    Xtr, Xte = Xtr.drop(columns=drop_c), Xte.drop(columns=drop_c)
    Xtr = np.sign(Xtr) * np.log1p(np.abs(Xtr))
    Xte = np.sign(Xte) * np.log1p(np.abs(Xte))
    classes = sorted(set(ytr))
    c2i = {c: i for i, c in enumerate(classes)}
    ytr_i = np.array([c2i[v] for v in ytr])
    yte_i = np.array([c2i[v] for v in yte])
    return Xtr, Xte, ytr_i, yte_i, classes


def project_xgboost(Xtr_s, Xte_s, ytr_i, n_qubits, seed=42):
    clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=seed, eval_metric="mlogloss")
    clf.fit(Xtr_s, ytr_i)
    top_idx = np.argsort(clf.feature_importances_)[::-1][:n_qubits]
    return Xtr_s[:, top_idx], Xte_s[:, top_idx]


# ---------------------------------------------------------------------------
# Generalized grouped-measurement local kernel (Section 3.2 of the brief)
# ---------------------------------------------------------------------------
def make_local_circuit_g(n_qubits, g, L=2):
    """
    Generalized version of V3's local-kernel measurement.

    g: group size. Must divide n_qubits evenly (checked below). Qubits are
       partitioned into n_qubits/g non-overlapping, contiguous groups; for
       each group we compute the joint |0...0> probability restricted to
       that group (summing/marginalizing over all other qubits), then
       average across groups. g=1 reduces to single-qubit marginals; g=8
       (n_qubits) reduces to the single full joint |00000000> probability,
       i.e. the original global kernel.

    Circuit (embedding) itself is UNCHANGED from the existing V3/V2
    re-upload kernel: L passes of AngleEmbedding + ring-CNOT entanglement
    for x1, mirrored (adjoint) for x2. No trainable weights are inserted
    here -- this is the no-KTA-alignment sweep per the brief.
    """
    assert n_qubits % g == 0, f"g={g} must divide n_qubits={n_qubits} evenly"
    dev = qml.device("lightning.qubit", wires=n_qubits)
    groups = [tuple(range(i, i + g)) for i in range(0, n_qubits, g)]

    @qml.qnode(dev)
    def probs_circuit(x1, x2):
        for _ in range(L):
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="Y")
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        for _ in range(L):
            for i in reversed(range(n_qubits)):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Y")
        return qml.probs(wires=range(n_qubits))

    # Precompute bitmask (over computational-basis indices) for each group
    # once per circuit, since it doesn't depend on x1/x2.
    n_states = 2 ** n_qubits
    group_masks = []
    for grp in groups:
        mask = np.array(
            [
                all((((idx >> (n_qubits - 1 - q)) & 1) == 0) for q in grp)
                for idx in range(n_states)
            ]
        )
        group_masks.append(mask)

    def local_kernel(x1, x2):
        probs = np.asarray(probs_circuit(x1, x2))
        vals = [probs[mask].sum() for mask in group_masks]
        return float(sum(vals) / len(vals))

    return local_kernel


# ---------------------------------------------------------------------------
# Nystrom QSVM runner (reused as-is from quantum_all_versions.py; this is the
# same runner V2/V3 use for the un-aligned case -- no w argument, no KTA)
# ---------------------------------------------------------------------------
def run_qsvm_nystrom(kernel_fn, Xtr, Xte, ytr, yte, n_landmarks=40, max_train=300,
                      max_test=40, C=1.0, seed=42):
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train:
        idx = rng.choice(len(Xtr), max_train, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > max_test:
        idx = rng.choice(len(Xte), max_test, replace=False)
        Xte, yte = Xte[idx], yte[idx]
    landmark_idx = rng.choice(len(Xtr), min(n_landmarks, len(Xtr)), replace=False)
    L_pts = Xtr[landmark_idx]
    L_labels = ytr[landmark_idx]

    t0 = time.time()
    K_train_land = np.array([[kernel_fn(a, b) for b in L_pts] for a in Xtr])
    K_test_land = np.array([[kernel_fn(a, b) for b in L_pts] for a in Xte])
    K_MM = np.array([[kernel_fn(a, b) for b in L_pts] for a in L_pts]) + 1e-6 * np.eye(len(L_pts))
    evals, evecs = np.linalg.eigh(K_MM)
    evals_clipped = np.clip(evals, 1e-8, None)
    K_MM_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals_clipped)) @ evecs.T
    psi_train = K_train_land @ K_MM_inv_sqrt
    psi_test = K_test_land @ K_MM_inv_sqrt
    clf = SVC(kernel="linear", C=C).fit(psi_train, ytr)
    pred = clf.predict(psi_test)
    kernel_time = time.time() - t0

    return {
        "acc": accuracy_score(yte, pred),
        "f1_macro": f1_score(yte, pred, average="macro"),
        "kernel_time_s": round(kernel_time, 1),
        "n_train_used": len(Xtr),
        "K_MM": K_MM,          # landmark-vs-landmark Gram (for diagnostics)
        "L_labels": L_labels,  # landmark labels (for diagnostics)
    }


# ---------------------------------------------------------------------------
# Diagnostics: gram_offdiag_std, kta_score, kta_chance_floor
# (re-implemented here; see module docstring for the explicit assumption
#  this rests on, since the Day 1 brief was not attached to this task)
# ---------------------------------------------------------------------------
def gram_offdiag_std(K):
    """Std of the off-diagonal (upper-triangular) entries of a symmetric Gram matrix."""
    n = K.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.std(K[iu]))


def _ideal_target_matrix(y):
    """+1 for same-class pairs, -1 for different-class pairs (matches the Y
    matrix already used in quantum_all_versions.py's kta_train(), lines 131-140)."""
    n = len(y)
    Y = np.array([[1.0 if y[i] == y[j] else -1.0 for j in range(n)] for i in range(n)])
    return Y


def kta_score(K, y):
    """Kernel-target alignment: <K, Y>_F / (||K||_F ||Y||_F)."""
    Y = _ideal_target_matrix(y)
    num = np.sum(K * Y)
    den = np.sqrt(np.sum(K * K) * np.sum(Y * Y)) + 1e-9
    return float(num / den)


def kta_chance_floor(K, y, n_trials=200, seed=42):
    """Mean KTA of the SAME Gram matrix K under random label permutations --
    i.e. the alignment score you'd expect from a kernel with no real relationship
    to the labels. Used as the baseline that 'kta_above_floor' = kta_score - this
    is measured against."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = []
    for _ in range(n_trials):
        y_perm = rng.permutation(y)
        scores.append(kta_score(K, y_perm))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Per-(g, dataset) run
# ---------------------------------------------------------------------------
def run_one(dataset, g, n_qubits=N_QUBITS, n_total=N_TOTAL, seed=SEED):
    X, y = load_3class(dataset, n_total, seed=seed)
    Xtr, Xte, ytr_i, yte_i, classes = base_filter(X, y, seed=seed)
    ss = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = ss.transform(Xtr), ss.transform(Xte)

    # V3 feature selection: XGBoost top-8
    Xtr_p, Xte_p = project_xgboost(Xtr_s, Xte_s, ytr_i, n_qubits, seed=seed)
    rs = RobustScaler().fit(Xtr_p)
    Xtr_r = np.clip(rs.transform(Xtr_p), -3, 3) / 3 * np.pi
    Xte_r = np.clip(rs.transform(Xte_p), -3, 3) / 3 * np.pi

    kernel_fn = make_local_circuit_g(n_qubits, g, L=L_REUPLOAD)
    res = run_qsvm_nystrom(
        kernel_fn, Xtr_r, Xte_r, ytr_i, yte_i,
        n_landmarks=N_LANDMARKS, max_train=MAX_TRAIN, max_test=MAX_TEST, seed=seed,
    )

    K_MM = res.pop("K_MM")
    L_labels = res.pop("L_labels")
    g_std = gram_offdiag_std(K_MM)
    kta = kta_score(K_MM, L_labels)
    floor = kta_chance_floor(K_MM, L_labels, n_trials=KTA_CHANCE_TRIALS, seed=seed)

    return {
        "g": g,
        "dataset": dataset,
        "qubits": n_qubits,
        "seed": seed,
        "accuracy": res["acc"],
        "macro_f1": res["f1_macro"],
        "gram_offdiag_std": g_std,
        "kta_score": kta,
        "kta_chance_floor": floor,
        "kta_above_floor": kta - floor,
        "kernel_time_s": res["kernel_time_s"],
        "n_train_used": res["n_train_used"],
        "classes": classes,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    all_results = []
    for dataset in DATASETS:
        for g in G_VALUES:
            print(f"Running g={g} dataset={dataset} ...", flush=True)
            rec = run_one(dataset, g)
            print(
                f"  -> acc={rec['accuracy']:.4f} f1={rec['macro_f1']:.4f} "
                f"gram_std={rec['gram_offdiag_std']:.4f} "
                f"kta_above_floor={rec['kta_above_floor']:.4f} "
                f"({rec['kernel_time_s']:.1f}s)",
                flush=True,
            )
            all_results.append(rec)

    out_path = os.path.join(RESULTS_DIR, "locality_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")
    return all_results


if __name__ == "__main__":
    main()
