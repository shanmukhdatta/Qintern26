"""
QSVM v3 on EMBER -- SAME architecture as pipeline_v3_full.py (the script that
produced the 0.875 MalMem/CTGAN result). Every function below is copied
verbatim from pipeline_v3_full.py:

    base_filter, project_xgboost, angle_scale, make_local_trainable_circuit,
    kernel_target_alignment_train, run_qsvm_v3, run_classical

The ONLY thing that changes is the data loader: load_ctgan() (3-class,
family-stratified sampling from a CSV) is replaced by load_ember() (binary,
label-stratified sampling from an EMBER feature file), because that's the only
part of the pipeline that is dataset-specific. No architecture, hyperparameter,
kernel design, or classifier logic has been altered.

Best known config from the MalMem run (used as defaults here too):
  variant='xgboost', n_qubits=8, seed=7, C=1.0, n_total=500
"""
import sys, json, os, time, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pennylane as qml
from pennylane import numpy as pnp
from xgboost import XGBClassifier

# ---------------- EMBER-specific loader (only new piece) ----------------
EMBER_PATH_ENV = os.environ.get("EMBER_PATH", "")

def _autodetect_ember_path():
    if EMBER_PATH_ENV and os.path.exists(EMBER_PATH_ENV):
        return EMBER_PATH_ENV
    candidates = []
    for base in ("/kaggle/input", "/mnt/user-data/uploads"):
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if "ember" in f.lower() and f.lower().endswith((".csv", ".parquet")):
                    candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(
            "No EMBER file auto-detected under /kaggle/input or /mnt/user-data/uploads. "
            "Set the EMBER_PATH environment variable to the exact file path."
        )
    return candidates[0]

def load_ember(n_total, seed=42, path=None):
    """Binary EMBER loader, stratified sampling to mirror load_ctgan()'s
    per-class balanced sampling (there it was per-family; here it's per binary
    label). Drops EMBER's -1 (unlabeled) rows per EMBER convention."""
    path = path or _autodetect_ember_path()
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    label_col = None
    for cand in ["label", "Label", "y", "target", "class"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError("Could not find a label column in the EMBER file -- expected one of "
                          "label/Label/y/target/class.")

    if df[label_col].min() < 0:
        df = df[df[label_col] >= 0].reset_index(drop=True)

    df["fam"] = df[label_col]  # reuse the same 'fam' grouping key as load_ctgan for base_filter compat
    per_class = n_total // 2
    parts = [g.sample(n=per_class, random_state=seed, replace=len(g) < per_class)
             for _, g in df.groupby("fam")]
    sub = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    y = sub["fam"].values
    X = sub.drop(columns=[label_col, "fam"]).select_dtypes(include=[np.number])
    return X, y

# ---------------- Everything below is UNCHANGED from pipeline_v3_full.py ----------------

def base_filter(X, y, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    variances = Xtr.var(); keep = variances[variances > 1e-8].index
    Xtr, Xte = Xtr[keep], Xte[keep]
    corr = Xtr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_c = [c for c in upper.columns if any(upper[c] > 0.95)]
    Xtr, Xte = Xtr.drop(columns=drop_c), Xte.drop(columns=drop_c)
    Xtr = np.sign(Xtr) * np.log1p(np.abs(Xtr)); Xte = np.sign(Xte) * np.log1p(np.abs(Xte))
    ss = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = ss.transform(Xtr), ss.transform(Xte)
    classes = sorted(set(ytr))
    c2i = {c: i for i, c in enumerate(classes)}
    ytr_i = np.array([c2i[v] for v in ytr]); yte_i = np.array([c2i[v] for v in yte])
    return Xtr_s, Xte_s, ytr_i, yte_i, classes

def project_hybrid(Xtr_s, Xte_s, ytr_i, n_qubits, seed=42):
    n_classes = len(set(ytr_i))
    lda_dim = min(n_classes - 1, n_qubits)
    lda = LinearDiscriminantAnalysis(n_components=lda_dim).fit(Xtr_s, ytr_i)
    Xtr_lda, Xte_lda = lda.transform(Xtr_s), lda.transform(Xte_s)
    remaining = n_qubits - lda_dim
    if remaining > 0:
        pca = PCA(n_components=remaining, random_state=seed).fit(Xtr_s)
        Xtr_p = np.hstack([Xtr_lda, pca.transform(Xtr_s)]); Xte_p = np.hstack([Xte_lda, pca.transform(Xte_s)])
    else:
        Xtr_p, Xte_p = Xtr_lda, Xte_lda
    return Xtr_p, Xte_p

def project_xgboost(Xtr_s, Xte_s, ytr_i, n_qubits, seed=42):
    clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=seed, eval_metric='mlogloss')
    clf.fit(Xtr_s, ytr_i)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n_qubits]
    return Xtr_s[:, top_idx], Xte_s[:, top_idx]

def angle_scale(Xtr_p, Xte_p):
    rs = RobustScaler().fit(Xtr_p)
    Xtr_r = np.clip(rs.transform(Xtr_p), -3, 3) / 3 * np.pi
    Xte_r = np.clip(rs.transform(Xte_p), -3, 3) / 3 * np.pi
    return Xtr_r, Xte_r

# ---------------- Local, trainable kernel ----------------
def make_local_trainable_circuit(n_qubits, L=2):
    dev = qml.device('lightning.qubit', wires=n_qubits)
    pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]  # non-overlapping pairs

    @qml.qnode(dev)
    def probs_circuit(x1, x2, w):
        for _ in range(L):
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
            for i in range(n_qubits): qml.CNOT(wires=[i, (i + 1) % n_qubits])
        for i in range(n_qubits):
            qml.RY(w[i], wires=i)          # trainable parameters (kernel alignment)
        for i in range(n_qubits):
            qml.RY(-w[i], wires=i)         # adjoint of trainable block before uncomputing x2
        for _ in range(L):
            for i in reversed(range(n_qubits)): qml.CNOT(wires=[i, (i + 1) % n_qubits])
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
        return qml.probs(wires=range(n_qubits))

    def local_kernel(x1, x2, w):
        probs = probs_circuit(x1, x2, w)
        n_states = len(probs)
        vals = []
        for (a, b) in pairs:
            mask = np.array([(((idx >> (n_qubits - 1 - a)) & 1) == 0) and (((idx >> (n_qubits - 1 - b)) & 1) == 0)
                              for idx in range(n_states)])
            vals.append(probs[mask].sum())
        return sum(vals) / len(vals)

    return local_kernel, pairs

def kernel_target_alignment_train(local_kernel, X_sub, y_sub, n_qubits, iters=8, lr=0.3, seed=42):
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    n = len(X_sub)
    Y = np.array([[1.0 if y_sub[i] == y_sub[j] else -1.0 for j in range(n)] for i in range(n)])

    def kta_loss(w):
        K = np.array([[local_kernel(X_sub[i], X_sub[j], w) for j in range(n)] for i in range(n)])
        num = np.sum(K * Y)
        den = np.sqrt(np.sum(K * K) * np.sum(Y * Y)) + 1e-9
        return -(num / den)

    w0 = 0.05 * rng.standard_normal(n_qubits)
    t0 = time.time()
    maxiter_eff = max(iters, n_qubits + 2)
    res = minimize(kta_loss, w0, method='COBYLA', options={'maxiter': maxiter_eff, 'rhobeg': 0.3})
    return res.x, time.time() - t0

# ---------------- Nystrom QSVM using the local, trained kernel ----------------
def run_qsvm_v3(Xtr, Xte, ytr, yte, n_qubits, L=2, n_landmarks=40, max_train=300, max_test=40,
                 align_subset=10, align_iters=5, C=1.0, seed=42):
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train:
        idx = rng.choice(len(Xtr), max_train, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > max_test:
        idx = rng.choice(len(Xte), max_test, replace=False); Xte, yte = Xte[idx], yte[idx]

    local_kernel, pairs = make_local_trainable_circuit(n_qubits, L=L)

    align_idx = rng.choice(len(Xtr), min(align_subset, len(Xtr)), replace=False)
    w_trained, align_time = kernel_target_alignment_train(local_kernel, Xtr[align_idx], ytr[align_idx],
                                                            n_qubits, iters=align_iters, seed=seed)

    landmark_idx = rng.choice(len(Xtr), min(n_landmarks, len(Xtr)), replace=False)
    L_pts = Xtr[landmark_idx]

    t0 = time.time()
    K_train_land = np.array([[local_kernel(a, b, w_trained) for b in L_pts] for a in Xtr])
    K_test_land = np.array([[local_kernel(a, b, w_trained) for b in L_pts] for a in Xte])
    K_MM = np.array([[local_kernel(a, b, w_trained) for b in L_pts] for a in L_pts]) + 1e-6 * np.eye(len(L_pts))
    evals, evecs = np.linalg.eigh(K_MM); evals = np.clip(evals, 1e-8, None)
    K_MM_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    psi_train = K_train_land @ K_MM_inv_sqrt
    psi_test = K_test_land @ K_MM_inv_sqrt
    kernel_time = time.time() - t0

    clf = SVC(kernel='linear', C=C).fit(psi_train, ytr)
    pred = clf.predict(psi_test)
    return {"acc": accuracy_score(yte, pred), "f1_macro": f1_score(yte, pred, average='macro'),
            "align_time_s": round(align_time, 1), "kernel_time_s": round(kernel_time, 1),
            "n_train_used": len(Xtr), "n_landmarks": len(L_pts), "C": C,
            "w_trained_norm": float(np.linalg.norm(w_trained))}

def run_classical(Xtr, Xte, ytr, yte, max_train=80, max_test=40, seed=42):
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train:
        idx = rng.choice(len(Xtr), max_train, replace=False); Xtr_s, ytr_s = Xtr[idx], ytr[idx]
    else:
        Xtr_s, ytr_s = Xtr, ytr
    if len(Xte) > max_test:
        idx = rng.choice(len(Xte), max_test, replace=False); Xte_s, yte_s = Xte[idx], yte[idx]
    else:
        Xte_s, yte_s = Xte, yte
    svm = SVC(kernel='rbf').fit(Xtr_s, ytr_s); p = svm.predict(Xte_s)
    svm_r = {"acc": accuracy_score(yte_s, p), "f1_macro": f1_score(yte_s, p, average='macro')}
    rf = RandomForestClassifier(n_estimators=200, random_state=seed).fit(Xtr_s, ytr_s); p = rf.predict(Xte_s)
    rf_r = {"acc": accuracy_score(yte_s, p), "f1_macro": f1_score(yte_s, p, average='macro')}
    return svm_r, rf_r

def run_one(variant, n_qubits, seed, n_total=500, C=1.0, out_path='results_ember_v3.json', ember_path=None):
    X, y = load_ember(n_total, seed=seed, path=ember_path)
    Xtr_s, Xte_s, ytr_i, yte_i, classes = base_filter(X, y, seed=seed)
    if variant == 'hybrid':
        Xtr_p, Xte_p = project_hybrid(Xtr_s, Xte_s, ytr_i, n_qubits, seed=seed)
    elif variant == 'xgboost':
        Xtr_p, Xte_p = project_xgboost(Xtr_s, Xte_s, ytr_i, n_qubits, seed=seed)
    else:
        raise ValueError(variant)
    Xtr_r, Xte_r = angle_scale(Xtr_p, Xte_p)
    svm_r, rf_r = run_classical(Xtr_r, Xte_r, ytr_i, yte_i, seed=seed)
    qsvm_r = run_qsvm_v3(Xtr_r, Xte_r, ytr_i, yte_i, n_qubits=n_qubits, C=C, seed=seed)
    record = {"variant": variant, "qubits": n_qubits, "seed": seed, "n_total": n_total, "C": C,
              "dataset": "EMBER", "classical_svm": svm_r, "classical_rf": rf_r, "qsvm_v3": qsvm_r}
    data = json.load(open(out_path)) if os.path.exists(out_path) else {"runs": []}
    key = (variant, n_qubits, seed, C)
    data["runs"] = [r for r in data["runs"] if (r["variant"], r["qubits"], r["seed"], r["C"]) != key]
    data["runs"].append(record)
    json.dump(data, open(out_path, "w"), indent=2)
    print(f"[{variant:>7}] q={n_qubits:<3} seed={seed:<3} C={C} | clSVM={svm_r['acc']:.3f} clRF={rf_r['acc']:.3f} "
          f"| QSVM_v3={qsvm_r['acc']:.3f} f1={qsvm_r['f1_macro']:.3f}", flush=True)
    return record

if __name__ == '__main__':
    variant = sys.argv[1] if len(sys.argv) > 1 else 'xgboost'
    n_qubits = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    C = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    run_one(variant, n_qubits, seed, C=C)
