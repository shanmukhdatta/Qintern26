"""
Ablation on the winning config (xgboost, q=8, seed=7, C=1.0):
  A: global kernel, w=0 (no alignment)      -- closest to v2's kernel design
  B: local kernel,  w=0 (no alignment)      -- local kernel effect alone
  C: global kernel, w=trained (alignment)   -- alignment effect alone
  D: local kernel,  w=trained (alignment)   -- both together (= the 0.875 result)
"""
import sys, json, time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import minimize
import pennylane as qml
sys.path.insert(0, '.')
from pipeline_v3_full import load_ctgan, base_filter, project_xgboost, angle_scale

def make_circuit(n_qubits, L=2):
    dev = qml.device('lightning.qubit', wires=n_qubits)
    pairs = [(i, i+1) for i in range(0, n_qubits - 1, 2)]

    @qml.qnode(dev)
    def probs_circuit(x1, x2, w):
        for _ in range(L):
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
            for i in range(n_qubits): qml.CNOT(wires=[i, (i+1) % n_qubits])
        for i in range(n_qubits): qml.RY(w[i], wires=i)
        for i in range(n_qubits): qml.RY(-w[i], wires=i)
        for _ in range(L):
            for i in reversed(range(n_qubits)): qml.CNOT(wires=[i, (i+1) % n_qubits])
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
        return qml.probs(wires=range(n_qubits))

    def global_kernel(x1, x2, w):
        return probs_circuit(x1, x2, w)[0]

    def local_kernel(x1, x2, w):
        probs = probs_circuit(x1, x2, w)
        n_states = len(probs)
        vals = []
        for (a, b) in pairs:
            mask = np.array([(((idx >> (n_qubits-1-a)) & 1)==0) and (((idx >> (n_qubits-1-b)) & 1)==0)
                              for idx in range(n_states)])
            vals.append(probs[mask].sum())
        return sum(vals) / len(vals)
    return global_kernel, local_kernel

def train_alignment(kernel_fn, X_sub, y_sub, n_qubits, iters=10, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X_sub)
    Y = np.array([[1.0 if y_sub[i]==y_sub[j] else -1.0 for j in range(n)] for i in range(n)])
    def kta_loss(w):
        K = np.array([[kernel_fn(X_sub[i], X_sub[j], w) for j in range(n)] for i in range(n)])
        return -(np.sum(K*Y) / (np.sqrt(np.sum(K*K)*np.sum(Y*Y)) + 1e-9))
    w0 = 0.05 * rng.standard_normal(n_qubits)
    res = minimize(kta_loss, w0, method='COBYLA', options={'maxiter': max(iters, n_qubits+2), 'rhobeg': 0.3})
    return res.x

def run_variant(kernel_fn, use_alignment, Xtr, Xte, ytr, yte, n_qubits, n_landmarks=40,
                 max_train=300, max_test=40, seed=42):
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train: idx = rng.choice(len(Xtr), max_train, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > max_test: idx = rng.choice(len(Xte), max_test, replace=False); Xte, yte = Xte[idx], yte[idx]

    if use_alignment:
        align_idx = rng.choice(len(Xtr), min(10, len(Xtr)), replace=False)
        w = train_alignment(kernel_fn, Xtr[align_idx], ytr[align_idx], n_qubits, seed=seed)
    else:
        w = np.zeros(n_qubits)

    landmark_idx = rng.choice(len(Xtr), min(n_landmarks, len(Xtr)), replace=False)
    L_pts = Xtr[landmark_idx]
    K_train_land = np.array([[kernel_fn(a, b, w) for b in L_pts] for a in Xtr])
    K_test_land  = np.array([[kernel_fn(a, b, w) for b in L_pts] for a in Xte])
    K_MM = np.array([[kernel_fn(a, b, w) for b in L_pts] for a in L_pts]) + 1e-6*np.eye(len(L_pts))
    evals, evecs = np.linalg.eigh(K_MM); evals = np.clip(evals, 1e-8, None)
    K_MM_inv_sqrt = evecs @ np.diag(1.0/np.sqrt(evals)) @ evecs.T
    psi_train = K_train_land @ K_MM_inv_sqrt
    psi_test  = K_test_land @ K_MM_inv_sqrt
    clf = SVC(kernel='linear', C=1.0).fit(psi_train, ytr)
    pred = clf.predict(psi_test)
    return accuracy_score(yte, pred), f1_score(yte, pred, average='macro')

if __name__ == '__main__':
    label = sys.argv[1]  # A, B, C, or D
    n_qubits, seed = 8, 7
    X, y = load_ctgan(500, seed=seed)
    Xtr_s, Xte_s, ytr_i, yte_i, classes = base_filter(X, y, seed=seed)
    Xtr_p, Xte_p = project_xgboost(Xtr_s, Xte_s, ytr_i, n_qubits, seed=seed)
    Xtr_r, Xte_r = angle_scale(Xtr_p, Xte_p)
    global_kernel, local_kernel = make_circuit(n_qubits, L=2)

    configs = {
        'A': (global_kernel, False, "global kernel, NO alignment"),
        'B': (local_kernel,  False, "LOCAL kernel, no alignment"),
        'C': (global_kernel, True,  "global kernel, WITH alignment"),
        'D': (local_kernel,  True,  "LOCAL kernel, WITH alignment (=full v3)"),
    }
    kernel_fn, use_align, desc = configs[label]
    t0 = time.time()
    acc, f1 = run_variant(kernel_fn, use_align, Xtr_r, Xte_r, ytr_i, yte_i, n_qubits, seed=seed)
    print(f"[{label}] {desc}: acc={acc:.3f} f1={f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
    import os
    out = 'ablation_results.json'
    data = json.load(open(out)) if os.path.exists(out) else {}
    data[label] = {"desc": desc, "acc": acc, "f1_macro": f1}
    json.dump(data, open(out, 'w'), indent=2)
