"""
Pipeline v2 - incorporating techniques from literature:
1. Hybrid LDA(<=C-1) + PCA(residual) projection (pure LDA caps at n_classes-1=3 for our 4-class problem,
   so we concat LDA(3) + PCA(q-3) to reach q=8/12 - noted as an explicit adaptation, not in the source papers
   since they had 23 classes and could use LDA directly up to q=8).
2. Data re-uploading feature map: L=2 reps of (AngleEmbedding + ring-entangling), vs single-pass in v1.
3. Nyström-landmark QSVM: use M=40 landmarks but evaluate against up to 300 training samples (vs v1's hard 80 cap).
4. COBYLA optimizer for VQC (gradient-free) instead of Adam/parameter-shift.
"""
import numpy as np, time, json, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import minimize
import pennylane as qml

def load_and_split(n_total, seed=42, path='/mnt/user-data/uploads/MalMem2022_SMOTE.csv'):
    df = pd.read_csv(path)
    per_class = n_total // 4
    parts = [g.sample(n=per_class, random_state=seed) for _, g in df.groupby('Label')]
    sub = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sub.drop(columns=['Label']), sub['Label'].values

def preprocess_v2(X, y, n_qubits, seed=42):
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
    n_classes = len(classes)
    lda_dim = min(n_classes - 1, n_qubits)  # LDA hard cap
    lda = LinearDiscriminantAnalysis(n_components=lda_dim).fit(Xtr_s, ytr)
    Xtr_lda, Xte_lda = lda.transform(Xtr_s), lda.transform(Xte_s)

    remaining = n_qubits - lda_dim
    if remaining > 0:
        pca = PCA(n_components=remaining, random_state=seed).fit(Xtr_s)
        Xtr_pca, Xte_pca = pca.transform(Xtr_s), pca.transform(Xte_s)
        Xtr_p = np.hstack([Xtr_lda, Xtr_pca]); Xte_p = np.hstack([Xte_lda, Xte_pca])
    else:
        Xtr_p, Xte_p = Xtr_lda, Xte_lda

    # robust scale + clip to [-pi, pi] (paper's approach, vs plain minmax in v1)
    rs = RobustScaler().fit(Xtr_p)
    Xtr_r, Xte_r = rs.transform(Xtr_p), rs.transform(Xte_p)
    Xtr_r = np.clip(Xtr_r, -3, 3) / 3 * np.pi
    Xte_r = np.clip(Xte_r, -3, 3) / 3 * np.pi
    return Xtr_r, Xte_r, np.array(ytr), np.array(yte), classes, lda_dim

# ---- Re-uploading feature map kernel (L reps) ----
def make_reupload_kernel(n_qubits, L=2):
    dev = qml.device('lightning.qubit', wires=n_qubits)
    @qml.qnode(dev)
    def kcircuit(x1, x2):
        for _ in range(L):
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i+1) % n_qubits])
        for _ in range(L):
            for i in reversed(range(n_qubits)):
                qml.CNOT(wires=[i, (i+1) % n_qubits])
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
        return qml.probs(wires=range(n_qubits))
    return lambda a, b: kcircuit(a, b)[0]

def run_qsvm_nystrom(Xtr, Xte, ytr, yte, n_qubits, L=2, n_landmarks=40, max_train=300, max_test=40, seed=42):
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train:
        idx = rng.choice(len(Xtr), max_train, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > max_test:
        idx = rng.choice(len(Xte), max_test, replace=False); Xte, yte = Xte[idx], yte[idx]
    landmark_idx = rng.choice(len(Xtr), min(n_landmarks, len(Xtr)), replace=False)
    L_pts = Xtr[landmark_idx]

    k = make_reupload_kernel(n_qubits, L=L)
    t0 = time.time()
    K_train_land = np.array([[k(a, b) for b in L_pts] for a in Xtr])   # N x M
    K_test_land = np.array([[k(a, b) for b in L_pts] for a in Xte])    # Ntest x M
    # explicit Nystrom features via K_MM^{-1/2}
    K_MM = np.array([[k(a, b) for b in L_pts] for a in L_pts])
    K_MM += 1e-6 * np.eye(len(L_pts))
    evals, evecs = np.linalg.eigh(K_MM)
    evals = np.clip(evals, 1e-8, None)
    K_MM_inv_sqrt = evecs @ np.diag(1.0/np.sqrt(evals)) @ evecs.T
    psi_train = K_train_land @ K_MM_inv_sqrt
    psi_test = K_test_land @ K_MM_inv_sqrt

    clf = SVC(kernel='linear').fit(psi_train, ytr)
    pred = clf.predict(psi_test)
    acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, average='macro')
    return {'acc': acc, 'f1_macro': f1, 'time_s': time.time()-t0, 'n_train_used': len(Xtr), 'n_landmarks': len(L_pts)}

# ---- Re-uploading VQC with COBYLA ----
def run_vqc_v2(Xtr, Xte, ytr, yte, n_qubits, classes, L=2, max_train=200, max_iter=60, seed=42):
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train:
        idx = rng.choice(len(Xtr), max_train, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    ytr = np.array([class_to_idx[v] for v in ytr])
    yte = np.array([class_to_idx[v] for v in yte])
    n_classes = len(classes); n_out = min(n_classes, n_qubits)
    dev = qml.device('lightning.qubit', wires=n_qubits)
    n_layers = 2

    @qml.qnode(dev)
    def circuit(x, weights):
        for _ in range(L):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_out)]

    shape = (n_layers, n_qubits, 3)
    n_params = int(np.prod(shape))
    w0 = 0.1 * rng.standard_normal(n_params)
    y_oh = np.eye(n_classes)[ytr][:, :n_out] if n_out < n_classes else np.eye(n_classes)[ytr]

    def loss(w_flat):
        w = w_flat.reshape(shape)
        total = 0.0
        for x, yv in zip(Xtr, y_oh):
            out = np.array(circuit(x, w))
            p = np.exp(out - out.max()); p = p / p.sum()
            total += -np.sum(yv * np.log(p + 1e-9))
        return total / len(Xtr)

    t0 = time.time()
    maxiter_eff = max(max_iter, n_params + 20)
    res = minimize(loss, w0, method='COBYLA', options={'maxiter': maxiter_eff, 'rhobeg': 0.5})
    w_final = res.x.reshape(shape)

    preds = []
    for x in Xte:
        out = np.array(circuit(x, w_final))
        preds.append(np.argmax(out))
    preds = np.array(preds)
    acc = accuracy_score(yte, preds); f1 = f1_score(yte, preds, average='macro')
    return {'acc': acc, 'f1_macro': f1, 'time_s': time.time()-t0, 'n_train_used': len(Xtr), 'iters': max_iter}

if __name__ == '__main__':
    n_total = int(sys.argv[1]); n_qubits = int(sys.argv[2])
    X, y = load_and_split(n_total)
    Xtr, Xte, ytr, yte, classes, lda_dim = preprocess_v2(X, y, n_qubits)
    print(f"n={n_total} q={n_qubits} lda_dim={lda_dim} pca_residual={n_qubits-lda_dim} train={len(Xtr)} test={len(Xte)}", flush=True)

    qsvm_r = run_qsvm_nystrom(Xtr, Xte, ytr, yte, n_qubits=n_qubits, L=2, n_landmarks=40, max_train=300, max_test=40)
    print(f"QSVM_v2 n={n_total} q={n_qubits} acc={qsvm_r['acc']:.3f} f1={qsvm_r['f1_macro']:.3f} time={qsvm_r['time_s']:.0f}s", flush=True)

    vqc_r = run_vqc_v2(Xtr, Xte, ytr, yte, n_qubits=n_qubits, classes=classes, L=2, max_train=200, max_iter=60)
    print(f"VQC_v2 n={n_total} q={n_qubits} acc={vqc_r['acc']:.3f} f1={vqc_r['f1_macro']:.3f} time={vqc_r['time_s']:.0f}s", flush=True)

    import os
    out = 'results_v2.json'
    data = json.load(open(out)) if os.path.exists(out) else {'runs': []}
    data['runs'] = [r for r in data['runs'] if not (r['n_total']==n_total and r['qubits']==n_qubits)]
    data['runs'].append({'n_total': n_total, 'qubits': n_qubits, 'lda_dim': lda_dim,
                          'qsvm_v2': qsvm_r, 'vqc_v2': vqc_r})
    json.dump(data, open(out, 'w'), indent=2)
    print("SAVED", flush=True)
