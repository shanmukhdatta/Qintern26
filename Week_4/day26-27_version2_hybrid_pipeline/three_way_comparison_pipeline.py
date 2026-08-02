"""
Unified 3-way comparison: SMOTE vs CTGAN vs Original, quantum (QSVM/VQC v2) vs classical.
Uses the winning config discovered in Week 4: LDA+PCA hybrid projection, data re-uploading
feature map (L=2), Nystrom-approximated QSVM kernel, and VQC trained with re-upload + Adam
(NOT COBYLA - isolation test showed COBYLA regresses, Adam is the correct optimizer choice).

Class scope: restricted to the 3 malware families (Ransomware/Spyware/Trojan) uniformly
across all three datasets. This is a deliberate, disclosed choice - CTGAN's own dataset only
contains these 3 classes (no Benign), and Benign was already shown to be trivially separable
(F1=0.9999 in the CTGAN classical run), so this scopes the comparison to the actually-hard
family-tagging problem rather than diluting it with an easy Benign/malware split.
"""
import sys, json, os, time
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

DATA_PATHS = {
    'smote': '/mnt/user-data/uploads/MalMem2022_SMOTE__1_.csv',
    'ctgan': '/mnt/user-data/uploads/malmem_ctgan__1_.csv',
    'original': '/mnt/user-data/uploads/malmem_original.csv',
}

def load_3class(dataset, n_total, seed=42):
    path = DATA_PATHS[dataset]
    df = pd.read_csv(path)
    if dataset == 'smote':
        df = df[df['Label'] != 'Benign'].copy()
        df['fam'] = df['Label']
        drop_cols = ['Label']
    elif dataset == 'ctgan':
        df['fam'] = df['Family']
        drop_cols = ['Family']
    elif dataset == 'original':
        df = df[df['Category'] != 'Benign'].copy()
        df['fam'] = df['Category'].str.split('-').str[0]
        drop_cols = ['Class', 'Category', 'Filename']
    else:
        raise ValueError(dataset)

    per_class = n_total // 3
    parts = []
    for fam, g in df.groupby('fam'):
        if len(g) < per_class:
            parts.append(g.sample(n=per_class, random_state=seed, replace=True))  # replace only if a class is short
        else:
            parts.append(g.sample(n=per_class, random_state=seed, replace=False))
    sub = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    y = sub['fam'].values
    X = sub.drop(columns=[c for c in drop_cols if c in sub.columns] + ['fam'])
    X = X.select_dtypes(include=[np.number])  # drop any stray non-numeric cols
    return X, y

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

    classes = sorted(set(ytr)); n_classes = len(classes)
    lda_dim = min(n_classes - 1, n_qubits)
    lda = LinearDiscriminantAnalysis(n_components=lda_dim).fit(Xtr_s, ytr)
    Xtr_lda, Xte_lda = lda.transform(Xtr_s), lda.transform(Xte_s)

    remaining = n_qubits - lda_dim
    if remaining > 0:
        pca = PCA(n_components=remaining, random_state=seed).fit(Xtr_s)
        Xtr_p = np.hstack([Xtr_lda, pca.transform(Xtr_s)])
        Xte_p = np.hstack([Xte_lda, pca.transform(Xte_s)])
    else:
        Xtr_p, Xte_p = Xtr_lda, Xte_lda

    rs = RobustScaler().fit(Xtr_p)
    Xtr_r = np.clip(rs.transform(Xtr_p), -3, 3) / 3 * np.pi
    Xte_r = np.clip(rs.transform(Xte_p), -3, 3) / 3 * np.pi
    class_to_idx = {c: i for i, c in enumerate(classes)}
    ytr_i = np.array([class_to_idx[v] for v in ytr])
    yte_i = np.array([class_to_idx[v] for v in yte])
    return Xtr_r, Xte_r, ytr_i, yte_i, classes, lda_dim

def make_reupload_kernel(n_qubits, L=2):
    dev = qml.device('lightning.qubit', wires=n_qubits)
    @qml.qnode(dev)
    def kcircuit(x1, x2):
        for _ in range(L):
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
            for i in range(n_qubits): qml.CNOT(wires=[i, (i+1) % n_qubits])
        for _ in range(L):
            for i in reversed(range(n_qubits)): qml.CNOT(wires=[i, (i+1) % n_qubits])
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
    K_train_land = np.array([[k(a, b) for b in L_pts] for a in Xtr])
    K_test_land  = np.array([[k(a, b) for b in L_pts] for a in Xte])
    K_MM = np.array([[k(a, b) for b in L_pts] for a in L_pts]) + 1e-6*np.eye(len(L_pts))
    evals, evecs = np.linalg.eigh(K_MM); evals = np.clip(evals, 1e-8, None)
    K_MM_inv_sqrt = evecs @ np.diag(1.0/np.sqrt(evals)) @ evecs.T
    psi_train = K_train_land @ K_MM_inv_sqrt
    psi_test  = K_test_land @ K_MM_inv_sqrt
    clf = SVC(kernel='linear').fit(psi_train, ytr)
    pred = clf.predict(psi_test)
    return {"acc": accuracy_score(yte, pred), "f1_macro": f1_score(yte, pred, average='macro'),
            "time_s": time.time()-t0, "n_train_used": len(Xtr), "n_landmarks": len(L_pts)}

def run_vqc_v2_adam(Xtr, Xte, ytr, yte, n_qubits, classes, L=2, max_train=200, epochs=10, lr=0.1, batch_size=16, seed=42):
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train:
        idx = rng.choice(len(Xtr), max_train, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    n_classes = len(classes); n_out = n_classes
    dev = qml.device('lightning.qubit', wires=n_qubits); n_layers = 2

    @qml.qnode(dev, diff_method='adjoint')
    def circuit(x, weights):
        for _ in range(L):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_out)]

    weights = pnp.array(0.1*np.random.randn(n_layers, n_qubits, 3), requires_grad=True)
    opt = qml.AdamOptimizer(lr)
    y_oh = np.eye(n_classes)[ytr]

    def batch_loss(w, Xb, Yb):
        total = 0.0
        for x, yv in zip(Xb, Yb):
            out = pnp.stack(circuit(x, w))
            p = pnp.exp(out - pnp.max(out)); p = p / pnp.sum(p)
            total = total - pnp.sum(yv * pnp.log(p + 1e-9))
        return total / len(Xb)

    t0 = time.time()
    n = len(Xtr)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            weights = opt.step(lambda w: batch_loss(w, Xtr[idx], y_oh[idx]), weights)
    preds = [np.argmax(np.array(circuit(x, weights))) for x in Xte]
    return {"acc": accuracy_score(yte, preds), "f1_macro": f1_score(yte, preds, average='macro'),
            "time_s": time.time()-t0, "n_train_used": n, "epochs": epochs}

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
    svm = SVC(kernel='rbf').fit(Xtr_s, ytr_s)
    p = svm.predict(Xte_s)
    svm_r = {"acc": accuracy_score(yte_s, p), "f1_macro": f1_score(yte_s, p, average='macro')}
    rf = RandomForestClassifier(n_estimators=200, random_state=seed).fit(Xtr_s, ytr_s)
    p = rf.predict(Xte_s)
    rf_r = {"acc": accuracy_score(yte_s, p), "f1_macro": f1_score(yte_s, p, average='macro')}
    return svm_r, rf_r

def run_one(dataset, n_total, n_qubits, out_path='results_v3.json'):
    X, y = load_3class(dataset, n_total)
    Xtr, Xte, ytr, yte, classes, lda_dim = preprocess_v2(X, y, n_qubits)
    svm_r, rf_r = run_classical(Xtr, Xte, ytr, yte)
    qsvm_r = run_qsvm_nystrom(Xtr, Xte, ytr, yte, n_qubits=n_qubits)
    vqc_r = run_vqc_v2_adam(Xtr, Xte, ytr, yte, n_qubits=n_qubits, classes=classes)
    record = {"dataset": dataset, "n_total": n_total, "qubits": n_qubits, "lda_dim": lda_dim,
              "classes": classes, "classical_svm": svm_r, "classical_rf": rf_r,
              "qsvm_v2": qsvm_r, "vqc_v2_adam": vqc_r}
    data = json.load(open(out_path)) if os.path.exists(out_path) else {"runs": []}
    data["runs"] = [r for r in data["runs"] if not (r["dataset"]==dataset and r["n_total"]==n_total and r["qubits"]==n_qubits)]
    data["runs"].append(record)
    json.dump(data, open(out_path, "w"), indent=2)
    print(f"[{dataset:>8}] n={n_total:<5} q={n_qubits:<3} | clSVM={svm_r['acc']:.3f} clRF={rf_r['acc']:.3f} "
          f"| QSVM={qsvm_r['acc']:.3f} VQC={vqc_r['acc']:.3f}", flush=True)
    return record

if __name__ == '__main__':
    dataset = sys.argv[1]; n_total = int(sys.argv[2]); n_qubits = int(sys.argv[3])
    run_one(dataset, n_total, n_qubits)
