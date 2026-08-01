import sys, json, time, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from scipy.optimize import minimize
import pennylane as qml
from pennylane import numpy as pnp

DATA_PATHS = {
    'original': '/mnt/user-data/uploads/MalMem2022__3_.csv',
    'ctgan': '/mnt/user-data/uploads/malmem_ctgan__2_.csv',
    'smote': '/mnt/user-data/uploads/MalMem2022_SMOTE__2_.csv',
}

def load_3class(dataset, n_total, seed=42):
    df = pd.read_csv(DATA_PATHS[dataset])
    if dataset == 'smote':
        df = df[df['Label'] != 'Benign'].copy(); df['fam'] = df['Label']; drop_cols = ['Label']
    elif dataset == 'ctgan':
        df['fam'] = df['Family']; drop_cols = ['Family']
    elif dataset == 'original':
        df = df[df['Category'] != 'Benign'].copy()
        df['fam'] = df['Category'].str.split('-').str[0]; drop_cols = ['Class', 'Category', 'Filename']
    per_class = n_total // 3
    parts = [g.sample(n=per_class, random_state=seed, replace=len(g) < per_class) for _, g in df.groupby('fam')]
    sub = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    y = sub['fam'].values
    X = sub.drop(columns=[c for c in drop_cols if c in sub.columns] + ['fam']).select_dtypes(include=[np.number])
    return X, y

def base_filter(X, y, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    variances = Xtr.var(); keep = variances[variances > 1e-8].index
    Xtr, Xte = Xtr[keep], Xte[keep]
    corr = Xtr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_c = [c for c in upper.columns if any(upper[c] > 0.95)]
    Xtr, Xte = Xtr.drop(columns=drop_c), Xte.drop(columns=drop_c)
    Xtr = np.sign(Xtr) * np.log1p(np.abs(Xtr)); Xte = np.sign(Xte) * np.log1p(np.abs(Xte))
    classes = sorted(set(ytr)); c2i = {c: i for i, c in enumerate(classes)}
    ytr_i = np.array([c2i[v] for v in ytr]); yte_i = np.array([c2i[v] for v in yte])
    return Xtr, Xte, ytr_i, yte_i, classes

def classical_baseline(Xtr, Xte, ytr, yte, seed=42):
    svm = SVC(kernel='rbf').fit(Xtr, ytr); p = svm.predict(Xte)
    svm_r = {'acc': accuracy_score(yte, p), 'f1_macro': f1_score(yte, p, average='macro')}
    rf = RandomForestClassifier(n_estimators=200, random_state=seed).fit(Xtr, ytr); p = rf.predict(Xte)
    rf_r = {'acc': accuracy_score(yte, p), 'f1_macro': f1_score(yte, p, average='macro')}
    return svm_r, rf_r

# ============== PROJECTIONS ==============
def project_pca(Xtr_s, Xte_s, n_qubits, seed=42):
    pca = PCA(n_components=n_qubits, random_state=seed).fit(Xtr_s)
    return pca.transform(Xtr_s), pca.transform(Xte_s)

def project_hybrid(Xtr_s, Xte_s, ytr_i, n_qubits, seed=42):
    lda_dim = min(len(set(ytr_i)) - 1, n_qubits)
    lda = LinearDiscriminantAnalysis(n_components=lda_dim).fit(Xtr_s, ytr_i)
    Xtr_lda, Xte_lda = lda.transform(Xtr_s), lda.transform(Xte_s)
    remaining = n_qubits - lda_dim
    if remaining > 0:
        pca = PCA(n_components=remaining, random_state=seed).fit(Xtr_s)
        return np.hstack([Xtr_lda, pca.transform(Xtr_s)]), np.hstack([Xte_lda, pca.transform(Xte_s)])
    return Xtr_lda, Xte_lda

def project_xgboost(Xtr_s, Xte_s, ytr_i, n_qubits, seed=42):
    clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=seed, eval_metric='mlogloss')
    clf.fit(Xtr_s, ytr_i)
    top_idx = np.argsort(clf.feature_importances_)[::-1][:n_qubits]
    return Xtr_s[:, top_idx], Xte_s[:, top_idx]

# ============== QUANTUM KERNELS/CIRCUITS ==============
def make_v1_kernel(n_qubits):
    dev = qml.device('lightning.qubit', wires=n_qubits)
    @qml.qnode(dev)
    def kc(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
        return qml.probs(wires=range(n_qubits))
    return lambda a, b: kc(a, b)[0]

def make_reupload_kernel(n_qubits, L=2):
    dev = qml.device('lightning.qubit', wires=n_qubits)
    @qml.qnode(dev)
    def kc(x1, x2):
        for _ in range(L):
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
            for i in range(n_qubits): qml.CNOT(wires=[i, (i+1) % n_qubits])
        for _ in range(L):
            for i in reversed(range(n_qubits)): qml.CNOT(wires=[i, (i+1) % n_qubits])
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
        return qml.probs(wires=range(n_qubits))
    return lambda a, b: kc(a, b)[0]

def make_local_trainable_circuit(n_qubits, L=2):
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
    def local_kernel(x1, x2, w):
        probs = probs_circuit(x1, x2, w)
        n_states = len(probs)
        vals = []
        for (a, b) in pairs:
            mask = np.array([(((idx >> (n_qubits-1-a)) & 1)==0) and (((idx >> (n_qubits-1-b)) & 1)==0)
                              for idx in range(n_states)])
            vals.append(probs[mask].sum())
        return sum(vals) / len(vals)
    return local_kernel

def kta_train(local_kernel, X_sub, y_sub, n_qubits, iters=8, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X_sub)
    Y = np.array([[1.0 if y_sub[i]==y_sub[j] else -1.0 for j in range(n)] for i in range(n)])
    def loss(w):
        K = np.array([[local_kernel(X_sub[i], X_sub[j], w) for j in range(n)] for i in range(n)])
        return -(np.sum(K*Y) / (np.sqrt(np.sum(K*K)*np.sum(Y*Y)) + 1e-9))
    w0 = 0.05 * rng.standard_normal(n_qubits)
    res = minimize(loss, w0, method='COBYLA', options={'maxiter': max(iters, n_qubits+2), 'rhobeg': 0.3})
    return res.x

# ============== QSVM RUNNERS ==============
def run_qsvm_simple(kernel_fn, Xtr, Xte, ytr, yte, max_train=80, max_test=40, C=1.0, seed=42):
    """v1 style: no landmarks, full pairwise kernel, small cap."""
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train: idx = rng.choice(len(Xtr), max_train, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > max_test: idx = rng.choice(len(Xte), max_test, replace=False); Xte, yte = Xte[idx], yte[idx]
    t0 = time.time()
    Ktr = np.array([[kernel_fn(a, b) for b in Xtr] for a in Xtr])
    Kte = np.array([[kernel_fn(a, b) for b in Xtr] for a in Xte])
    clf = SVC(kernel='precomputed', C=C).fit(Ktr, ytr)
    pred = clf.predict(Kte)
    return {'acc': accuracy_score(yte, pred), 'f1_macro': f1_score(yte, pred, average='macro'),
            'time_s': round(time.time()-t0,1), 'n_train_used': len(Xtr)}

def run_qsvm_nystrom(kernel_fn, Xtr, Xte, ytr, yte, n_landmarks=40, max_train=300, max_test=40, C=1.0, seed=42):
    """v2/v3 style: Nystrom landmarks."""
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train: idx = rng.choice(len(Xtr), max_train, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > max_test: idx = rng.choice(len(Xte), max_test, replace=False); Xte, yte = Xte[idx], yte[idx]
    landmark_idx = rng.choice(len(Xtr), min(n_landmarks, len(Xtr)), replace=False)
    L_pts = Xtr[landmark_idx]
    t0 = time.time()
    K_train_land = np.array([[kernel_fn(a, b) for b in L_pts] for a in Xtr])
    K_test_land  = np.array([[kernel_fn(a, b) for b in L_pts] for a in Xte])
    K_MM = np.array([[kernel_fn(a, b) for b in L_pts] for a in L_pts]) + 1e-6*np.eye(len(L_pts))
    evals, evecs = np.linalg.eigh(K_MM); evals = np.clip(evals, 1e-8, None)
    K_MM_inv_sqrt = evecs @ np.diag(1.0/np.sqrt(evals)) @ evecs.T
    psi_train = K_train_land @ K_MM_inv_sqrt; psi_test = K_test_land @ K_MM_inv_sqrt
    clf = SVC(kernel='linear', C=C).fit(psi_train, ytr)
    pred = clf.predict(psi_test)
    return {'acc': accuracy_score(yte, pred), 'f1_macro': f1_score(yte, pred, average='macro'),
            'time_s': round(time.time()-t0,1), 'n_train_used': len(Xtr)}

def run_qsvm_local_aligned(Xtr, Xte, ytr, yte, n_qubits, n_landmarks=40, max_train=300, max_test=40,
                            align_subset=10, align_iters=8, C=1.0, seed=42):
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train: idx = rng.choice(len(Xtr), max_train, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    if len(Xte) > max_test: idx = rng.choice(len(Xte), max_test, replace=False); Xte, yte = Xte[idx], yte[idx]
    local_kernel = make_local_trainable_circuit(n_qubits, L=2)
    align_idx = rng.choice(len(Xtr), min(align_subset, len(Xtr)), replace=False)
    w, align_time = None, 0
    t0 = time.time()
    w = kta_train(local_kernel, Xtr[align_idx], ytr[align_idx], n_qubits, iters=align_iters, seed=seed)
    align_time = time.time() - t0
    landmark_idx = rng.choice(len(Xtr), min(n_landmarks, len(Xtr)), replace=False)
    L_pts = Xtr[landmark_idx]
    t0 = time.time()
    K_train_land = np.array([[local_kernel(a, b, w) for b in L_pts] for a in Xtr])
    K_test_land  = np.array([[local_kernel(a, b, w) for b in L_pts] for a in Xte])
    K_MM = np.array([[local_kernel(a, b, w) for b in L_pts] for a in L_pts]) + 1e-6*np.eye(len(L_pts))
    evals, evecs = np.linalg.eigh(K_MM); evals = np.clip(evals, 1e-8, None)
    K_MM_inv_sqrt = evecs @ np.diag(1.0/np.sqrt(evals)) @ evecs.T
    psi_train = K_train_land @ K_MM_inv_sqrt; psi_test = K_test_land @ K_MM_inv_sqrt
    clf = SVC(kernel='linear', C=C).fit(psi_train, ytr)
    pred = clf.predict(psi_test)
    return {'acc': accuracy_score(yte, pred), 'f1_macro': f1_score(yte, pred, average='macro'),
            'align_time_s': round(align_time,1), 'kernel_time_s': round(time.time()-t0,1),
            'n_train_used': len(Xtr)}

# ============== VQC RUNNERS ==============
def run_vqc(Xtr, Xte, ytr, yte, n_qubits, classes, L=1, epochs=15, max_train=140, batch_size=16,
            lr=0.1, init_scale=0.1, seed=42):
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train: idx = rng.choice(len(Xtr), max_train, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    n_classes = len(classes)
    dev = qml.device('lightning.qubit', wires=n_qubits); n_layers = 2

    @qml.qnode(dev, diff_method='adjoint')
    def circuit(x, weights):
        for _ in range(L):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
            if L > 1:
                for i in range(n_qubits): qml.CNOT(wires=[i, (i+1) % n_qubits])
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_classes)]

    weights = pnp.array(init_scale * np.random.randn(n_layers, n_qubits, 3), requires_grad=True)
    opt = qml.AdamOptimizer(lr)
    y_oh = np.eye(n_classes)[ytr]

    def batch_loss(w, Xb, Yb):
        total = 0.0
        for x, yv in zip(Xb, Yb):
            out = pnp.stack(circuit(x, w)); p = pnp.exp(out-pnp.max(out)); p = p/pnp.sum(p)
            total = total - pnp.sum(yv * pnp.log(p+1e-9))
        return total / len(Xb)

    t0 = time.time(); n = len(Xtr)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            weights = opt.step(lambda w: batch_loss(w, Xtr[idx], y_oh[idx]), weights)
    preds = [np.argmax(np.array(circuit(x, weights))) for x in Xte]
    return {'acc': accuracy_score(yte, preds), 'f1_macro': f1_score(yte, preds, average='macro'),
            'time_s': round(time.time()-t0,1), 'epochs': epochs, 'n_train_used': n}

# ============== VERSION DISPATCH ==============
def run_qsvm_version(version, dataset, n_qubits, n_total=500, seed=7):
    X, y = load_3class(dataset, n_total, seed=seed)
    Xtr, Xte, ytr, yte, classes = base_filter(X, y, seed=seed)
    ss = StandardScaler().fit(Xtr); Xtr_s, Xte_s = ss.transform(Xtr), ss.transform(Xte)

    if version == 'v1':
        Xtr_p, Xte_p = project_pca(Xtr_s, Xte_s, n_qubits, seed=seed)
        mm = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtr_p)
        Xtr_r, Xte_r = mm.transform(Xtr_p), mm.transform(Xte_p)
        kernel = make_v1_kernel(n_qubits)
        q_result = run_qsvm_simple(kernel, Xtr_r, Xte_r, ytr, yte, seed=seed)
    elif version == 'v2':
        Xtr_p, Xte_p = project_hybrid(Xtr_s, Xte_s, ytr, n_qubits, seed=seed)
        rs = RobustScaler().fit(Xtr_p)
        Xtr_r = np.clip(rs.transform(Xtr_p), -3, 3) / 3 * np.pi
        Xte_r = np.clip(rs.transform(Xte_p), -3, 3) / 3 * np.pi
        kernel = make_reupload_kernel(n_qubits, L=2)
        q_result = run_qsvm_nystrom(kernel, Xtr_r, Xte_r, ytr, yte, seed=seed)
    elif version == 'v3':
        Xtr_p, Xte_p = project_xgboost(Xtr_s, Xte_s, ytr, n_qubits, seed=seed)
        rs = RobustScaler().fit(Xtr_p)
        Xtr_r = np.clip(rs.transform(Xtr_p), -3, 3) / 3 * np.pi
        Xte_r = np.clip(rs.transform(Xte_p), -3, 3) / 3 * np.pi
        q_result = run_qsvm_local_aligned(Xtr_r, Xte_r, ytr, yte, n_qubits=n_qubits, seed=seed)

    svm_r, rf_r = classical_baseline(Xtr_r, Xte_r, ytr, yte, seed=seed)
    return {'model': 'qsvm', 'version': version, 'dataset': dataset, 'n_qubits': n_qubits,
            'n_total': n_total, 'seed': seed, 'result': q_result,
            'classical_svm_same_features': svm_r, 'classical_rf_same_features': rf_r}

def run_vqc_version(version, dataset, n_qubits, n_total=500, seed=7):
    X, y = load_3class(dataset, n_total, seed=seed)
    Xtr, Xte, ytr, yte, classes = base_filter(X, y, seed=seed)
    ss = StandardScaler().fit(Xtr); Xtr_s, Xte_s = ss.transform(Xtr), ss.transform(Xte)

    if version == 'v1':
        Xtr_p, Xte_p = project_pca(Xtr_s, Xte_s, n_qubits, seed=seed)
        mm = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtr_p)
        Xtr_r, Xte_r = mm.transform(Xtr_p), mm.transform(Xte_p)
        q_result = run_vqc(Xtr_r, Xte_r, ytr, yte, n_qubits, classes, L=1, epochs=15, init_scale=0.1, seed=seed)
    elif version == 'v2':
        Xtr_p, Xte_p = project_hybrid(Xtr_s, Xte_s, ytr, n_qubits, seed=seed)
        rs = RobustScaler().fit(Xtr_p)
        Xtr_r = np.clip(rs.transform(Xtr_p), -3, 3) / 3 * np.pi
        Xte_r = np.clip(rs.transform(Xte_p), -3, 3) / 3 * np.pi
        q_result = run_vqc(Xtr_r, Xte_r, ytr, yte, n_qubits, classes, L=2, epochs=15, init_scale=0.1, seed=seed)
    elif version == 'v3':
        Xtr_p, Xte_p = project_xgboost(Xtr_s, Xte_s, ytr, n_qubits, seed=seed)
        rs = RobustScaler().fit(Xtr_p)
        Xtr_r = np.clip(rs.transform(Xtr_p), -3, 3) / 3 * np.pi
        Xte_r = np.clip(rs.transform(Xte_p), -3, 3) / 3 * np.pi
        # v3: XGBoost selection + re-upload + informed (near-identity) init
        q_result = run_vqc(Xtr_r, Xte_r, ytr, yte, n_qubits, classes, L=2, epochs=15, init_scale=0.02, seed=seed)

    svm_r, rf_r = classical_baseline(Xtr_r, Xte_r, ytr, yte, seed=seed)
    return {'model': 'vqc', 'version': version, 'dataset': dataset, 'n_qubits': n_qubits,
            'n_total': n_total, 'seed': seed, 'result': q_result,
            'classical_svm_same_features': svm_r, 'classical_rf_same_features': rf_r}

if __name__ == '__main__':
    model, version, dataset, n_qubits = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 7
    n_total = int(sys.argv[6]) if len(sys.argv) > 6 else 500

    if model == 'qsvm':
        rec = run_qsvm_version(version, dataset, n_qubits, n_total=n_total, seed=seed)
    else:
        rec = run_vqc_version(version, dataset, n_qubits, n_total=n_total, seed=seed)

    print(f"[{model}-{version}] {dataset} q={n_qubits} seed={seed} | "
          f"acc={rec['result']['acc']:.4f} f1={rec['result']['f1_macro']:.4f} | "
          f"clSVM={rec['classical_svm_same_features']['acc']:.4f} clRF={rec['classical_rf_same_features']['acc']:.4f}",
          flush=True)

    os.makedirs('results_quantum', exist_ok=True)
    outpath = f"results_quantum/{model}_{version}_{dataset}.json"
    data = {'runs': []}
    if os.path.exists(outpath):
        data = json.load(open(outpath))
    data['runs'] = [r for r in data['runs'] if not (r['n_qubits']==n_qubits and r['seed']==seed)]
    data['runs'].append(rec)
    json.dump(data, open(outpath, 'w'), indent=2, default=str)
