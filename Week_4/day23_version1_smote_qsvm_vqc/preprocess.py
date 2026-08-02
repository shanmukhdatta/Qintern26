import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def load_and_split(n_total, seed=42, path='/mnt/user-data/uploads/MalMem2022_SMOTE.csv'):
    df = pd.read_csv(path)
    # stratified subsample matched scale n_total (balanced across 4 classes)
    per_class = n_total // 4
    parts = []
    for lbl, g in df.groupby('Label'):
        parts.append(g.sample(n=per_class, random_state=seed))
    sub = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    y = sub['Label'].values
    X = sub.drop(columns=['Label'])
    return X, y

def preprocess_pipeline(X, y, n_components, variance_thresh=1e-8, corr_thresh=0.95, seed=42):
    # split first (leakage-safe correlation filtering on train split only)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    # variance filter (fit on train)
    variances = Xtr.var()
    keep_var = variances[variances > variance_thresh].index
    Xtr, Xte = Xtr[keep_var], Xte[keep_var]
    # correlation filter on train only
    corr = Xtr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [c for c in upper.columns if any(upper[c] > corr_thresh)]
    Xtr = Xtr.drop(columns=drop_corr)
    Xte = Xte.drop(columns=drop_corr)
    # log1p (locked decision) - only for non-negative skewed cols; apply safely
    Xtr = np.sign(Xtr) * np.log1p(np.abs(Xtr))
    Xte = np.sign(Xte) * np.log1p(np.abs(Xte))
    # StandardScaler
    ss = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = ss.transform(Xtr), ss.transform(Xte)
    # PCA
    pca = PCA(n_components=n_components, random_state=seed).fit(Xtr_s)
    Xtr_p, Xte_p = pca.transform(Xtr_s), pca.transform(Xte_s)
    evr = pca.explained_variance_ratio_.sum()
    # MinMax to [0, pi] for angle encoding
    mm = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtr_p)
    Xtr_a, Xte_a = mm.transform(Xtr_p), mm.transform(Xte_p)
    return Xtr_a, Xte_a, ytr, yte, evr, Xtr.shape[1]

if __name__ == "__main__":
    for n in [200, 1000]:
        X, y = load_and_split(n)
        for k in [4, 6, 8]:
            Xtr, Xte, ytr, yte, evr, nfeat_after_filter = preprocess_pipeline(X, y, k)
            print(f"n={n} qubits={k} feat_after_filter={nfeat_after_filter} explained_var={evr:.3f} train={Xtr.shape} test={Xte.shape}")
