import sys, json, time, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

DATA_PATHS = {
    'original': '/mnt/user-data/uploads/MalMem2022__3_.csv',
    'ctgan': '/mnt/user-data/uploads/malmem_ctgan__2_.csv',
    'smote': '/mnt/user-data/uploads/MalMem2022_SMOTE__2_.csv',
}

def load_3class(dataset, seed=42):
    path = DATA_PATHS[dataset]
    df = pd.read_csv(path)
    if dataset == 'smote':
        df = df[df['Label'] != 'Benign'].copy(); df['fam'] = df['Label']
        drop_cols = ['Label']
    elif dataset == 'ctgan':
        df['fam'] = df['Family']; drop_cols = ['Family']
    elif dataset == 'original':
        df = df[df['Category'] != 'Benign'].copy()
        df['fam'] = df['Category'].str.split('-').str[0]
        drop_cols = ['Class', 'Category', 'Filename']
    y = df['fam'].values
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ['fam'])
    X = X.select_dtypes(include=[np.number])
    return X, y

def run_classical(dataset, seed=42, n_total=500):
    X, y = load_3class(dataset, seed=seed)
    # Match quantum's sample scale (n_total), NOT full dataset -- full-scale classical
    # trivially wins (LightGBM hit 98.8% on full CTGAN) which makes "quantum should win
    # or match" impossible by construction. Matched scale is the fair comparison point.
    per_class = n_total // len(set(y))
    parts = []
    Xy = X.copy(); Xy['__y__'] = y
    for cls, g in Xy.groupby('__y__'):
        parts.append(g.sample(n=min(per_class, len(g)), random_state=seed))
    sub = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    y_sub = sub['__y__'].values
    X_sub = sub.drop(columns=['__y__'])

    # same variance/correlation filter used by the quantum pipeline, for a clean feature set
    Xtr, Xte, ytr, yte = train_test_split(X_sub, y_sub, test_size=0.3, random_state=seed, stratify=y_sub)
    variances = Xtr.var(); keep = variances[variances > 1e-8].index
    Xtr, Xte = Xtr[keep], Xte[keep]
    corr = Xtr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_c = [c for c in upper.columns if any(upper[c] > 0.95)]
    Xtr, Xte = Xtr.drop(columns=drop_c), Xte.drop(columns=drop_c)

    results = {'dataset': dataset, 'n_total': n_total, 'n_train': len(Xtr), 'n_test': len(Xte),
               'n_features_used': Xtr.shape[1]}

    # LightGBM with light random search
    lgbm_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, -1],
        'learning_rate': [0.05, 0.1, 0.2],
        'num_leaves': [15, 31, 63],
    }
    t0 = time.time()
    lgbm_search = RandomizedSearchCV(LGBMClassifier(random_state=seed, verbose=-1), lgbm_grid,
                                      n_iter=8, cv=3, random_state=seed, n_jobs=-1, scoring='f1_macro')
    lgbm_search.fit(Xtr, ytr)
    best_lgbm = lgbm_search.best_estimator_
    pred = best_lgbm.predict(Xte)
    results['lightgbm'] = {
        'best_params': lgbm_search.best_params_,
        'acc': accuracy_score(yte, pred), 'f1_macro': f1_score(yte, pred, average='macro'),
        'search_time_s': round(time.time()-t0, 1)
    }
    print(f"[{dataset}] LightGBM acc={results['lightgbm']['acc']:.4f} f1={results['lightgbm']['f1_macro']:.4f} "
          f"best_params={lgbm_search.best_params_}", flush=True)

    # Random Forest with light random search
    rf_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2'],
    }
    t0 = time.time()
    rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=seed), rf_grid,
                                    n_iter=8, cv=3, random_state=seed, n_jobs=-1, scoring='f1_macro')
    rf_search.fit(Xtr, ytr)
    best_rf = rf_search.best_estimator_
    pred = best_rf.predict(Xte)
    results['random_forest'] = {
        'best_params': rf_search.best_params_,
        'acc': accuracy_score(yte, pred), 'f1_macro': f1_score(yte, pred, average='macro'),
        'search_time_s': round(time.time()-t0, 1)
    }
    print(f"[{dataset}] RandomForest acc={results['random_forest']['acc']:.4f} f1={results['random_forest']['f1_macro']:.4f} "
          f"best_params={rf_search.best_params_}", flush=True)

    os.makedirs('results_classical', exist_ok=True)
    with open(f'results_classical/{dataset}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return results

if __name__ == '__main__':
    run_classical(sys.argv[1])
