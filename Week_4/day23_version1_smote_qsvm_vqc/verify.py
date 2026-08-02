import numpy as np, time, json
from preprocess import load_and_split, preprocess_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

np.random.seed(42)

# Best config from Day23: n=200, q=4
X, y = load_and_split(200)
Xtr, Xte, ytr, yte, evr, nfeat = preprocess_pipeline(X, y, 4)
print("Full train/test sizes:", Xtr.shape, Xte.shape)

# What QSVM/VQC actually saw (subsampled to 80/40)
rng = np.random.default_rng(42)
idx_tr = rng.choice(len(Xtr), min(80,len(Xtr)), replace=False)
idx_te = rng.choice(len(Xte), min(40,len(Xte)), replace=False)
Xtr_s, ytr_s = Xtr[idx_tr], np.array(ytr)[idx_tr]
Xte_s, yte_s = Xte[idx_te], np.array(yte)[idx_te]

results = {}

# Dummy baseline (stratified random)
dc = DummyClassifier(strategy='stratified', random_state=42).fit(Xtr_s, ytr_s)
pred = dc.predict(Xte_s)
results['dummy_stratified'] = {'acc': accuracy_score(yte_s,pred), 'f1': f1_score(yte_s,pred,average='macro')}

# Classical RBF-SVM on SAME 80/40 subsample, SAME 4 PCA features
csvm = SVC(kernel='rbf').fit(Xtr_s, ytr_s)
pred = csvm.predict(Xte_s)
results['classical_svm_rbf_same80_40'] = {'acc': accuracy_score(yte_s,pred), 'f1': f1_score(yte_s,pred,average='macro')}

# Classical RF on SAME 80/40 subsample
rf_small = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr_s, ytr_s)
pred = rf_small.predict(Xte_s)
results['classical_rf_same80_40'] = {'acc': accuracy_score(yte_s,pred), 'f1': f1_score(yte_s,pred,average='macro')}

# Classical RF on FULL train/test (no subsample cap) - what a real classical pipeline would use
rf_full = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
pred = rf_full.predict(Xte)
results['classical_rf_full_data'] = {'acc': accuracy_score(yte,pred), 'f1': f1_score(yte,pred,average='macro')}

for k,v in results.items():
    print(f"{k:35s} acc={v['acc']:.3f} f1={v['f1']:.3f}")

json.dump(results, open('results_baseline_check.json','w'), indent=2)
