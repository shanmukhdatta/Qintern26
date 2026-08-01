import sys, json, os, time
import numpy as np
from preprocess import load_and_split, preprocess_pipeline
from quantum_models import run_qsvm, run_vqc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

n_total = int(sys.argv[1])
n_qubits = int(sys.argv[2])
OUT = 'results_grid.json'

np.random.seed(42)
X, y = load_and_split(n_total)
Xtr, Xte, ytr, yte, evr, nfeat = preprocess_pipeline(X, y, n_qubits)
ytr = np.array(ytr); yte = np.array(yte)
classes = sorted(set(ytr))

record = {'n_total': n_total, 'qubits': n_qubits, 'explained_var': round(float(evr),4),
          'n_features_post_filter': int(nfeat), 'full_train': len(Xtr), 'full_test': len(Xte)}

# same subsample QSVM/VQC will use, for apples-to-apples classical comparison
rng = np.random.default_rng(42)
idx_tr = rng.choice(len(Xtr), min(80, len(Xtr)), replace=False)
idx_te = rng.choice(len(Xte), min(40, len(Xte)), replace=False)
Xtr_s, ytr_s = Xtr[idx_tr], ytr[idx_tr]
Xte_s, yte_s = Xte[idx_te], yte[idx_te]

dc = DummyClassifier(strategy='stratified', random_state=42).fit(Xtr_s, ytr_s)
p = dc.predict(Xte_s)
record['dummy'] = {'acc': round(accuracy_score(yte_s,p),4), 'f1': round(f1_score(yte_s,p,average='macro'),4)}

svm_c = SVC(kernel='rbf').fit(Xtr_s, ytr_s)
p = svm_c.predict(Xte_s)
record['classical_svm'] = {'acc': round(accuracy_score(yte_s,p),4), 'f1': round(f1_score(yte_s,p,average='macro'),4)}

rf_c = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr_s, ytr_s)
p = rf_c.predict(Xte_s)
record['classical_rf'] = {'acc': round(accuracy_score(yte_s,p),4), 'f1': round(f1_score(yte_s,p,average='macro'),4)}

t0=time.time()
np.random.seed(42)
qsvm_r = run_qsvm(Xtr, Xte, ytr, yte, n_qubits=n_qubits, max_train=80, max_test=40)
record['qsvm'] = {'acc': round(qsvm_r['acc'],4), 'f1': round(qsvm_r['f1_macro'],4), 'time_s': round(qsvm_r['time_s'],1)}
print(f"n={n_total} q={n_qubits} QSVM done acc={qsvm_r['acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)

t0=time.time()
np.random.seed(42)
epochs = 8 if n_qubits >= 12 else 10
vqc_r = run_vqc(Xtr, Xte, ytr, yte, n_qubits=n_qubits, classes=classes, epochs=epochs, max_train=140)
record['vqc'] = {'acc': round(vqc_r['acc'],4), 'f1': round(vqc_r['f1_macro'],4), 'time_s': round(vqc_r['time_s'],1), 'epochs': epochs}
print(f"n={n_total} q={n_qubits} VQC done acc={vqc_r['acc']:.3f} ({time.time()-t0:.0f}s)", flush=True)

data = json.load(open(OUT)) if os.path.exists(OUT) else {'runs': []}
data['runs'] = [r for r in data['runs'] if not (r['n_total']==n_total and r['qubits']==n_qubits)]
data['runs'].append(record)
json.dump(data, open(OUT,'w'), indent=2)
print("SAVED", record, flush=True)
