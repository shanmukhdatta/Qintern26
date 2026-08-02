import numpy as np, json, time
from preprocess import load_and_split, preprocess_pipeline
from quantum_models import run_qsvm, run_vqc, VQC
import pennylane as qml

X, y = load_and_split(200)
Xtr, Xte, ytr, yte, evr, nfeat = preprocess_pipeline(X, y, 4)
ytr = np.array(ytr); yte = np.array(yte)
classes = sorted(set(ytr))

out = {'qsvm': [], 'vqc': []}
for seed in [1, 7, 42]:
    np.random.seed(seed)
    r = run_qsvm(Xtr, Xte, ytr, yte, n_qubits=4, max_train=80, max_test=40)
    r['seed'] = seed
    out['qsvm'].append(r)
    print('QSVM seed', seed, r['acc'], r['f1_macro'], flush=True)

for seed in [1, 7, 42]:
    np.random.seed(seed)
    r = run_vqc(Xtr, Xte, ytr, yte, n_qubits=4, classes=classes, epochs=10, max_train=140)
    r['seed'] = seed
    out['vqc'].append(r)
    print('VQC seed', seed, r.get('acc'), r.get('f1_macro'), flush=True)

json.dump(out, open('results_multiseed.json','w'), indent=2)
