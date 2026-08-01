import numpy as np, time, json
from pipeline_v2 import load_and_split, preprocess_v2
import pennylane as qml

n_total, n_qubits, L = 500, 12, 2
X, y = load_and_split(n_total)
Xtr, Xte, ytr, yte, classes, lda_dim = preprocess_v2(X, y, n_qubits)
class_to_idx = {c:i for i,c in enumerate(classes)}
ytr = np.array([class_to_idx[v] for v in ytr]); yte = np.array([class_to_idx[v] for v in yte])
n_classes = len(classes); n_out = min(n_classes, n_qubits)

rng = np.random.default_rng(42)
if len(Xtr) > 200:
    idx = rng.choice(len(Xtr), 200, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]

dev = qml.device('lightning.qubit', wires=n_qubits)
n_layers = 2

@qml.qnode(dev, diff_method='adjoint')
def circuit(x, weights):
    for _ in range(L):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_out)]

from pennylane import numpy as pnp
weights = pnp.array(0.1*np.random.randn(n_layers, n_qubits, 3), requires_grad=True)
opt = qml.AdamOptimizer(0.1)
y_oh = np.eye(n_classes)[ytr]

def batch_loss(w, Xb, Yb):
    total = 0.0
    for x, yv in zip(Xb, Yb):
        out = pnp.stack(circuit(x, w))
        p = pnp.exp(out - pnp.max(out)); p = p/pnp.sum(p)
        total = total - pnp.sum(yv * pnp.log(p+1e-9))
    return total/len(Xb)

t0=time.time()
n = len(Xtr); bs=16; epochs=10
for ep in range(epochs):
    perm = np.random.permutation(n)
    for s in range(0,n,bs):
        idx = perm[s:s+bs]
        weights = opt.step(lambda w: batch_loss(w, Xtr[idx], y_oh[idx]), weights)
print(f"trained in {time.time()-t0:.0f}s", flush=True)

preds=[]
for x in Xte:
    out = np.array(circuit(x, weights))
    preds.append(np.argmax(out))
preds=np.array(preds)
from sklearn.metrics import accuracy_score, f1_score
acc=accuracy_score(yte,preds); f1=f1_score(yte,preds,average='macro')
print(f"VQC reupload+Adam n=500 q=12: acc={acc:.3f} f1={f1:.3f}")
json.dump({'acc':acc,'f1':f1}, open('vqc_reupload_adam_result.json','w'))
