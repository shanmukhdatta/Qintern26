import numpy as np, pennylane as qml, time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

np.random.seed(42)

def make_kernel(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)
    def ring_cnots(wires):
        for i in range(len(wires)):
            qml.CNOT(wires=[wires[i], wires[(i+1) % len(wires)]])

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
        ring_cnots(list(range(n_qubits)))
        qml.adjoint(ring_cnots)(list(range(n_qubits)))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
        return qml.probs(wires=range(n_qubits))
    def kernel(x1, x2):
        return kernel_circuit(x1, x2)[0]
    return kernel

def run_qsvm(Xtr, Xte, ytr, yte, n_qubits, max_train=100, max_test=50):
    if max_train and len(Xtr) > max_train:
        idx = np.random.choice(len(Xtr), max_train, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if max_test and len(Xte) > max_test:
        idx = np.random.choice(len(Xte), max_test, replace=False)
        Xte, yte = Xte[idx], yte[idx]
    k = make_kernel(n_qubits)
    t0=time.time()
    Ktr = np.array([[k(a,b) for b in Xtr] for a in Xtr])
    Kte = np.array([[k(a,b) for b in Xtr] for a in Xte])
    clf = SVC(kernel='precomputed').fit(Ktr, ytr)
    pred = clf.predict(Kte)
    acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, average='macro')
    return {"qubits": n_qubits, "acc": acc, "f1_macro": f1, "time_s": time.time()-t0, "n_train_used": len(Xtr)}

class VQC:
    def __init__(self, n_qubits, n_classes, n_layers=2, seed=42):
        self.n_qubits=n_qubits; self.n_classes=n_classes; self.n_layers=n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.1, size=(n_layers, n_qubits, 3))
        self._build()

    def _build(self):
        n_qubits=self.n_qubits; n_out=min(self.n_classes, n_qubits)
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_out)]
        self.circuit = circuit

    def predict_scores(self, X, weights):
        return np.array([self.circuit(x, weights) for x in X])

    def fit(self, Xtr, ytr_onehot, epochs=25, lr=0.15, batch_size=32):
        opt = qml.AdamOptimizer(lr)
        w = qml.numpy.array(self.weights, requires_grad=True)
        n = len(Xtr)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                bidx = idx[start:start+batch_size]
                xb, yb = Xtr[bidx], ytr_onehot[bidx]
                def cost(w_):
                    scores = qml.numpy.stack([self.circuit(x, w_) for x in xb])
                    probs = (scores + 1) / 2
                    probs = probs / qml.numpy.sum(probs, axis=1, keepdims=True)
                    return -qml.numpy.mean(qml.numpy.sum(yb * qml.numpy.log(probs + 1e-8), axis=1))
                w = opt.step(cost, w)
        self.weights = w
        return self

def run_vqc(Xtr, Xte, ytr, yte, n_qubits, classes, epochs=25, max_train=None):
    if max_train and len(Xtr) > max_train:
        idx = np.random.choice(len(Xtr), max_train, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    n_classes = len(classes)
    y2i = {c:i for i,c in enumerate(classes)}
    ytr_idx = np.array([y2i[v] for v in ytr])
    yte_idx = np.array([y2i[v] for v in yte])
    ytr_oh = np.eye(min(n_classes,n_qubits))[ytr_idx % min(n_classes,n_qubits)]
    vqc = VQC(n_qubits, n_classes)
    t0=time.time()
    vqc.fit(Xtr, ytr_oh, epochs=epochs)
    scores_te = vqc.predict_scores(Xte, vqc.weights)
    pred = np.argmax(scores_te, axis=1)
    acc = accuracy_score(yte_idx % min(n_classes,n_qubits), pred)
    f1 = f1_score(yte_idx % min(n_classes,n_qubits), pred, average='macro')
    return {"qubits": n_qubits, "acc": acc, "f1_macro": f1, "time_s": time.time()-t0, "n_train_used": len(Xtr), "epochs": epochs}
