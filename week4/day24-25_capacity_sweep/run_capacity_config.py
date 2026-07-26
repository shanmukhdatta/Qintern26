import sys, json, os, time
import numpy as np
from preprocess import load_and_split, preprocess_pipeline
from quantum_models import run_vqc, VQC

q = int(sys.argv[1])
np.random.seed(42)

# Fixed protocol: n=200 scale, capped train=100, fixed epochs=10, fixed batch=20
# This isolates qubit-count effect from the epoch/compute-budget confound in Day23.
X, y = load_and_split(200)
classes = sorted(set(y))
Xtr, Xte, ytr, yte, evr, nfeat = preprocess_pipeline(X, y, q)

t0 = time.time()
vqc_res = run_vqc(Xtr, Xte, ytr, yte, q, classes, epochs=10, max_train=100)
elapsed = time.time() - t0

# variance-tracking metric: variance of raw circuit output scores across test set
# (proxy for how much of the qubit register's expressive capacity is being used)
n_classes = len(classes)
n_out = min(n_classes, q)
vqc = VQC(q, n_classes)
# re-evaluate output score variance using a fresh untrained circuit at this width (structural capacity proxy)
rng = np.random.default_rng(0)
sample_idx = rng.choice(len(Xte), size=min(30, len(Xte)), replace=False)
scores = vqc.predict_scores(Xte[sample_idx], vqc.weights)
output_variance = float(np.mean(np.var(scores, axis=0)))

entry = {
    "qubits": q, "explained_var_pca": round(float(evr), 4),
    "acc": round(vqc_res["acc"], 4), "f1_macro": round(vqc_res["f1_macro"], 4),
    "time_s": round(elapsed, 1), "n_train_used": vqc_res["n_train_used"],
    "epochs": 10, "output_score_variance": round(output_variance, 5),
}

path = "results_capacity.json"
data = {"sweep": []}
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
data["sweep"].append(entry)
with open(path, "w") as f:
    json.dump(data, f, indent=2)

print("SAVED", entry, flush=True)
