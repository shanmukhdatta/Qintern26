"""
QTagger+ | Three-way comparison: does CTGAN augmentation help the QUANTUM model
specifically, or only the classical one?

  A) quantum-on-augmented   (already have: n=200 exact kernel, 63.3% acc)
  B) quantum-on-original    (this script: same n=200/60, SAME fitted scaler+PCA
                              from the augmented pipeline, applied to the real,
                              family-imbalanced CIC-MalMem-2022 malware rows)
  C) classical-on-augmented (already have: 87%, full augmented train set)

Reusing the augmented-fit scaler/PCA (not re-fitting on original) isolates the
one variable we actually care about: does the TRAINING DATA (augmented vs
original) change quantum-model performance, holding the preprocessing pipeline
fixed? Re-fitting a fresh scaler/PCA on the original data would confound that.
"""
import json, time
import numpy as np
import pandas as pd
import joblib
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ORIG_PATH = "/mnt/user-data/uploads/MalMem2022__2_.csv"
OUT = "/mnt/user-data/outputs"
N_QUBITS = 8
N_TRAIN_SUB, N_TEST_SUB = 200, 60
RANDOM_STATE = 42

# ---- load original, derive Family, malware-only (Benign not in augmented set) ---
df = pd.read_csv(ORIG_PATH)
df = df[df["Class"] == "Malware"].copy()
df["Family"] = df["Category"].str.split("-").str[0]
print("Original malware-only family distribution:", df["Family"].value_counts().to_dict())

with open(f"{OUT}/pipeline_report.json") as f:
    report = json.load(f)
keep_corr = report["features_after_filters"]

X_orig = df[keep_corr]
y_orig = df["Family"]

X_train, X_test, y_train, y_test = train_test_split(
    X_orig, y_orig, test_size=0.20, stratify=y_orig, random_state=RANDOM_STATE)

# reuse the SAME fitted scaler + PCA from the augmented-data pipeline (no re-fit)
scaler = joblib.load(f"{OUT}/scaler.joblib")
pca = joblib.load(f"{OUT}/pca.joblib")
X_train_pca = pca.transform(scaler.transform(X_train))
X_test_pca = pca.transform(scaler.transform(X_test))

le = LabelEncoder().fit(y_train)
y_train_enc, y_test_enc = le.transform(y_train), le.transform(y_test)

_, idx_tr = train_test_split(np.arange(len(X_train_pca)), test_size=N_TRAIN_SUB,
                              stratify=y_train_enc, random_state=RANDOM_STATE)
_, idx_te = train_test_split(np.arange(len(X_test_pca)), test_size=N_TEST_SUB,
                              stratify=y_test_enc, random_state=RANDOM_STATE)
Xtr, ytr = X_train_pca[idx_tr], y_train_enc[idx_tr]
Xte, yte = X_test_pca[idx_te], y_test_enc[idx_te]
print("Subsample train class counts:", dict(zip(*np.unique(ytr, return_counts=True))))
print("Subsample test class counts:", dict(zip(*np.unique(yte, return_counts=True))))

ang_scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtr)
Xtr_ang, Xte_ang = ang_scaler.transform(Xtr), ang_scaler.transform(Xte)

# ---- exact quantum fidelity kernel (n=200 is feasible exactly) ------------
dev = qml.device("lightning.qubit", wires=N_QUBITS)
def feature_map(x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    for i in range(N_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Z")
adj_feature_map = qml.adjoint(feature_map)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    adj_feature_map(x2)
    return qml.probs(wires=range(N_QUBITS))

def fidelity(x1, x2):
    return kernel_circuit(x1, x2)[0]

def build_km(A, B, symmetric=False):
    n, m = len(A), len(B)
    K = np.zeros((n, m))
    for i in range(n):
        j0 = i if symmetric else 0
        for j in range(j0, m):
            v = fidelity(A[i], B[j])
            K[i, j] = v
            if symmetric:
                K[j, i] = v
    return K

t0 = time.time()
K_train = build_km(Xtr_ang, Xtr_ang, symmetric=True)
K_test = build_km(Xte_ang, Xtr_ang, symmetric=False)
kernel_time = time.time() - t0

qsvm = SVC(kernel="precomputed", C=1.0, random_state=RANDOM_STATE)
qsvm.fit(K_train, ytr)
y_pred_q = qsvm.predict(K_test)
q_acc = accuracy_score(yte, y_pred_q)
q_f1 = f1_score(yte, y_pred_q, average="macro")
q_report = classification_report(yte, y_pred_q, target_names=le.classes_, output_dict=True)
q_cm = confusion_matrix(yte, y_pred_q).tolist()

rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
rbf.fit(Xtr, ytr)
c_acc = accuracy_score(yte, rbf.predict(Xte))
c_f1 = f1_score(yte, rbf.predict(Xte), average="macro")

result = {
    "dataset": "original CIC-MalMem-2022, malware-only, family-imbalanced",
    "family_distribution_full": df["Family"].value_counts().to_dict(),
    "n_train": N_TRAIN_SUB, "n_test": N_TEST_SUB, "n_qubits": N_QUBITS,
    "kernel_build_time_sec": round(kernel_time, 1),
    "qsvm_on_original_accuracy": round(q_acc, 4),
    "qsvm_on_original_macro_f1": round(q_f1, 4),
    "qsvm_confusion_matrix": q_cm,
    "qsvm_per_class": {k: v for k, v in q_report.items() if k in le.classes_},
    "classical_rbf_on_original_accuracy": round(c_acc, 4),
    "classical_rbf_on_original_macro_f1": round(c_f1, 4),
    "class_labels": le.classes_.tolist(),
}
with open(f"{OUT}/qsvm_on_original_results.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
