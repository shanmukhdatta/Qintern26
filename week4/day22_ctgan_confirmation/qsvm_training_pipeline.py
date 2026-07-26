import time
import json
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

OUT = "/mnt/user-data/outputs"
N_QUBITS = 8
N_TRAIN_SUB = 250   # kernel matrix is O(n^2) circuit evals -> keep small for free-tier CPU
N_TEST_SUB = 80
SEED = 42
np.random.seed(SEED)

# Data loading / subsampling
def load_stage1_outputs(out_dir: str):
    X_train = np.load(f"{out_dir}/X_train_pca.npy")
    X_test = np.load(f"{out_dir}/X_test_pca.npy")
    y_train = pd.read_csv(f"{out_dir}/y_train.csv").iloc[:, 0].values
    y_test = pd.read_csv(f"{out_dir}/y_test.csv").iloc[:, 0].values
    return X_train, X_test, y_train, y_test


def encode_labels(y_train, y_test):
    le = LabelEncoder().fit(y_train)
    return le, le.transform(y_train), le.transform(y_test)


def stratified_subsample(X_train, y_train_enc, X_test, y_test_enc,
                          n_train_sub, n_test_sub, seed):
    _, X_tr_sub, _, y_tr_sub = train_test_split(
        X_train, y_train_enc, test_size=n_train_sub, stratify=y_train_enc, random_state=seed
    )
    _, X_te_sub, _, y_te_sub = train_test_split(
        X_test, y_test_enc, test_size=n_test_sub, stratify=y_test_enc, random_state=seed
    )
    return X_tr_sub, y_tr_sub, X_te_sub, y_te_sub

# Custom transformer — the one piece sklearn has no built-in for:
# a precomputed quantum fidelity kernel, wrapped as fit/transform
class QuantumFidelityKernel(BaseEstimator, TransformerMixin):
    """
    fit(X)       -> stores X as the reference set, builds the qnode
    transform(X) -> returns Gram matrix K[i,j] = fidelity(X[i], reference[j])

    This is the standard sklearn idiom for a custom/precomputed kernel:
    fit() remembers the training points, transform() computes similarities
    of *any* input against them. Calling transform() on the same data used
    in fit() naturally reproduces the train-train kernel; calling it on new
    data reproduces the test-train kernel. Pipeline.fit_transform() and
    Pipeline.predict() drive this automatically.
    """

    def __init__(self, n_qubits: int, device_name: str = "lightning.qubit"):
        self.n_qubits = n_qubits
        self.device_name = device_name

    def _build_qnode(self):
        dev = qml.device(self.device_name, wires=self.n_qubits)

        def feature_map(x):
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Z")

        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            feature_map(x1)
            qml.adjoint(feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))

        return kernel_circuit

    def _fidelity(self, x1, x2):
        return self._kernel_circuit(x1, x2)[0]  # prob of all-zero state

    def _gram(self, A, B, symmetric: bool) -> np.ndarray:
        n, m = len(A), len(B)
        K = np.zeros((n, m))
        for i in range(n):
            j_start = i if symmetric else 0
            for j in range(j_start, m):
                v = self._fidelity(A[i], B[j])
                K[i, j] = v
                if symmetric:
                    K[j, i] = v
        return K

    def fit(self, X, y=None):
        self.X_fit_ = np.asarray(X)
        self._kernel_circuit = self._build_qnode()
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X)
        symmetric = X.shape == self.X_fit_.shape and np.array_equal(X, self.X_fit_)
        return self._gram(X, self.X_fit_, symmetric=symmetric)

#----------------------------------------------------------
# Pipeline factories

def build_qsvm_pipeline(n_qubits: int, seed: int) -> Pipeline:
    return Pipeline(steps=[
        ("angle_scaler", MinMaxScaler(feature_range=(0, np.pi))),
        ("quantum_kernel", QuantumFidelityKernel(n_qubits=n_qubits)),
        ("svm", SVC(kernel="precomputed", C=1.0, random_state=seed)),
    ])


def build_classical_baseline_pipeline(seed: int) -> Pipeline:
    return Pipeline(steps=[
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)),
    ])
#--------------------------------------------------
# Reporting
def build_result_report(
    n_qubits, n_train_sub, n_test_sub, kernel_time,
    y_te_sub, y_pred_q, q_acc, q_f1, le,
    c_acc, c_f1,
) -> dict:
    q_report = classification_report(
        y_te_sub, y_pred_q, target_names=le.classes_, output_dict=True
    )
    q_cm = confusion_matrix(y_te_sub, y_pred_q).tolist()

    return {
        "n_qubits": n_qubits,
        "train_subsample": n_train_sub,
        "test_subsample": n_test_sub,
        "kernel_build_time_sec": round(kernel_time, 1),
        "qsvm_accuracy": round(q_acc, 4),
        "qsvm_macro_f1": round(q_f1, 4),
        "qsvm_confusion_matrix": q_cm,
        "qsvm_per_class": {k: v for k, v in q_report.items() if k in le.classes_},
        "classical_rbf_svm_accuracy": round(c_acc, 4),
        "classical_rbf_svm_macro_f1": round(c_f1, 4),
        "class_labels": le.classes_.tolist(),
    }



def save_report(out_dir: str, result: dict):
    with open(f"{out_dir}/qsvm_results.json", "w") as f:
        json.dump(result, f, indent=2)

## 
def main():
    X_train, X_test, y_train, y_test = load_stage1_outputs(OUT)
    le, y_train_enc, y_test_enc = encode_labels(y_train, y_test)

    X_tr_sub, y_tr_sub, X_te_sub, y_te_sub = stratified_subsample(
        X_train, y_train_enc, X_test, y_test_enc, N_TRAIN_SUB, N_TEST_SUB, SEED
    )

    # ---- QSVM pipeline: MinMax -> quantum fidelity kernel -> precomputed SVC
    qsvm_pipeline = build_qsvm_pipeline(N_QUBITS, SEED)

    t0 = time.time()
    qsvm_pipeline.fit(X_tr_sub, y_tr_sub)      # scales, builds K_train, fits SVC
    y_pred_q = qsvm_pipeline.predict(X_te_sub)  # scales, builds K_test, predicts
    kernel_time = time.time() - t0

    q_acc = accuracy_score(y_te_sub, y_pred_q)
    q_f1 = f1_score(y_te_sub, y_pred_q, average="macro")

    # ---- classical RBF-SVM baseline, identical subsample, raw (unscaled) features
    rbf_pipeline = build_classical_baseline_pipeline(SEED)
    rbf_pipeline.fit(X_tr_sub, y_tr_sub)
    y_pred_c = rbf_pipeline.predict(X_te_sub)

    c_acc = accuracy_score(y_te_sub, y_pred_c)
    c_f1 = f1_score(y_te_sub, y_pred_c, average="macro")

    result = build_result_report(
        N_QUBITS, N_TRAIN_SUB, N_TEST_SUB, kernel_time,
        y_te_sub, y_pred_q, q_acc, q_f1, le,
        c_acc, c_f1,
    )
    save_report(OUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()