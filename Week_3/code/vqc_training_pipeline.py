import time
import json
import numpy as _np
import pandas as pd
import pennylane as qml
from pennylane import numpy as np          # autograd-wrapped numpy for trainable weights
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

OUT = "/mnt/user-data/outputs"
N_QUBITS = 8
REPS = 1                       # StronglyEntanglingLayers depth -- swap for Ge's config when ready
N_CLASSES = 3                  # Ransomware / Spyware / Trojan
DEVICE_NAME = "lightning.qubit"

N_TRAIN_SUB, N_TEST_SUB = 1000, 200   # ramp target; sed to (200, 60) etc. for smoke tests
BATCH_SIZE = 32
N_STEPS = 400
LOG_EVERY = 50
SEED = 42

np.random.seed(SEED)
_np.random.seed(SEED)

PROTOCOL = f"Anh ramp n=200->n=1000, VQC via SPSA (Ge config PENDING -- placeholder arch)"
ARCH_NOTE = f"StronglyEntanglingLayers placeholder, reps={REPS}, {N_QUBITS}q -- swap for Ge's config"

# ----------------------------------------------------------
# Data loading / subsampling (same conventions as the QSVM pipeline)
def load_stage1_outputs(out_dir: str):
    X_train = _np.load(f"{out_dir}/X_train_pca.npy")
    X_test = _np.load(f"{out_dir}/X_test_pca.npy")
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


def scale_to_angles(X_tr, X_te):
    # same MinMaxScaler(0, pi) convention as the QSVM angle_scaler step
    scaler = MinMaxScaler(feature_range=(0, _np.pi))
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    return X_tr_scaled, X_te_scaled


# ----------------------------------------------------------
# VQC: AngleEmbedding feature map + StronglyEntanglingLayers ansatz,
# class scores read off as PauliZ expvals on the first N_CLASSES wires
def build_qnode(n_qubits: int, device_name: str):
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(weights, x):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_CLASSES)]

    return circuit


def init_weights(reps: int, n_qubits: int, seed: int):
    shape = qml.StronglyEntanglingLayers.shape(n_layers=reps, n_wires=n_qubits)
    rng = _np.random.default_rng(seed)
    return np.array(rng.uniform(0, 2 * _np.pi, size=shape), requires_grad=True)


def softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def sample_loss(weights, circuit, x, y_int):
    scores = np.stack(circuit(weights, x))          # raw expvals in [-1, 1] as logits
    probs = softmax(scores)
    return -np.log(probs[y_int] + 1e-9)


def batch_cost(weights, circuit, X_batch, y_batch):
    losses = [sample_loss(weights, circuit, X_batch[i], y_batch[i]) for i in range(len(X_batch))]
    return np.mean(np.stack(losses))


def predict(weights, circuit, X):
    preds = []
    for x in X:
        scores = np.stack(circuit(weights, x))
        preds.append(int(np.argmax(scores)))
    return _np.array(preds)


# ----------------------------------------------------------
# Training loop (SPSA, mini-batched, matches the "step k/N loss=... t=Ys" log format)
def train_vqc(circuit, X_tr, y_tr, n_steps: int, batch_size: int, reps: int,
              n_qubits: int, seed: int, log_every: int = LOG_EVERY):
    weights = init_weights(reps, n_qubits, seed)
    opt = qml.SPSAOptimizer(maxiter=n_steps)
    rng = _np.random.default_rng(seed)

    t0 = time.time()
    final_loss = None
    for step in range(n_steps):
        idx = rng.choice(len(X_tr), size=min(batch_size, len(X_tr)), replace=False)
        X_batch, y_batch = X_tr[idx], y_tr[idx]

        cost_fn = lambda w: batch_cost(w, circuit, X_batch, y_batch)
        weights, loss_val = opt.step_and_cost(cost_fn, weights)
        final_loss = float(loss_val)

        if step % log_every == 0:
            print(f"step {step}/{n_steps} loss={final_loss:.4f} t={int(time.time() - t0)}s")

    train_time = time.time() - t0
    print(f"training done at {int(train_time)}s")
    return weights, final_loss, train_time


# ----------------------------------------------------------
# Reporting -- same schema as the pasted VQC results / QSVM report style
def build_result_report(
    protocol, arch_note, n_train_sub, n_test_sub, n_qubits, reps,
    batch_size, n_steps, train_time, final_train_loss, train_accuracy,
    y_te_sub, y_pred_v, v_acc, v_f1, le,
) -> dict:
    v_report = classification_report(
        y_te_sub, y_pred_v, target_names=le.classes_, output_dict=True
    )
    v_cm = confusion_matrix(y_te_sub, y_pred_v).tolist()

    return {
        "protocol": protocol,
        "arch_note": arch_note,
        "n_train": n_train_sub,
        "n_test": n_test_sub,
        "n_qubits": n_qubits,
        "reps": reps,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "train_time_sec": round(train_time, 1),
        "final_train_loss": final_train_loss,
        "train_accuracy": round(train_accuracy, 4),
        "vqc_test_accuracy": round(v_acc, 4),
        "vqc_test_macro_f1": round(v_f1, 4),
        "vqc_confusion_matrix": v_cm,
        "vqc_per_class": {k: v for k, v in v_report.items() if k in le.classes_},
        "class_labels": le.classes_.tolist(),
    }


def save_report(out_dir: str, result: dict, filename: str = "vqc_n1000_spsa_results.json"):
    with open(f"{out_dir}/{filename}", "w") as f:
        json.dump(result, f, indent=2)


# ----------------------------------------------------------
def main():
    X_train, X_test, y_train, y_test = load_stage1_outputs(OUT)
    le, y_train_enc, y_test_enc = encode_labels(y_train, y_test)

    X_tr_sub, y_tr_sub, X_te_sub, y_te_sub = stratified_subsample(
        X_train, y_train_enc, X_test, y_test_enc, N_TRAIN_SUB, N_TEST_SUB, SEED
    )

    # same angle-scaling convention as the QSVM pipeline's angle_scaler step
    X_tr_scaled, X_te_scaled = scale_to_angles(X_tr_sub, X_te_sub)

    circuit = build_qnode(N_QUBITS, DEVICE_NAME)

    weights, final_train_loss, train_time = train_vqc(
        circuit, X_tr_scaled, y_tr_sub,
        n_steps=N_STEPS, batch_size=BATCH_SIZE, reps=REPS,
        n_qubits=N_QUBITS, seed=SEED,
    )

    y_pred_train = predict(weights, circuit, X_tr_scaled)
    train_accuracy = accuracy_score(y_tr_sub, y_pred_train)

    y_pred_v = predict(weights, circuit, X_te_scaled)
    v_acc = accuracy_score(y_te_sub, y_pred_v)
    v_f1 = f1_score(y_te_sub, y_pred_v, average="macro")

    result = build_result_report(
        PROTOCOL, ARCH_NOTE, N_TRAIN_SUB, N_TEST_SUB, N_QUBITS, REPS,
        BATCH_SIZE, N_STEPS, train_time, final_train_loss, train_accuracy,
        y_te_sub, y_pred_v, v_acc, v_f1, le,
    )
    save_report(OUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
