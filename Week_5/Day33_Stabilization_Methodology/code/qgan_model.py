"""
qgan_model.py
Hybrid quantum-classical GAN.

Generator  : d-qubit variational circuit. Latent noise z sets RY angle-embedding
             per qubit; a shared trainable (L, d, 3) tensor drives RX/RY/RZ
             rotation layers separated by a ring of CNOTs; Pauli-Z expectation
             values on all qubits are the synthetic sample (bounded to [-1,1]).
Discriminator: classical 2-layer MLP (LeakyReLU + dropout) -> sigmoid.

Stabilization (from Q-SYNTH): instance noise, label smoothing, feature matching,
moment matching, gradient-norm clipping.

Note on scope: unlike Q-SYNTH's classical front-end that reshapes z into the
rotation angles, here the rotation tensor Theta is a directly-trained parameter
(a Born-machine-style generator) and z only drives the angle embedding. This is
a deliberate simplification to keep the circuit compact and training stable on
CPU simulation; the entangling-layer structure and readout are unchanged.
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

import config

N_QUBITS = config.N_QUBITS
N_LAYERS = config.N_VARIATIONAL_LAYERS

# default.qubit + backprop lets PennyLane broadcast the whole batch through the
# circuit in one vectorized pass instead of looping per-sample (~3x faster than
# lightning.qubit + parameter-shift/adjoint for this circuit size, and simpler).
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="autograd", diff_method="backprop")
def _quantum_generator_circuit(z, theta):
    # Angle embedding: one RY per qubit from the latent vector (batched over z's leading dim)
    for q in range(N_QUBITS):
        qml.RY(z[..., q], wires=q)
    # Variational entangling layers (shared trainable theta across the batch)
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(theta[l, q, 0], wires=q)
            qml.RY(theta[l, q, 1], wires=q)
            qml.RZ(theta[l, q, 2], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


def generate_batch(Z, theta):
    """Z: (B, N_QUBITS) latent batch -> (B, N_QUBITS) synthetic samples in [-1,1]."""
    outs = _quantum_generator_circuit(Z, theta)
    return pnp.stack(outs, axis=-1)


def sample_latent(batch_size):
    return pnp.array(np.random.uniform(-np.pi / 2, np.pi / 2, size=(batch_size, N_QUBITS)))


def init_generator_params():
    theta = pnp.array(
        0.1 * np.random.randn(N_LAYERS, N_QUBITS, 3), requires_grad=True
    )
    return theta


# ---------------- Discriminator ----------------

def init_discriminator_params():
    h1, h2, d = config.DISC_HIDDEN_1, config.DISC_HIDDEN_2, N_QUBITS
    rng = np.random.RandomState(config.RANDOM_STATE)
    params = {
        "W1": pnp.array(rng.randn(h1, d) * np.sqrt(2.0 / d), requires_grad=True),
        "b1": pnp.array(np.zeros(h1), requires_grad=True),
        "W2": pnp.array(rng.randn(h2, h1) * np.sqrt(2.0 / h1), requires_grad=True),
        "b2": pnp.array(np.zeros(h2), requires_grad=True),
        "w": pnp.array(rng.randn(h2) * np.sqrt(2.0 / h2), requires_grad=True),
        "b": pnp.array(0.0, requires_grad=True),
    }
    return params


def _leaky_relu(x, alpha=config.LEAKY_RELU_ALPHA):
    return pnp.where(x > 0, x, alpha * x)


def discriminator_forward(X, params, dropout_mask1=None, dropout_mask2=None):
    """X: (B, d). Returns (probs (B,), hidden_features h2 (B, h2)) for feature matching."""
    U1 = X @ params["W1"].T + params["b1"]
    V1 = _leaky_relu(U1)
    if dropout_mask1 is not None:
        V1 = V1 * dropout_mask1 / (1.0 - config.DISC_DROPOUT_P)
    U2 = V1 @ params["W2"].T + params["b2"]
    H2 = _leaky_relu(U2)
    if dropout_mask2 is not None:
        H2 = H2 * dropout_mask2 / (1.0 - config.DISC_DROPOUT_P)
    logits = H2 @ params["w"] + params["b"]
    probs = 1.0 / (1.0 + pnp.exp(-logits))
    return probs, H2


def bce(probs, target):
    eps = 1e-8
    probs = pnp.clip(probs, eps, 1 - eps)
    return -pnp.mean(target * pnp.log(probs) + (1 - target) * pnp.log(1 - probs))


def add_instance_noise(X, sigma):
    noise = np.random.normal(0, sigma, size=X.shape)
    return pnp.clip(X + noise, -1.0, 1.0)


def moment_matching_loss(x_real, x_fake):
    mu_r, mu_f = pnp.mean(x_real, axis=0), pnp.mean(x_fake, axis=0)
    sd_r = pnp.sqrt(pnp.mean((x_real - mu_r) ** 2, axis=0) + 1e-6)
    sd_f = pnp.sqrt(pnp.mean((x_fake - mu_f) ** 2, axis=0) + 1e-6)
    return config.MOMENT_MATCH_ALPHA * pnp.sum(pnp.abs(mu_r - mu_f)) + \
        config.MOMENT_MATCH_BETA * pnp.sum(pnp.abs(sd_r - sd_f))


def feature_matching_loss(feat_real, feat_fake):
    return config.FEATURE_MATCHING_WEIGHT * pnp.sum(
        pnp.abs(pnp.mean(feat_real, axis=0) - pnp.mean(feat_fake, axis=0))
    )


def clip_grad_(grads_dict_or_list, max_norm=config.GRAD_CLIP_NORM):
    """In-place-style clip: returns rescaled grads with global L2 norm <= max_norm."""
    if isinstance(grads_dict_or_list, dict):
        flat = np.concatenate([np.asarray(g).ravel() for g in grads_dict_or_list.values()])
        norm = np.linalg.norm(flat)
        if norm > max_norm:
            scale = max_norm / (norm + 1e-8)
            return {k: g * scale for k, g in grads_dict_or_list.items()}
        return grads_dict_or_list
    else:
        norm = np.linalg.norm(np.asarray(grads_dict_or_list).ravel())
        if norm > max_norm:
            scale = max_norm / (norm + 1e-8)
            return grads_dict_or_list * scale
        return grads_dict_or_list
