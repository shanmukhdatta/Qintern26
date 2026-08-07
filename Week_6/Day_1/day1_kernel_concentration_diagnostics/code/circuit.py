"""
V3 local-kernel circuit with trainable per-qubit alignment rotation.

Fidelity-style kernel: encode x1 forward, apply a trainable RY(w_i) rotation
per qubit, then decode x2 via the adjoint embedding. The RY(w) block sits
between the forward and adjoint halves of the circuit, so it participates
directly in the fidelity overlap and is shaped by KTA training. Two readouts
are exposed: the full all-qubits-agree global fidelity, and a local variant
that averages the P(00) marginal over non-overlapping qubit pairs.
"""
import numpy as np
import pennylane as qml


def make_circuit(n_qubits, L=2):
    dev = qml.device('lightning.qubit', wires=n_qubits)
    pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]

    @qml.qnode(dev)
    def probs_circuit(x1, x2, w):
        for _ in range(L):
            qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        for i in range(n_qubits):
            qml.RY(w[i], wires=i)  # trainable per-qubit alignment rotation
        for _ in range(L):
            for i in reversed(range(n_qubits)):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
        return qml.probs(wires=range(n_qubits))

    def global_kernel(x1, x2, w):
        return probs_circuit(x1, x2, w)[0]

    def local_kernel(x1, x2, w):
        probs = probs_circuit(x1, x2, w)
        n_states = len(probs)
        vals = []
        for (a, b) in pairs:
            mask = np.array([(((idx >> (n_qubits - 1 - a)) & 1) == 0) and
                              (((idx >> (n_qubits - 1 - b)) & 1) == 0)
                              for idx in range(n_states)])
            vals.append(probs[mask].sum())
        return sum(vals) / len(vals)

    return global_kernel, local_kernel
