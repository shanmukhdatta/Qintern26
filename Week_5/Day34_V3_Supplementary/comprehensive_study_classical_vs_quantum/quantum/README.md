# Quantum models — QSVM and VQC, versions 1-3

One script (quantum_all_versions.py) implements all six model/version combinations, parameterized by dataset. Results are split into subfolders by model+version for easy navigation:

- qsvm_v1/ — PCA-only projection, single-pass angle encoding, no entanglement, global fidelity kernel
- qsvm_v2/ — LDA+PCA hybrid, data re-uploading (L=2), Nystrom-approximated kernel
- qsvm_v3/ — XGBoost feature selection, local pair-averaged kernel, kernel-target alignment (BEST: 0.875 on CTGAN)
- vqc_v1/ — PCA-only, single-pass encoding, StronglyEntanglingLayers, Adam
- vqc_v2/ — LDA+PCA hybrid, re-upload circuit, Adam
- vqc_v3/ — XGBoost feature selection, re-upload circuit, near-identity informed initialization, Adam

Each JSON file (e.g. qsvm_v3/qsvm_v3_ctgan.json) contains, per run: the quantum result (accuracy, F1, timing), AND the classical baseline trained on the identical same-feature inputs -- so every single result file is independently checkable for the quantum-vs-classical comparison without needing to cross-reference the classical/ folder.

See ../comparative_results/COMPARATIVE_RESULTS.md for the consolidated analysis and goal-check.
