# Day 23 — Version 1: QSVM + VQC on SMOTE data

Pipeline: variance filter -> correlation filter (train-only) -> log1p -> StandardScaler -> PCA(=n_qubits) -> MinMax[0,pi] -> single-pass AngleEmbedding (no entanglement).

Ran QSVM + VQC at matched scales n=200 and n=1000, 3 qubit widths each (4/6/8). Best: VQC n=200 51.7% acc, VQC n=1000 44.7% acc. Classical baseline (RF/SVM) beat both quantum models by 8-17 points on identical inputs (results_baseline_check.json). Seed-stability check across 3 seeds in results_multiseed.json.

Files: preprocess.py, quantum_models.py (core v1 pipeline), verify.py (classical baseline check), multiseed.py (seed stability), results_*.json (raw outputs).
