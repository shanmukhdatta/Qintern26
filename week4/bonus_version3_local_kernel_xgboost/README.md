# Bonus (beyond assigned scope) — Version 3: local trainable kernel + XGBoost

Not part of the original Day 22-28 task list -- extra work done this week pushing QSVM further after v2.

Three changes vs v2: (1) XGBoost feature selection tested head-to-head against LDA+PCA hybrid -- XGBoost won decisively (mean acc 0.837 vs 0.716, beats classical in 9/9 configs vs 6/9). (2) Kernel-target alignment: trainable per-qubit rotation optimized via gradient-free COBYLA. (3) Local kernel: pair-averaged marginal fidelity instead of global joint measurement -- fixes kernel concentration (documented phenomenon where global kernels lose discriminative power as qubit count rises).

Best result: 0.875 accuracy / 0.873 F1 (q=8, CTGAN, beats classical by up to 20 points) -- highest of the entire project.

Ablation (isolating each change): global kernel no alignment = 0.700 (~v2 baseline) -> +local kernel alone = 0.825 -> +alignment alone = 0.725 -> both together = 0.875. Local kernel is ~5x more impactful than alignment alone.

Files: pipeline_v3_full.py, ablation.py, results_v3full.json, ablation_results.json, QSVM_v3_CTGAN_LocalKernel_XGBoost.ipynb, QSVM_v3_Architecture_Technical_Report.md.
