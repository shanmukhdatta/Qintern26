# Day 26-27 — Version 2 pipeline + three-way comparison

Version 2 changes vs v1: LDA(capped at n_classes-1) + PCA hybrid projection (supervised+unsupervised), data re-uploading feature map (L=2, ring-CNOT entangling), Nystrom-approximated kernel (40 landmarks) for QSVM, Adam optimizer confirmed better than COBYLA for VQC (isolation-tested).

Literature-informed: 4 papers reviewed (kernel malware classification, feature engineering for QML, QML survey, cybersecurity QML taxonomy) -- techniques adapted, not copy-pasted.

Also includes: full three-way comparison (SMOTE vs CTGAN vs Original, 27 configs, q=8/10/12 x n=250/500/1000). Finding: CTGAN easiest for classical (mean acc 0.70), quantum wins most (56%) on the real/unaugmented Original dataset, especially at n=1000 (QSVM 0.558 vs classical 0.317).

Confirmed Day 22 finding along the way: the "87%" figure was classical RandomForest on full-scale CTGAN data, not a quantum result (see day22 folder).

Files: pipeline_v2.py, run_grid_config.py, vqc_reupload_adam.py (optimizer isolation test), three_way_comparison_pipeline.py, results_*.json, Comparison_*.md, Detailed_Report_3Way_Comparison.md, QuantumVsClassical_3Way_Showcase.ipynb (notebook with real outputs).
