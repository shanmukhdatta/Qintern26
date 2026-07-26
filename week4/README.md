# Week 4 — QSVM/VQC Quantum ML for Malware Family Classification

Assigned scope was Day 22-28. Work done goes beyond that (Version 3 in `bonus_version3_local_kernel_xgboost/`).

## Folder structure (day-wise / version-wise)

- `day22_ctgan_confirmation/` — confirmed the 87% accuracy figure's exact source (classical RandomForest, full-scale CTGAN data)
- `day23_version1_smote_qsvm_vqc/` — Version 1 pipeline, QSVM+VQC on SMOTE, matched scales n=200/1000
- `day24-25_capacity_sweep/` — qubit-count capacity analysis (q=4 vs q=12)
- `day26-27_version2_hybrid_pipeline/` — Version 2 pipeline (LDA+PCA hybrid, data re-uploading, Nystrom kernel) + full SMOTE-vs-CTGAN-vs-Original three-way comparison
- `day28_writeup/` — full technical report + notebooks
- `bonus_version3_local_kernel_xgboost/` — Version 3 (local trainable kernel + XGBoost feature selection), best result of the project: 0.875 accuracy
- `uploaded_source_files_and_datasets/` — every raw file uploaded to the chat this week (datasets + the CTGAN-track notebook/script), unmodified, with a README mapping each file to what used it

See `WEEK_NOTES.md` for day-wise task completion status and blockers going into next week.
