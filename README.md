# Qintern'26 — Quantum Machine Learning Internship (2026)

Repository containing research, pipelines, notebooks, and technical reports for Quantum Machine Learning tasks.

## Directory Structure

```
.
├── README.md
└── week4/
    ├── README.md
    ├── WEEK_NOTES.md
    ├── NOTION_NOTES.md
    ├── day22_ctgan_confirmation/
    ├── day23_version1_smote_qsvm_vqc/
    ├── day24-25_capacity_sweep/
    ├── day26-27_version2_hybrid_pipeline/
    ├── day28_writeup/
    ├── bonus_version3_local_kernel_xgboost/
    └── uploaded_source_files_and_datasets/
```

## Week Highlights

### [`week4/`](./week4/) — QSVM / VQC Quantum ML for Malware Family Classification

- **[Bonus Version 3](./week4/bonus_version3_local_kernel_xgboost/)**: Local trainable quantum kernel + XGBoost feature selection. Achieved **0.875 accuracy** (best result of the project, beating classical models by up to 20 points).
- **[Day 22 CTGAN Confirmation](./week4/day22_ctgan_confirmation/)**: Traced 87% baseline source (Random Forest, full CTGAN).
- **[Day 23 Version 1](./week4/day23_version1_smote_qsvm_vqc/)**: Baseline QSVM + VQC models on SMOTE data across qubit scales.
- **[Day 24–25 Capacity Sweep](./week4/day24-25_capacity_sweep/)**: Qubit capacity analysis and circuit scaling limitations.
- **[Day 26–27 Version 2 Hybrid Pipeline](./week4/day26-27_version2_hybrid_pipeline/)**: Hybrid LDA+PCA, data re-uploading feature map, Nyström kernel approximation, and 3-way SMOTE vs CTGAN vs Original comparison.
- **[Day 28 Technical Report & Writeup](./week4/day28_writeup/)**: Complete technical writeup, full notebooks, and comprehensive result tables.
- **[Uploaded Datasets & Scripts](./week4/uploaded_source_files_and_datasets/)**: Complete raw dataset files (`MalMem2022_SMOTE.csv`, `malmem_ctgan__1_.csv`, `malmem_original.csv`) and source scripts.
