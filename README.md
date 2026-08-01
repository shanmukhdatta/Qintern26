# Qintern'26

Repository containing research, pipelines, notebooks, and technical reports for Quantum Machine Learning tasks.


## Week Highlights

### [`week4/`](./week4/) — QSVM / VQC Quantum ML for Malware Family Classification

- **[Bonus Version 3](./week4/bonus_version3_local_kernel_xgboost/)**: Local trainable quantum kernel + XGBoost feature selection. Achieved **0.875 accuracy** (best result of the project, beating classical models by up to 20 points).
- **[Day 22 CTGAN Confirmation](./week4/day22_ctgan_confirmation/)**: Traced 87% baseline source (Random Forest, full CTGAN).
- **[Day 23 Version 1](./week4/day23_version1_smote_qsvm_vqc/)**: Baseline QSVM + VQC models on SMOTE data across qubit scales.
- **[Day 24–25 Capacity Sweep](./week4/day24-25_capacity_sweep/)**: Qubit capacity analysis and circuit scaling limitations.
- **[Day 26–27 Version 2 Hybrid Pipeline](./week4/day26-27_version2_hybrid_pipeline/)**: Hybrid LDA+PCA, data re-uploading feature map, Nyström kernel approximation, and 3-way SMOTE vs CTGAN vs Original comparison.
- **[Day 28 Technical Report & Writeup](./week4/day28_writeup/)**: Complete technical writeup, full notebooks, and comprehensive result tables.
- **[Uploaded Datasets & Scripts](./week4/uploaded_source_files_and_datasets/)**: Complete raw dataset files (`MalMem2022_SMOTE.csv`, `malmem_ctgan__1_.csv`, `malmem_original.csv`) and source scripts.

### [`Week_5/`](./Week_5/) — QTagger+ Days 29–35: Multi-Seed Statistical Correction, Fidelity Reconciliation & Final Consolidation

- **[Day 29 Downstream Classifier Comparison](./Week_5/Day29_Downstream_Classifier_Comparison/)**: Evaluated 4 augmentation methods (SMOTE, CTGAN, QGAN, Original) across RF, XGBoost, LightGBM, and SVM under identical evaluation protocols.
- **[Day 30 Fidelity Reconciliation](./Week_5/Day30_Fidelity_Reconciliation/)**: Computed KS, Wasserstein, and MMD distance metrics for all 4 augmentation methods.
- **[Day 31 QGAN Through V3 QSVM & Multi-Seed Correction](./Week_5/Day31_QGAN_QSVM_Pipeline/)**: Expanded seed testing (3-5 seeds) revealing that only Original raw data exhibits a robust, low-variance quantum win over classical models.
- **[Day 32 Master Comparison Tables](./Week_5/Day32_Master_Comparison_Tables/)**: Consolidated 27-config V2 grid and V3 multi-seed metrics into unified master comparison tables with zero open cells.
- **[Day 33 QGAN Stabilization & V1–V3 Methodology Writeup](./Week_5/Day33_Stabilization_Methodology/)**: Deep-dive diagnosis of QGAN training instability, output-bias calibration, quantile mapping, and capacity scaling.
- **[Day 34 V3 Supplementary Material](./Week_5/Day34_V3_Supplementary/)**: Comprehensive study cross-verifying QSVM v1/v2/v3 and VQC v1/v2/v3 pipelines.
- **[Day 35 Final Packaging & Consolidation](./Week_5/Day35_Final_Consolidation/)**: Package manifest, cross-day consistency verification, and final findings report (`FINAL_FINDINGS_REPORT.md`).

