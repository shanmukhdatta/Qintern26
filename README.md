# Qintern'26 — QTagger+
## Quantum Machine Learning & Data Augmentation for Malware Family Classification

Welcome to the **Qintern'26 — QTagger+** repository. This repository contains the complete research, experimental pipelines, Jupyter notebooks, datasets, statistical evaluations, and technical write-ups covering **Weeks 1 through 5** (Days 1–35) of the Quantum Machine Learning & Generative Augmentation project for malware classification.

---

## 📌 Executive Summary & Key Research Takeaways

Across 5 weeks of rigorous empirical research, this project investigated the interplay between **data augmentation** (SMOTE, Borderline-SMOTE, ADASYN, CTGAN, TVAE, QGAN) and **Quantum Machine Learning** (QSVM, VQC) on memory malware datasets (CIC-MalMem-2022 and EMBER 2018).

### 🔍 Core Findings:
1. **Quantum Advantage Reality Check (Multi-Seed Correction)**: While initial single-seed experiments yielded a headline **87.5% accuracy** for QSVM v3 on CTGAN data, multi-seed statistical testing (3–5 seeds) revealed that **only Original raw data exhibits a robust, low-variance quantum advantage over classical models ($0.542 \pm 0.031$, 3/3 win rate)**. Synthetic/augmented datasets add noise that reduces quantum separability.
2. **Fidelity vs. Utility Paradox**: Statistical fidelity metrics (KS-distance, Wasserstein distance, MMD) rank synthetic generators as **SMOTE > CTGAN > QGAN**. However, downstream classifier utility ranks **SMOTE >> QGAN $\approx$ CTGAN** (CTGAN can actually degrade downstream classification relative to raw imbalanced data). High synthetic volume paired with distribution fidelity (SMOTE) moves classifier metrics, whereas low-volume/high-variance generative models struggle.
3. **Architectural Breakthrough (v1 $\rightarrow$ v2 $\rightarrow$ v3)**: Resolving global kernel concentration required moving from unsupervised PCA + single-pass angle encoding (v1) to supervised LDA+PCA + data re-uploading (v2), and ultimately to **XGBoost Feature Selection + Trainable $RY(w)$ Mid-Layers + Kernel-Target Alignment (KTA) + Local Pairwise Measurement (v3)**.

---

## 🗓️ Weekly Progress & Milestones

### 📁 [`Week_1/`](./Week_1/) — Literature Review, Oversampling Baselines & Initial CTGAN
- **Focus**: Dataset acquisition, preprocessing, and preliminary balancing of CIC-MalMem-2022 (imbalance ratio 0.324) and EMBER 2018.
- **Key Deliverables**:
  - Preprocessed CIC-MalMem-2022 into 4 clean malware classes and 55 numerical features.
  - Implemented classical oversamplers (**SMOTE**, **Borderline-SMOTE**, **ADASYN**), bringing imbalance ratios to near-1.000.
  - Trained initial **CTGAN**, **TVAE**, and **cGAN** models under a 50-epoch budget.
  - Benchmarked baseline classifiers (**Random Forest**, **XGBoost**, **LightGBM**, **SVM**).
  - *Result*: Classical Random Forest + Borderline-SMOTE led with **Macro-F1: 0.93, ROC-AUC: 0.99**. Identified 50-epoch CTGAN undertraining as a key bottleneck.
- **Documentation**: [`Week_1/summary.md`](./Week_1/summary.md)

---

### 📁 [`Week_2/`](./Week_2/) — Full-Scale CTGAN Retrain, Fidelity & Downstream Correction
- **Focus**: Full-scale CTGAN-v2 retraining (300+ epochs, un-capped minority samples), feature-level fidelity evaluation, and multi-classifier re-benchmarking.
- **Key Deliverables**:
  - Retrained **CTGAN-v2** with checkpointing and loss logging to prevent mode collapse.
  - Evaluated synthetic sample fidelity via Kolmogorov-Smirnov (KS) tests, correlation matrix deltas, and PCA/t-SNE projections.
  - Corrected premature Week 1 findings: CTGAN-v2 downstream accuracy rose to **87.95% (LightGBM)** and **87.25% (Random Forest)** with Macro-F1 of **0.8183**.
  - **SMOTE + Random Forest** retained the top classical spot (**Accuracy: 0.9324, Macro-F1: 0.9324, ROC-AUC: 0.9899**).
  - Documented `Worm`-class deficit in public volatile memory dump repositories (SOREL-20M, MalwareBazaar binary-only).
- **Documentation**: [`Week_2/summary.md`](./Week_2/summary.md), CSV metrics in [`Week_2/outputs/`](./Week_2/outputs/)

---

### 📁 [`Week_3/`](./Week_3/) — Quantum ML Pipeline Integration & Capacity Diagnosis
- **Focus**: Integration of CTGAN-v2 augmented data into the 8-qubit quantum pipeline, benchmarking QSVM and VQC models, and diagnosing quantum performance bottlenecks.
- **Key Deliverables**:
  - Formatted augmented data through Team C's pipeline (variance filter, 0.95 correlation filter, StandardScaler, 8-qubit PCA).
  - Ran **Exact Fidelity Kernel QSVM** at $n=200$ (**63.3% accuracy / 0.625 Macro-F1** vs. classical matched control **60.0% / 0.592**).
  - Resolved **VQC barren plateaus** by pruning `StronglyEntanglingLayers` depth from 2 reps to 1 rep (**43.3% accuracy**).
  - Scaled models to $n=1000$ using **50-landmark Nyström approximation** (**QSVM: 69.0% accuracy / 0.683 Macro-F1**).
  - Demonstrated that data augmentation provides a **+26.6% accuracy lift** for quantum models (preventing single-class Trojan collapse), compared to +15.0% for classical models.
  - Diagnosed capacity constraints: Qubit scaling ($4 \rightarrow 8 \rightarrow 12$ qubits) produced steady accuracy gains ($42\% \rightarrow 58\% \rightarrow 68\%$) matching retained PCA variance ($45.6\% \rightarrow 54.6\% \rightarrow 62.0\%$).
- **Documentation**: [`Week_3/summary.md`](./Week_3/summary.md)

---

### 📁 [`Week_4/`](./Week_4/) — Architectural Evolution: QSVM/VQC Pipeline v1 $\rightarrow$ v2 $\rightarrow$ v3
- **Focus**: Systematic engineering of quantum pipeline architectures to eliminate classical performance gaps and kernel concentration.
- **Key Subdirectories & Architecture Phases**:
  - **[`day22_ctgan_confirmation/`](./Week_4/day22_ctgan_confirmation/)**: Traced and confirmed 87% Random Forest baseline on full CTGAN retrain.
  - **[`day23_version1_smote_qsvm_vqc/`](./Week_4/day23_version1_smote_qsvm_vqc/)**: **Pipeline v1 Baseline** — Unsupervised PCA + non-entangled `AngleEmbedding`. (Classical beat quantum by 8–17 points).
  - **[`day24-25_capacity_sweep/`](./Week_4/day24-25_capacity_sweep/)**: Qubit capacity sweep ($q=4$ vs $q=12$) demonstrating output-variance decay (3.4x) without accuracy gain under global fidelity.
  - **[`day26-27_version2_hybrid_pipeline/`](./Week_4/day26-27_version2_hybrid_pipeline/)**: **Pipeline v2** — Supervised LDA+PCA hybrid projection + data re-uploading ($L=2$) + 40-landmark Nyström approximation. (QSVM reversed classical gap on SMOTE: 67.5% vs 45.0–62.5%).
  - **[`bonus_version3_local_kernel_xgboost/`](./Week_4/bonus_version3_local_kernel_xgboost/)**: **Pipeline v3 (Winner 🏆)** — XGBoost Feature Selection + Trainable $RY(w)$ mid-layers + COBYLA Kernel-Target Alignment (KTA) + Local Pairwise Kernel Measurement. Achieved **87.5% single-seed accuracy / 0.873 F1**.
  - **[`day28_writeup/`](./Week_4/day28_writeup/)**: Final technical report, comprehensive tables, and full Jupyter notebooks.
- **Documentation**: [`Week_4/README.md`](./Week_4/README.md), [`Week_4/WEEK_NOTES.md`](./Week_4/WEEK_NOTES.md)

---

### 📁 [`Week_5/`](./Week_5/) — QTagger+: Multi-Seed Correction, Fidelity Reconciliation & Final Consolidation
- **Focus**: Days 29–35 sprint: QGAN integration, multi-seed statistical correction, distance-based fidelity metrics, and master consolidation.
- **Key Deliverables & Subdirectories**:
  - **[`Day29_Downstream_Classifier_Comparison/`](./Week_5/Day29_Downstream_Classifier_Comparison/)**: Evaluated all 4 augmentation methods (SMOTE, CTGAN, QGAN, Original) across RF, XGBoost, LightGBM, SVM on identical held-out test splits.
  - **[`Day30_Fidelity_Reconciliation/`](./Week_5/Day30_Fidelity_Reconciliation/)**: Computed KS-distance, Wasserstein distance, and MMD across all 4 augmentation methods (**SMOTE > CTGAN > QGAN**).
  - **[`Day31_QGAN_QSVM_Pipeline/`](./Week_5/Day31_QGAN_QSVM_Pipeline/)**: Expanded seed testing (3–5 seeds per dataset). Uncovered that single-seed 0.875 CTGAN headline was a high-variance outlier ($0.708 \pm 0.170$). Confirmed **only Original raw data shows a robust quantum win ($0.542 \pm 0.031$, 3/3 win rate)**.
  - **[`Day32_Master_Comparison_Tables/`](./Week_5/Day32_Master_Comparison_Tables/)**: Integrated 27-config V2 grid and V3 multi-seed metrics into unified master comparison tables with zero open cells.
  - **[`Day33_Stabilization_Methodology/`](./Week_5/Day33_Stabilization_Methodology/)**: Documented QGAN loss-curve capacity mismatch diagnosis and 3 stabilization attempts (output-bias calibration, quantile mapping, generator capacity expansion).
  - **[`Day34_V3_Supplementary/`](./Week_5/Day34_V3_Supplementary/)** & **[`Day35_Final_Consolidation/`](./Week_5/Day35_Final_Consolidation/)**: Final findings report, package manifest, and verification logs.
- **Documentation**: [`Week_5/README.md`](./Week_5/README.md), [`Week_5/FINAL_FINDINGS_REPORT.md`](./Week_5/FINAL_FINDINGS_REPORT.md)

---

## 🗺️ Workspace Directory Overview

```
Qintern26/
├── Datasets/                                # Raw and processed dataset files
├── Week_1/                                  # Week 1: Oversampling & CTGAN baseline study
│   ├── code/                                # Data preprocessing & oversampling scripts
│   ├── ouputs/                              # Generated dataset CSVs and metrics
│   └── summary.md                           # One-page executive summary (Week 1)
├── Week_2/                                  # Week 2: CTGAN-v2 retrain & downstream benchmarks
│   ├── code/                                # CTGAN retrain & notebook pipelines
│   ├── outputs/                             # Model comparison CSVs & full synthetic datasets
│   ├── notes.md                             # Experimental notes
│   └── summary.md                           # One-page executive summary (Week 2)
├── Week_3/                                  # Week 3: Quantum ML pipeline & capacity diagnosis
│   ├── code/                                # QSVM & VQC initial quantum pipelines
│   ├── outputs/                             # Quantum benchmark outputs
│   ├── notes.md                             # Experimental notes
│   └── summary.md                           # One-page executive summary (Week 3)
├── Week_4/                                  # Week 4: Architectural evolution (v1 -> v2 -> v3)
│   ├── bonus_version3_local_kernel_xgboost/ # v3 Local trainable kernel & KTA alignment
│   ├── day22_ctgan_confirmation/            # 87% classical baseline confirmation
│   ├── day23_version1_smote_qsvm_vqc/       # v1 Baseline pipeline
│   ├── day24-25_capacity_sweep/             # Qubit capacity sweep experiments
│   ├── day26-27_version2_hybrid_pipeline/   # v2 LDA+PCA hybrid pipeline & Nyström QSVM
│   ├── day28_writeup/                       # Week 4 technical report & notebooks
│   ├── uploaded_source_files_and_datasets/  # Raw dataset CSVs & uploaded scripts
│   ├── WEEK_NOTES.md                        # Day-by-day technical log
│   └── README.md                            # Comprehensive Week 4 guide
├── Week_5/                                  # Week 5: QTagger+ multi-seed & master consolidation
│   ├── Day29_Downstream_Classifier_Comparison/
│   ├── Day30_Fidelity_Reconciliation/
│   ├── Day31_QGAN_QSVM_Pipeline/            # Multi-seed statistical correction analysis
│   ├── Day32_Master_Comparison_Tables/      # Master comparison tables (zero open cells)
│   ├── Day33_Stabilization_Methodology/     # QGAN stabilization diagnosis & attempted fixes
│   ├── Day34_V3_Supplementary/              # V3 supplementary findings
│   ├── Day35_Final_Consolidation/           # Final package manifest & cross-day verification
│   ├── shared_data/                         # Duplicate-safe hash matching & shared CSVs
│   ├── FINAL_FINDINGS_REPORT.md             # Comprehensive final research report
│   └── README.md                            # Comprehensive Week 5 guide
└── README.md                                # Root repository guide (this file)
```

---

## 📊 Summary Comparison Matrix

| Augmentation Method | Statistical Fidelity Rank | Classical Classifier F1 (RF/XGB) | Quantum QSVM v3 Accuracy (Single Seed) | Corrected Quantum QSVM v3 Accuracy (Multi-Seed Mean ± Std) | Quantum Win Rate vs Classical |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original (Imbalanced)** | Baseline | 0.800 | 0.500 | **$0.542 \pm 0.031$** | **3 / 3 (100%)** |
| **SMOTE** | **1 (Best)** | **0.932** | 0.600 | $0.444 \pm 0.094$ | 1 / 4 (25%) |
| **CTGAN-v2** | 2 | 0.818 | **0.875** *(Outlier)* | $0.708 \pm 0.170$ | 1 / 3 (33%) |
| **QGAN** | 3 | 0.812 | 0.625 | $0.540 \pm 0.049$ | 1 / 5 (20%) |

---

## 💻 Environment & Requirements

- **Python**: 3.10+
- **Quantum Machine Learning**: `pennylane`, `qiskit`, `scikit-learn`
- **Generative Oversampling**: `ctgan`, `imbalanced-learn` (`imblearn`)
- **Classical ML**: `xgboost`, `lightgbm`, `scikit-learn`
- **Data & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

---

## 📜 Citation & Credits
Developed as part of the **Qintern 2026 Internship Project — QTagger+** by **Shanmukh** in collaboration with team members and mentors.
