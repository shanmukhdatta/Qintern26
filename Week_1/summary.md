# Week 1 Summary — July 8, 2026

## Objective
To address severe class imbalance in memory malware datasets (CIC-MalMem-2022 and EMBER 2018) by implementing, evaluating, and benchmarking classical oversampling techniques (SMOTE, Borderline-SMOTE, ADASYN) against deep generative synthesis models (CTGAN, TVAE, cGAN) for downstream malware classification.

## Tasks Completed
- **Day 1:** Reviewed literature on CTGAN, SMOTE, ADASYN, TVAE, and cGAN; analyzed malware imbalance challenges including benign skew, overlapping feature spaces, and multi-space complexity.
- **Day 2:** Preprocessed CIC-MalMem-2022 (derived 4 clean classes from `Category`, 55 numeric features) and structured EMBER 2018 (2,351 static features); created train/val/test splits.
- **Day 3:** Implemented SMOTE, Borderline-SMOTE, and ADASYN on CIC-MalMem-2022, balancing the dataset from an initial imbalance ratio of 0.324 to near-perfect (~1.000).
- **Day 4:** Trained CTGAN, TVAE, and a custom cGAN to generate synthetic minority-class samples for generative dataset balancing.
- **Day 5:** Evaluated synthetic data quality (Original vs. SMOTE vs. CTGAN) using PCA, t-SNE, UMAP, and 7 statistical similarity metrics alongside downstream classifier validation.
- **Day 6:** Benchmarked Random Forest, XGBoost, LightGBM, and SVM across all balancing strategies evaluating Accuracy, Macro-F1, Weighted-F1, MCC, ROC-AUC, and PR-AUC.
- **Day 7:** Designed a generalization-validation protocol on EMBER 2018 incorporating artificial class imbalance for protocol testing.

## Challenges
- **CTGAN Undertraining:** Initial CTGAN evaluation used a reduced budget (50 epochs / 1,500 rows per class); full retrain (300+ epochs) is required for final conclusions.
- **Class & Multi-Family Gaps:** EMBER 2018 is binary-only, preventing 5-class validation; CIC-MalMem-2022 lacks the target `Worm` class (4 classes available).
- **Dataset Dependency:** Behavioral/sandbox multi-family dataset from Team A remains pending.

## Results
- **Oversampling Efficiency:** SMOTE, Borderline-SMOTE, and ADASYN successfully balanced CIC-MalMem-2022 without missing values or feature count degradation.
- **Generative Comparison:** TVAE trained faster and matched real data distributions better than CTGAN and cGAN.
- **Downstream Performance:** SMOTE outperformed 50-epoch CTGAN across metrics; 50-epoch CTGAN reduced downstream F1 to 0.80 vs. SMOTE (0.89) and raw baseline (0.80).
- **Best Baseline Model:** **Random Forest + Borderline-SMOTE** achieved peak performance (**Macro-F1: 0.93, ROC-AUC: 0.99**).

## Next Week's Plan
- Scale CTGAN training to 300+ epochs with full minority data to definitively evaluate deep generative synthesis.
- Complete the EMBER 2018 binary generalization experiment with artificial imbalance.
- Address the missing `Worm` class via secondary dataset acquisition or mentor guidance.
- Integrate and validate on Team A's behavioral/sandbox dataset upon arrival.
