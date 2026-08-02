# Week 2 Summary — July 15, 2026

## Objective
To execute a full-scale retrain of CTGAN (300+ epochs, un-capped minority samples), evaluate synthetic sample fidelity via statistical tests and feature projections, and benchmark downstream classifiers (RF, XGBoost, LightGBM, SVM) across augmented variants to produce a corrected baseline and investigate Worm-class dataset availability.

## Tasks Completed
- **Day 1:** Reconfigured CTGAN for full-scale retraining (300+ epochs, full minority row counts, loss logging, mid-run checkpointing); initiated secondary search for `Worm`-class samples (SOREL-20M, MalwareBazaar, Malimg) and flagged missing class to mentors.
- **Day 2:** Completed full-scale CTGAN-v2 training on CIC-MalMem-2022; evaluated synthetic sample fidelity against Original, SMOTE, and ADASYN using feature-wise KS-tests, correlation-matrix deltas, and PCA/t-SNE overlays.
- **Day 3:** Benchmarked Random Forest, XGBoost, LightGBM, and SVM across Original, SMOTE, ADASYN, Borderline-SMOTE, TVAE, and CTGAN-v2 dataset variants; ran EMBER 2018 as a binary generalization check and finalized the Worm-class decision memo and Augmentation Comparison Report.

## Challenges
- **Generative vs. Classical Oversampling Gap:** Although full-scale retrained CTGAN-v2 improved over Week 1's undertrained model, it still trails classical interpolation methods (SMOTE/ADASYN) in tabular feature fidelity and downstream performance.
- **Worm-Class Memory Dump Deficit:** Public repositories (SOREL-20M, MalwareBazaar) provide binary samples or static features rather than volatile memory dumps matching CIC-MalMem-2022's 55 features; mentor decision memo submitted.
- **Classifier Sensitivity:** SVM models struggled significantly on high-dimensional synthetic tabular features (Macro-F1 ~0.58–0.75) compared to tree-based ensembles.

## Results
- **Full CTGAN-v2 Retrain Improvement:** CTGAN-v2 accuracy reached **0.8795** (LightGBM) and **0.8725** (Random Forest) with Macro-F1 of **0.8183**, significantly correcting Week 1's premature verdict (0.80 F1).
- **Top Downstream Performer:** **SMOTE + Random Forest** achieved highest performance (**Accuracy: 0.9324, Macro-F1: 0.9324, MCC: 0.9099, ROC-AUC: 0.9899, PR-AUC: 0.9719**), closely followed by **ADASYN + Random Forest** (**Macro-F1: 0.9234, ROC-AUC: 0.9889**).
- **Secondary Variants:** Borderline-SMOTE + XGBoost achieved Macro-F1 of **0.8191** (ROC-AUC: 0.9732); TVAE + XGBoost yielded Macro-F1 of **0.7862** (ROC-AUC: 0.9420).
- **EMBER 2018 Generalization:** Verified binary classification consistency across oversampling strategies on external static PE features.

## Next Week's Plan
- Finalize mentor sign-off on the Worm-class dataset strategy (4-class memory alignment vs. synthetic generation/external feature mapping).
- Transition to hybrid/advanced augmentation strategies (e.g., SMOTE-CTGAN hybrids or feature selection filtering) to bridge the generative gap.
- Prepare baseline feature pipeline for upcoming multi-family behavioral dataset integration.
