# Uploaded source files and datasets

Every file uploaded to the chat this week, unmodified, collected in one place. These are the raw inputs — everything else in this repo was derived from these.

| File | Uploaded | What it is | Used by |
|---|---|---|---|
| `MalMem2022_SMOTE.csv` | Day 23 (initial task) | SMOTE-balanced CIC-MalMem-2022, 117,192 rows, 4-class (Benign/Ransomware/Spyware/Trojan) | Version 1 pipeline (`day23_version1_smote_qsvm_vqc/`), Version 2 grid extension, three-way comparison (`day26-27_version2_hybrid_pipeline/`) |
| `ctgan-retrianing.ipynb` | Day 22 confirmation | Full-scale CTGAN retrain notebook — source of the 87% RandomForest accuracy figure | `day22_ctgan_confirmation/` |
| `qsvm_training_pipeline.py` | Day 22 confirmation | CTGAN-track QSVM pipeline script (separate implementation from this project's own pipeline) | `day22_ctgan_confirmation/` — also the source of the kernel-design comparison that flagged the missing-entanglement gap in Version 1 |
| `qsvm_training_results_n_1000.json` | Day 22 confirmation | CTGAN-track QSVM result at n=1000: 69% accuracy, Nystrom kernel, 3-class (no Benign) | `day22_ctgan_confirmation/` |
| `malmem_ctgan__1_.csv` | Day 26-27 (three-way comparison) | CTGAN-augmented dataset, 46,257 rows, 3-class only (Trojan/Ransomware/Spyware, no Benign) | Three-way comparison, Version 3 (`bonus_version3_local_kernel_xgboost/`) |
| `malmem_original.csv` | Day 26-27 (three-way comparison) | Original, unaugmented, imbalanced CIC-MalMem-2022, 58,596 rows, 4-class derivable from `Category` | Three-way comparison |

Note: `malmem_original.csv` and `malmem_ctgan__1_.csv` are not directly usable by the day22/day23/day26-27 pipeline scripts without pointing those scripts' file paths at wherever this folder ends up after download — the scripts as written expect the original upload path (`/mnt/user-data/uploads/...`). Update the path constants at the top of each script if re-running outside this environment.
