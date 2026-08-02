# Day 22 - Confirmed

The 87% is RandomForest, accuracy=0.8725, from ctgan-retrianing.ipynb.

Config: full CIC-MalMem-2022 (58,596 rows), 4-class family label, 80/20
stratified split (test set 100% real, never touched by CTGAN), CTGAN
full-scale retrain (no 1,500-row cap, 320 epochs/class), balanced up to
the real Benign train count (23,438/class), RF trained on all 55 raw
features, no PCA, classical only - not a quantum result.

LightGBM actually scored higher in the same run (accuracy 0.8795, macro F1
0.8183 vs RF's 0.8082) and was flagged BEST MODEL in the notebook.

Caveat: the notebook cell that trains these models is currently commented
out - the 87% is a real prior output but not reproducible by re-running
the notebook top to bottom as uploaded.

CTGAN-track QSVM (found while confirming this): 69% accuracy at n=1000,
Nystrom kernel, q=8, 50 landmarks (see qsvm_training_results_n_1000.json).
Classical RBF-SVM on the same subsample: 71%. Note this QSVM run is
3-class only (Benign excluded) - not directly comparable to the SMOTE
track's 4-class numbers yet.
