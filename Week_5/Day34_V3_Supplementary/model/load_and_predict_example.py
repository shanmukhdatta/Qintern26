"""
Example: how to load the pickled model bundle and run inference on new data.

The pickled bundle contains every fitted component needed to reproduce the
0.875-accuracy pipeline end to end: preprocessing scalers, the XGBoost feature
selector, the trained quantum kernel-alignment weights, the Nystrom landmark
transform, and the final linear SVM. This script shows the exact call order.

Note: the quantum kernel functions themselves (make_local_trainable_circuit)
live in pipeline_v3_full.py, not inside the pickle -- pickle can't serialize
a live PennyLane QNode. Import that module alongside this script.
"""
import pickle
import numpy as np
import pandas as pd
from pipeline_v3_full import make_local_trainable_circuit

def load_bundle(path='model_bundle_xgboost_q8_seed7_acc0875.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict(bundle, X_new_raw: pd.DataFrame):
    """
    X_new_raw: a DataFrame with the SAME raw column set as the original
    malmem_ctgan__1_.csv (before any filtering), one row per sample.
    """
    meta = bundle['meta']
    prep = bundle['preprocessing']
    quantum = bundle['quantum']
    clf = bundle['classifier']

    # 1. apply the same column filtering used at train time
    X = X_new_raw[meta['kept_columns_after_filter']]
    X = X.drop(columns=[c for c in meta['dropped_correlated_columns'] if c in X.columns])

    # 2. same log transform (no fitted params)
    X = np.sign(X) * np.log1p(np.abs(X))

    # 3. StandardScaler (already fit on training data)
    X_s = prep['standard_scaler'].transform(X)

    # 4. XGBoost-selected feature indices (already chosen on training data)
    X_p = X_s[:, prep['selected_feature_indices']]

    # 5. RobustScaler + clip -> angle range (already fit on training data)
    X_r = np.clip(prep['robust_scaler'].transform(X_p), -3, 3) / 3 * np.pi

    # 6. rebuild the quantum kernel circuit (same n_qubits, same L)
    local_kernel, _ = make_local_trainable_circuit(quantum['n_qubits'], L=quantum['L'])
    w = quantum['trained_alignment_weights']
    landmarks = quantum['landmark_points']

    # 7. compute similarity of each new sample against the fixed landmark points
    K_new_land = np.array([[local_kernel(x, lp, w) for lp in landmarks] for x in X_r])
    psi_new = K_new_land @ quantum['K_MM_inv_sqrt']

    # 8. classify
    pred_idx = clf['svm'].predict(psi_new)
    idx_to_class = {v: k for k, v in meta['class_to_idx'].items()}
    return [idx_to_class[i] for i in pred_idx]

if __name__ == '__main__':
    bundle = load_bundle()
    print("Loaded model trained on:", bundle['meta']['dataset'])
    print("Recorded accuracy at save time:", bundle['meta']['accuracy'])
    print("Classes:", bundle['meta']['classes'])
    print()
    print("To predict on new data, call predict(bundle, your_dataframe)")
    print("your_dataframe must have the same raw column schema as malmem_ctgan__1_.csv")
