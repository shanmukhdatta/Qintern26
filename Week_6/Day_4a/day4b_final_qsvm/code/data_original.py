"""
Loader for the Original (unaugmented) CIC-MalMem-2022 dataset.
Mirrors pipeline_v3_full.load_ctgan()'s interface/behavior exactly, so it can be
dropped into base_filter() / project_xgboost() / angle_scale() unchanged.

Original CIC-MalMem-2022 (raw, /mnt/user-data/uploads/MalMem2022.csv) ships with
'Class' (Benign/Malware) and 'Category' (Benign, or '<Family>-<Variant>' for
malware, e.g. 'Ransomware-Shade'). The 4-class label used throughout the project
(Benign / Ransomware / Spyware / Trojan) is the prefix of 'Category' before '-'.
"""
import os
import numpy as np
import pandas as pd

ORIGINAL_PATH = os.environ.get(
    'ORIGINAL_PATH', os.path.join(os.path.dirname(__file__), '..', 'data', 'MalMem2022.csv')
)


def load_original(n_total, seed=42):
    df = pd.read_csv(ORIGINAL_PATH)
    df['fam'] = df['Category'].astype(str).str.split('-').str[0]  # Benign/Ransomware/Spyware/Trojan
    per_class = n_total // 4
    parts = [g.sample(n=per_class, random_state=seed, replace=len(g) < per_class)
             for _, g in df.groupby('fam')]
    sub = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    y = sub['fam'].values
    drop_cols = [c for c in ['Class', 'Category', 'Filename', 'fam'] if c in sub.columns]
    X = sub.drop(columns=drop_cols).select_dtypes(include=[np.number])
    return X, y
