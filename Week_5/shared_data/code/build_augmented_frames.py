"""
Builds the real-vs-synthetic-flagged, downstream/fidelity-ready dataframes for
SMOTE and CTGAN from the raw uploaded files, matching the exact protocol
already used for QGAN in Days 29-31.

Key technique: neither SMOTE's nor CTGAN's CSV includes an explicit
is_synthetic flag. Real rows are identified via a hash-based duplicate-safe
match against the original dataset (exact-match on all 55 feature columns +
label) -- NOT a plain merge, which explodes on this dataset's duplicate rows.
Confirmed exact match against known real counts before trusting results:
SMOTE retains all real rows (9791/10020/9487 matched exactly per malware
class) plus synthetic fill to 29,298/class. CTGAN retains ZERO real rows --
every CTGAN row is synthetic, a finding worth knowing before interpreting its
fidelity/downstream numbers.
"""
import pandas as pd
import numpy as np

def flag_synthetic(df_new, key_cols, real_counts):
    key = pd.util.hash_pandas_object(df_new[key_cols], index=False)
    df_new = df_new.copy()
    df_new['_key'] = key
    df_new['_cumcount'] = df_new.groupby('_key').cumcount()
    df_new['_real_budget'] = df_new['_key'].map(real_counts).fillna(0).astype(int)
    df_new['is_synthetic'] = (df_new['_cumcount'] >= df_new['_real_budget']).astype(int)
    return df_new.drop(columns=['_key', '_cumcount', '_real_budget'])


def build(original_csv, smote_csv, ctgan_csv):
    df_orig = pd.read_csv(original_csv)
    df_orig['MalwareType'] = df_orig['Category'].str.split('-').str[0]
    df_orig['is_synthetic'] = 0

    feature_cols = [c for c in df_orig.columns
                     if c not in ('Category', 'Class', 'MalwareType', 'Filename', 'is_synthetic')
                     and pd.api.types.is_numeric_dtype(df_orig[c])]
    key_cols = feature_cols + ['MalwareType']

    real_counts = pd.util.hash_pandas_object(df_orig[key_cols], index=False).value_counts()

    df_smote = pd.read_csv(smote_csv).rename(columns={'Label': 'MalwareType'})
    df_smote = flag_synthetic(df_smote, key_cols, real_counts)

    df_ctgan = pd.read_csv(ctgan_csv).rename(columns={'Family': 'MalwareType'})
    df_ctgan = flag_synthetic(df_ctgan, key_cols, real_counts)

    df_aug_smote = pd.concat([df_orig, df_smote[df_smote['is_synthetic'] == 1]],
                              ignore_index=True, sort=False)
    df_aug_ctgan = pd.concat([df_orig, df_ctgan[df_ctgan['is_synthetic'] == 1]],
                              ignore_index=True, sort=False)

    return df_orig, df_aug_smote, df_aug_ctgan, feature_cols


if __name__ == '__main__':
    df_orig, df_aug_smote, df_aug_ctgan, feature_cols = build(
        '../malmem_original.csv', '../malmem_smote.csv', '../malmem_ctgan.csv')
    print("original:", df_orig.shape, "aug_smote:", df_aug_smote.shape,
          "aug_ctgan:", df_aug_ctgan.shape, "n_features:", len(feature_cols))
