"""
main.py
End-to-end pipeline (now crash-safe / resumable):
  1. load malmem_original.csv
  2. for each minority malware family (Category != Benign): fit per-class
     Standardize->PCA(6)->[-1,1] representation, train the hybrid QGAN,
     sample enough synthetic points to balance that class up to the target count
  3. inverse-transform synthetic points back into the original 54-feature schema
  4. write out the augmented CSV (original rows + synthetic rows, flagged)
  5. compute fidelity metrics (KS, Wasserstein, MMD) per class
  6. run the downstream classifier comparison (RF/XGBoost/LightGBM/SVM),
     Original-only vs QGAN-augmented, on a held-out real test split

Resume behaviour:
  - Per-class QGAN training checkpoints every config.CHECKPOINT_EVERY epochs
    (see train_qgan.py). If this script is killed/crashes mid-class, re-running
    it resumes that class from its last checkpoint instead of from epoch 0.
  - Once a class's synthetic samples have been generated, they are cached to
    outputs/partial_synth_<class>.csv and a marker file is written. On restart,
    already-finished classes are loaded from cache instead of being retrained.
  - Any uncaught exception is logged with full traceback to logs/training_run.log
    (crash log) before the process exits, so nothing is lost silently.
"""

import os
import sys
import json
import time
import logging
import traceback
import numpy as np
import pandas as pd

import config
import data_prep
import train_qgan
import evaluate


def _setup_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_path = os.path.join(config.LOG_DIR, "training_run.log")
    logger = logging.getLogger("qgan")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="a")  # append -> survives resume
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def _partial_synth_path(cls):
    return os.path.join(config.OUTPUT_DIR, f"partial_synth_{cls}.csv")


def _partial_fidelity_path(cls):
    return os.path.join(config.OUTPUT_DIR, f"partial_fidelity_{cls}.json")


def _class_done_marker(cls):
    return os.path.join(config.OUTPUT_DIR, f"class_done_{cls}.marker")


def run_pipeline():
    logger = _setup_logging()
    t_start = time.time()
    logger.info("=" * 70)
    logger.info(f"PIPELINE START (or RESUME) — EPOCHS={config.EPOCHS}, "
                f"N_VARIATIONAL_LAYERS={config.N_VARIATIONAL_LAYERS}, "
                f"CHECKPOINT_EVERY={config.CHECKPOINT_EVERY}")
    logger.info("=" * 70)

    logger.info("Loading data...")
    df = data_prep.load_raw()
    feature_cols = data_prep.get_feature_columns(df)
    logger.info(f"Loaded {len(df)} rows, {len(feature_cols)} numeric features.")

    integer_cols = [c for c in feature_cols if pd.api.types.is_integer_dtype(df[c])]

    classes = data_prep.classes_to_augment(df)
    target_n = data_prep.target_count(df)
    logger.info(f"Classes to augment ({len(classes)}): {classes}")
    logger.info(f"Target sample count per malware class: {target_n}")

    by_class = data_prep.split_by_class(df, feature_cols)

    synthetic_rows = []
    fidelity_rows = []

    for cls in classes:
        X_real = by_class[cls]
        n_real = len(X_real)
        n_needed = max(0, target_n - n_real)
        logger.info(f"\n=== {cls}: n_real={n_real}, n_needed={n_needed} ===")
        if n_needed == 0:
            logger.info("  Already at/above target; skipping generation.")
            continue

        # ---- Resume shortcut: this class already fully finished in a prior run ----
        if os.path.exists(_class_done_marker(cls)) and os.path.exists(_partial_synth_path(cls)):
            logger.info(f"  [{cls}] found completed cache from a previous run -> loading, skipping retrain.")
            synth_df = pd.read_csv(_partial_synth_path(cls))
            synthetic_rows.append(synth_df)
            with open(_partial_fidelity_path(cls)) as f:
                fidelity_rows.append(json.load(f))
            continue

        rep = data_prep.ClassRepresentation()
        X_norm = rep.fit_transform(X_real)

        t0 = time.time()
        result = train_qgan.train_one_class_safe(X_norm, cls)
        logger.info(f"  Trained in {time.time()-t0:.1f}s")

        synth_norm = train_qgan.sample_synthetic(result["theta"], n_needed)
        synth_orig = rep.inverse_transform(synth_norm)

        # Post-process: these are all non-negative count/ratio features in MalMem.
        synth_orig = np.clip(synth_orig, 0, None)
        synth_df = pd.DataFrame(synth_orig, columns=feature_cols)
        for c in integer_cols:
            synth_df[c] = synth_df[c].round().astype("int64")
        synth_df[config.LABEL_COL] = cls
        synth_df[config.FAMILY_LABEL_COL] = f"Synthetic-{cls}"
        synth_df[config.BINARY_LABEL_COL] = "Malware"
        synth_df["is_synthetic"] = 1
        synthetic_rows.append(synth_df)

        # Fidelity: compare real class data vs the synthetic data actually generated
        eval_n = min(n_real, len(synth_orig))
        fid = evaluate.fidelity_report(X_real[:eval_n], synth_orig[:eval_n], feature_cols, cls)
        fidelity_rows.append(fid)

        # ---- Cache this class's finished output to disk so a later crash
        #      (e.g. during downstream classifier training) doesn't force a retrain ----
        synth_df.to_csv(_partial_synth_path(cls), index=False)
        with open(_partial_fidelity_path(cls), "w") as f:
            json.dump(fid, f)
        with open(_class_done_marker(cls), "w") as f:
            f.write(f"done at {time.time()}\n")
        logger.info(f"  [{cls}] cached finished synthetic output -> {_partial_synth_path(cls)}")

    logger.info("\nAssembling augmented dataset...")
    df_original = df.copy()
    df_original["is_synthetic"] = 0
    df_augmented = pd.concat([df_original] + synthetic_rows, ignore_index=True, sort=False)
    df_augmented.to_csv(config.AUGMENTED_CSV_PATH, index=False)
    logger.info(f"Saved augmented CSV -> {config.AUGMENTED_CSV_PATH} ({len(df_augmented)} rows)")

    fidelity_df = pd.DataFrame(fidelity_rows)
    fidelity_df.to_csv(config.FIDELITY_TABLE_PATH, index=False)
    logger.info(f"Saved fidelity metrics -> {config.FIDELITY_TABLE_PATH}")
    logger.info("\n" + fidelity_df.to_string(index=False))

    logger.info("\nRunning downstream classifier comparison (this trains 8 models)...")
    downstream_df = evaluate.downstream_comparison(df_original, df_augmented, feature_cols)
    downstream_df.to_csv(config.DOWNSTREAM_TABLE_PATH, index=False)
    logger.info(f"Saved downstream results -> {config.DOWNSTREAM_TABLE_PATH}")
    logger.info("\n" + downstream_df.to_string(index=False))

    logger.info(f"\nTotal pipeline time this run: {time.time()-t_start:.1f}s")
    logger.info("PIPELINE COMPLETE — writing DONE marker.")
    with open(os.path.join(config.OUTPUT_DIR, "PIPELINE_DONE.marker"), "w") as f:
        f.write("done\n")


def main():
    logger = logging.getLogger("qgan")
    try:
        run_pipeline()
    except Exception:
        # Make sure the crash reason is on disk even if this process is about
        # to be killed (OOM, timeout, etc.) — logging module already flushes
        # to the FileHandler synchronously on each call, so this is safe.
        tb = traceback.format_exc()
        try:
            logging.getLogger("qgan").error(f"PIPELINE CRASHED:\n{tb}")
        except Exception:
            pass
        with open(os.path.join(config.LOG_DIR, "last_crash.txt"), "w") as f:
            f.write(tb)
        print(f"PIPELINE CRASHED. See logs/last_crash.txt and logs/training_run.log for details.\n{tb}",
              file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
