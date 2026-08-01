"""
calibrate.py
Output-bias calibration for the QGAN generator.

Problem this fixes: moment_matching_loss (already in qgan_model.py) tries to
pull the generator's mean/std toward the real data's mean/std DURING training,
as a soft penalty competing against the adversarial loss. The stability
diagnosis showed the discriminator winning that competition throughout
training (d_loss falling, g_loss rising, growing oscillation) -- so the soft
constraint never fully bites and the generator's raw output distribution ends
up biased/mis-scaled relative to real data, independent of whether individual
samples look locally plausible.

Fix: a POST-HOC affine correction applied after generation, not during
training. This is deterministic and doesn't depend on adversarial convergence
at all -- it just forces the first and second moments of the synthetic output
to match the real data exactly, per PCA-angle-encoded dimension (the
[-1,1]^N_QUBITS space the generator operates in, same space real data is
mapped into via ClassRepresentation.fit_transform).

Two calibration modes:
  - "diagonal": per-dimension mean+std matching (bias + scale calibration,
    the technique named in the task). Cheap, always safe, doesn't require the
    generator to have learned correlations between dimensions.
  - "full": full-covariance whitening+recoloring (Cholesky-based). Stronger
    correction -- also fixes cross-dimension covariance structure -- but only
    helps if the generator already learned roughly the right correlation
    *shape* and just has the wrong scale/orientation; can't fix a generator
    that learned no structure at all. Reported for comparison, diagonal is
    the default/primary since it's what "output-bias calibration" refers to.

Calibration is fit ONCE per class from (real X_norm) vs (large raw synthetic
sample from the frozen final checkpoint) and then applied to every future
sample drawn from that checkpoint -- this is the "freeze + calibrate" step
that turns a checkpoint into a "frozen, calibrated QGAN".
"""
import numpy as np


def fit_calibration(X_real_norm: np.ndarray, X_fake_raw: np.ndarray) -> dict:
    """Fit both diagonal and full-covariance calibration params from real vs
    raw-synthetic samples in the shared [-1,1]^N_QUBITS normalized space."""
    mu_real = X_real_norm.mean(axis=0)
    sd_real = X_real_norm.std(axis=0) + 1e-8
    mu_fake = X_fake_raw.mean(axis=0)
    sd_fake = X_fake_raw.std(axis=0) + 1e-8

    cov_real = np.cov(X_real_norm, rowvar=False)
    cov_fake = np.cov(X_fake_raw, rowvar=False)
    d = cov_real.shape[0]
    cov_real_r = cov_real + 1e-6 * np.eye(d)
    cov_fake_r = cov_fake + 1e-6 * np.eye(d)
    L_real = np.linalg.cholesky(cov_real_r)
    L_fake = np.linalg.cholesky(cov_fake_r)
    L_fake_inv = np.linalg.inv(L_fake)
    transform_full = L_real @ L_fake_inv

    return {
        "mu_real": mu_real, "sd_real": sd_real,
        "mu_fake": mu_fake, "sd_fake": sd_fake,
        "transform_full": transform_full,
    }


def apply_calibration(X_fake_raw: np.ndarray, calib: dict, mode: str = "diagonal",
                       clip_bound: float = 1.0) -> np.ndarray:
    """Apply a fitted calibration to freshly-sampled raw generator output.
    Output is clipped back to [-1, 1]^N_QUBITS since that's the valid encoding
    range expected by ClassRepresentation.inverse_transform downstream."""
    if mode == "diagonal":
        z = (X_fake_raw - calib["mu_fake"]) / calib["sd_fake"]
        out = z * calib["sd_real"] + calib["mu_real"]
    elif mode == "full":
        out = (X_fake_raw - calib["mu_fake"]) @ calib["transform_full"].T + calib["mu_real"]
    else:
        raise ValueError(mode)
    return np.clip(out, -clip_bound, clip_bound)


def calibration_report(X_real_norm: np.ndarray, X_fake_raw: np.ndarray, calib: dict) -> dict:
    """Quick per-dimension moment-match diagnostic, before vs after calibration."""
    fake_diag = apply_calibration(X_fake_raw, calib, mode="diagonal")
    fake_full = apply_calibration(X_fake_raw, calib, mode="full")
    mu_r, sd_r = X_real_norm.mean(axis=0), X_real_norm.std(axis=0)

    def moment_gap(X):
        return {
            "mean_abs_diff": float(np.abs(X.mean(axis=0) - mu_r).mean()),
            "std_abs_diff": float(np.abs(X.std(axis=0) - sd_r).mean()),
        }

    return {
        "uncalibrated": moment_gap(X_fake_raw),
        "diagonal_calibrated": moment_gap(fake_diag),
        "full_calibrated": moment_gap(fake_full),
    }
