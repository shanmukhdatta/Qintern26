"""
train_qgan.py
Adversarial training loop for one malware-family class at a time (each class
gets its own generator theta + discriminator, trained in that class's bounded
6-D PCA representation). Implements the Q-SYNTH-style stabilization recipe:
instance noise, label smoothing, feature matching, moment matching, grad clipping.
"""

import os
import pickle
import time
import logging
import numpy as np
from pennylane import numpy as pnp
import autograd

import config
import qgan_model as qm
from optim import AdamOptimizer

logger = logging.getLogger("qgan")


def _dropout_masks(rng, batch_size):
    m1 = rng.binomial(1, 1 - config.DISC_DROPOUT_P, size=(batch_size, config.DISC_HIDDEN_1)).astype(float)
    m2 = rng.binomial(1, 1 - config.DISC_DROPOUT_P, size=(batch_size, config.DISC_HIDDEN_2)).astype(float)
    return m1, m2


def _ckpt_path(class_name: str) -> str:
    return os.path.join(config.CHECKPOINT_DIR, f"{class_name}_ckpt.pkl")


def _save_checkpoint(class_name, epoch, theta, disc, opt_g, opt_d, history, rng_state):
    """Atomic checkpoint write (write to tmp then rename) so a crash mid-write
    never leaves a corrupt checkpoint that resume would choke on."""
    path = _ckpt_path(class_name)
    tmp_path = path + ".tmp"
    payload = {
        "epoch": epoch,               # last COMPLETED epoch index (0-based)
        "theta": np.asarray(theta),
        "disc": {k: np.asarray(v) for k, v in disc.items()},
        "opt_g_state": {"t": opt_g.t, "m": opt_g.m, "v": opt_g.v},
        "opt_d_state": {"t": opt_d.t, "m": opt_d.m, "v": opt_d.v},
        "history": history,
        "rng_state": rng_state,
        "config_epochs": config.EPOCHS,   # so resume can detect config changes
    }
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp_path, path)  # atomic on POSIX
    logger.info(f"  [{class_name}] checkpoint saved @ epoch {epoch+1}/{config.EPOCHS} -> {path}")


def _load_checkpoint(class_name: str):
    path = _ckpt_path(class_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload
    except Exception as e:
        logger.warning(f"  [{class_name}] checkpoint at {path} is corrupt/unreadable ({e}); ignoring and restarting.")
        return None


def train_one_class(X_real_norm: np.ndarray, class_name: str, verbose: bool = True) -> dict:
    rng = np.random.RandomState(config.RANDOM_STATE)
    n = X_real_norm.shape[0]
    batch_size = min(config.BATCH_SIZE, n)

    theta = qm.init_generator_params()
    disc = qm.init_discriminator_params()
    opt_g = AdamOptimizer(lr=config.LR_GEN)
    opt_d = AdamOptimizer(lr=config.LR_DISC)

    history = {"d_loss": [], "g_loss": []}
    n_batches = max(1, n // batch_size)
    start_epoch = 0
    t0 = time.time()

    # ---- Resume from checkpoint if one exists for this class ----
    ckpt = _load_checkpoint(class_name)
    if ckpt is not None:
        if ckpt.get("config_epochs") != config.EPOCHS:
            logger.info(f"  [{class_name}] checkpoint was saved with EPOCHS={ckpt.get('config_epochs')}, "
                        f"current config EPOCHS={config.EPOCHS}. Resuming anyway from epoch {ckpt['epoch']+1}.")
        theta = pnp.array(ckpt["theta"], requires_grad=True)
        disc = {k: pnp.array(v, requires_grad=True) for k, v in ckpt["disc"].items()}
        opt_g.t = ckpt["opt_g_state"]["t"]
        opt_g.m = ckpt["opt_g_state"]["m"]
        opt_g.v = ckpt["opt_g_state"]["v"]
        opt_d.t = ckpt["opt_d_state"]["t"]
        opt_d.m = ckpt["opt_d_state"]["m"]
        opt_d.v = ckpt["opt_d_state"]["v"]
        history = ckpt["history"]
        rng.set_state(ckpt["rng_state"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"  [{class_name}] RESUMED from checkpoint: starting at epoch {start_epoch+1}/{config.EPOCHS} "
                    f"({len(history['d_loss'])} batch-steps already logged)")
        if start_epoch >= config.EPOCHS:
            logger.info(f"  [{class_name}] checkpoint already reached target EPOCHS; nothing to do.")
            return {"theta": theta, "disc": disc, "history": history}

    for epoch in range(start_epoch, config.EPOCHS):
        perm = rng.permutation(n)
        # instance-noise sigma anneals down over training (Q-SYNTH-style schedule)
        sigma = config.INSTANCE_NOISE_BASE * (1 - epoch / config.EPOCHS)

        if getattr(config, "LR_DECAY", False):
            decay = config.LR_MIN_FRAC + (1 - config.LR_MIN_FRAC) * 0.5 * (1 + np.cos(np.pi * epoch / config.EPOCHS))
            opt_g.lr = config.LR_GEN * decay
            opt_d.lr = config.LR_DISC * decay

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if len(idx) == 0:
                continue
            x_real = pnp.array(X_real_norm[idx])
            m1, m2 = _dropout_masks(rng, len(idx))

            # ---- Discriminator step (fake batch detached: no quantum backprop here) ----
            z_d = qm.sample_latent(len(idx))
            fake_detached = pnp.array(np.asarray(qm.generate_batch(z_d, theta)), requires_grad=False)
            x_real_noisy = qm.add_instance_noise(x_real, sigma)
            x_fake_noisy = qm.add_instance_noise(fake_detached, sigma)

            def cost_d(disc_params):
                p_real, _ = qm.discriminator_forward(x_real_noisy, disc_params, m1, m2)
                p_fake, _ = qm.discriminator_forward(x_fake_noisy, disc_params, m1, m2)
                return qm.bce(p_real, config.LABEL_SMOOTHING_GAMMA) + qm.bce(p_fake, 0.0)

            d_grads = autograd.grad(cost_d)(disc)
            d_grads = qm.clip_grad_(d_grads)
            disc = opt_d.step(disc, d_grads)
            d_loss_val = cost_d(disc)

            # ---- Generator step (backprop through the quantum circuit) ----
            z_g = qm.sample_latent(len(idx))

            def cost_g(theta_):
                fake = qm.generate_batch(z_g, theta_)
                fake_noisy = qm.add_instance_noise(fake, sigma)
                p_fake, feat_fake = qm.discriminator_forward(fake_noisy, disc, None, None)
                _, feat_real = qm.discriminator_forward(x_real_noisy, disc, None, None)
                adv = qm.bce(p_fake, config.LABEL_SMOOTHING_GAMMA)
                fm = qm.feature_matching_loss(feat_real, feat_fake)
                mm = qm.moment_matching_loss(x_real, fake)
                return adv + fm + mm

            g_grads = autograd.grad(cost_g)(theta)
            g_grads = qm.clip_grad_(g_grads)
            theta = opt_g.step(theta, g_grads)
            g_loss_val = cost_g(theta)

            history["d_loss"].append(float(d_loss_val))
            history["g_loss"].append(float(g_loss_val))

        if verbose and (epoch + 1) % config.EVAL_EVERY_N_EPOCHS == 0:
            msg = (f"  [{class_name}] epoch {epoch+1}/{config.EPOCHS} "
                   f"d_loss={history['d_loss'][-1]:.4f} g_loss={history['g_loss'][-1]:.4f} "
                   f"({time.time()-t0:.1f}s elapsed)")
            print(msg, flush=True)
            logger.info(msg)

        if (epoch + 1) % config.CHECKPOINT_EVERY == 0 or (epoch + 1) == config.EPOCHS:
            try:
                _save_checkpoint(class_name, epoch, theta, disc, opt_g, opt_d, history, rng.get_state())
            except Exception as e:
                # Never let a checkpoint-write failure kill the training run itself.
                logger.warning(f"  [{class_name}] checkpoint save failed at epoch {epoch+1}: {e}")

    return {"theta": theta, "disc": disc, "history": history}


def train_one_class_safe(X_real_norm: np.ndarray, class_name: str, verbose: bool = True) -> dict:
    """Wrapper that catches any exception mid-training, makes sure the last good
    checkpoint is on disk, logs the crash with a full traceback, and re-raises
    so the caller (main.py) can decide how to proceed. Because train_one_class
    checkpoints every CHECKPOINT_EVERY epochs, simply re-running the pipeline
    after a crash resumes from the last checkpoint instead of restarting."""
    import traceback
    try:
        return train_one_class(X_real_norm, class_name, verbose=verbose)
    except Exception:
        logger.error(f"  [{class_name}] TRAINING CRASHED:\n{traceback.format_exc()}")
        logger.error(f"  [{class_name}] Last checkpoint on disk (if any) is intact. "
                      f"Re-run the pipeline to resume from there.")
        raise


def sample_synthetic(theta, n_samples: int) -> np.ndarray:
    """Draw n_samples synthetic points (in the bounded [-1,1]^N_QUBITS space) from a trained generator."""
    out = []
    remaining = n_samples
    max_chunk = 512
    while remaining > 0:
        chunk = min(max_chunk, remaining)
        z = qm.sample_latent(chunk)
        fake = np.asarray(qm.generate_batch(z, theta))
        out.append(fake)
        remaining -= chunk
    return np.concatenate(out, axis=0)
