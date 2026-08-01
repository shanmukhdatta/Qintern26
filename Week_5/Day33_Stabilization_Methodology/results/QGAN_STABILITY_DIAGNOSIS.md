# Why the QGAN isn't stabilizing — diagnosis

Based on the full per-batch loss history saved in checkpoints (7,400-7,800
batch-steps per class, not just the epoch-10 printouts), all three classes
show the **same failure pattern**:

## The evidence

| Class | d_loss first-10-steps → last-10-steps | g_loss first-10 → last-10 | oscillation (2nd half vs 1st half) |
|---|---|---|---|
| Ransomware | 1.356 → 1.136 (↓) | 0.837 → 1.140 (↑) | step-to-step noise ~1.8x higher |
| Trojan | 1.357 → 1.272 (↓) | 0.827 → 0.929 (↑) | step-to-step noise ~1.9x higher |
| Spyware | 1.348 → 1.171 (↓) | 0.825 → 1.032 (↑) | step-to-step noise ~2.7x higher |

See `../plots/plots_loss_curves.png` for the visual — all three show discriminator
loss trending down while generator loss trends up, with the gap widening and
getting noisier as training progresses, not converging to a stable equilibrium.

## Root cause: capacity mismatch, not a learning-rate imbalance

Checked the obvious suspect first — LR imbalance — and ruled it out:
- `LR_GEN = 7e-3` is actually **3.5x higher** than `LR_DISC = 2e-3`
- Discriminator and generator get exactly one update each per batch (1:1 ratio,
  confirmed in `../code/train_qgan.py`)

So the generator has a nominal training-speed advantage and isn't being
out-trained by update frequency. The real issue is **representational capacity**:

- **Generator**: 4-layer variational circuit on 6 qubits (`N_VARIATIONAL_LAYERS=4`)
  — a fairly shallow quantum circuit expected to reproduce a full 6-dimensional
  PCA-projected data distribution.
- **Discriminator**: classical MLP with 16+8 hidden units (`DISC_HIDDEN_1=16`,
  `DISC_HIDDEN_2=8`) — small by classical standards, but still a universal
  function approximator with far more effective degrees of freedom than a
  4-layer/6-qubit circuit for a binary classification sub-task (real vs. fake),
  which is a fundamentally easier problem than *generating* a full distribution.

This is the standard GAN pathology of an under-powered generator facing an
adversary that's structurally easier to optimize — quantum-specific because the
generator's expressivity is bottlenecked by circuit depth/qubit count rather
than parameter count, and shallow variational circuits are known to have
limited expressivity ("barren plateau"-adjacent capacity limits) compared to
even small classical MLPs.

## Why more epochs alone gave mixed results (matches what we saw earlier)

- **Ransomware & Spyware**: still trending in the right direction at epoch 100
  (d_loss/g_loss gap still narrowing on net, if noisily) — more epochs would
  likely help further, though the growing oscillation suggests diminishing
  returns without also fixing the underlying imbalance.
- **Trojan**: smallest net movement of the three, consistent with hitting a
  noisy plateau earlier than the other two classes.

## What would actually fix it (in priority order)

1. **Increase generator capacity**: bump `N_VARIATIONAL_LAYERS` from 4 to 6-8
   (benchmarked earlier at ~1.4x slower/epoch on this hardware — a real but
   affordable cost). This is the most direct fix for the capacity mismatch.
2. **Weaken/regularize the discriminator**: reduce hidden units, raise
   `DISC_DROPOUT_P` above 0.10, or add label smoothing / one-sided noise on
   real labels — makes the adversary's task harder, closer to matching the
   generator's difficulty.
3. **Add a learning-rate schedule**: decay both LRs over training (e.g. cosine
   or step decay) — directly targets the growing late-training oscillation
   seen in all three loss curves, independent of the capacity fix.
4. **Try TTUR (two-timescale update rule) tuning**: current 3.5x gen:disc LR
   ratio was clearly insufficient to compensate for the capacity gap; an even
   larger gap, combined with fix #1, may be needed.
5. Track a distribution-level metric (e.g. Wasserstein distance mid-training,
   already computed post-hoc in `../code/evaluate.py`) every N epochs during training
   itself — right now stability is only visible in the adversarial loss, which
   is known to be a poor proxy for actual sample quality/coverage. This would
   let you early-stop at the genuinely-best checkpoint instead of just the
   last one.
