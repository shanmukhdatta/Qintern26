# QGAN: 30 epochs (baseline) vs 100 epochs (this run)

Trained on 1 CPU core (sandbox), backprop-diff `default.qubit` simulator, same
architecture/hyperparameters as baseline — only EPOCHS changed (30 -> 100).
Total wall time this run: ~50 min training (2 classes x 100 epochs) + ~4 min eval.

## Fidelity metrics (lower = synthetic data closer to real distribution)

| Class | Metric | 30 epochs | 100 epochs | Change |
|---|---|---|---|---|
| Ransomware | KS_median | 0.3886 | 0.3275 | **-15.7% (better)** |
| Ransomware | Wasserstein_median | 1.0819 | 0.6643 | **-38.6% (better)** |
| Ransomware | Wasserstein_p75 | 7.1222 | 7.3678 | +3.4% (slightly worse, tail) |
| Ransomware | MMD | 0.1743 | 0.1628 | **-6.6% (better)** |
| Trojan | KS_median | 0.2889 | 0.2927 | +1.3% (flat/slightly worse) |
| Trojan | Wasserstein_median | 0.5679 | 0.8712 | +53.4% (worse) |
| Trojan | Wasserstein_p75 | 3.2582 | 2.6987 | -17.2% (better, tail) |
| Trojan | MMD | 0.3366 | 0.3422 | +1.7% (flat) |

## Downstream classifier impact (augmented vs original-only, F1-macro)

| Classifier | 30-epoch aug F1 | 100-epoch aug F1 | Original-only F1 |
|---|---|---|---|
| RandomForest | 0.8119 | 0.8121 | 0.8127 |
| XGBoost | 0.8203 | 0.8200 | 0.8211 |
| LightGBM | 0.8137 | 0.8108 | 0.8129 |
| SVM | 0.5994 | 0.6039 | 0.6089 |

## Honest read

- **Ransomware clearly benefited** from more epochs — Wasserstein distance dropped
  ~39%, KS ~16%. Loss curves show real (if slow) convergence through epoch 100
  (d_loss 1.35 -> 1.12 in the last 20 epochs, still trending down at cutoff).
- **Trojan did not clearly benefit** — d_loss/g_loss oscillated in a narrow band
  (~1.26-1.37 / 0.78-0.93) for the full 100 epochs with no real trend. Median
  Wasserstein actually got worse, though the tail (p75) improved. This class's
  GAN reached an adversarial equilibrium early and more epochs just moved it
  around within that equilibrium rather than improving it.
- **Downstream classifier accuracy/F1 barely moved either way** (deltas are in
  the 3rd decimal place, within run-to-run noise). This is expected: synthetic
  rows are 229 and 533 respectively, against a ~47K-row training split — too
  small a fraction to shift an ensemble classifier already at 87%+ accuracy.
  This was true at 30 epochs and is still true at 100; **epoch count was never
  going to move this number**, regardless of which malware class we're synthesizing.

## What this means for "more epochs"

Epochs are not a uniformly good lever. Diminishing/mixed returns kicked in for one
of two classes at the exact same epoch count. If you want a bigger jump in fidelity
and (eventually) downstream metrics, the next-highest-leverage changes per the
original README §7 are:
1. Increase `N_VARIATIONAL_LAYERS` (4 -> 6-8) for more generator expressivity —
   benchmarked at ~1.4x slower/epoch on this hardware, so budget for it.
2. Try the V2/V3 architecture ideas already validated elsewhere in QTagger+
   (data re-uploading, Nyström-style kernel tricks) rather than the vanilla VQC.
3. Increase the *proportion* of synthetic data used in downstream training
   (currently just filling the gap to class balance) if the goal is to actually
   move classifier metrics, not just improve standalone generator fidelity.
