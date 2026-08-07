# Day 5a — KTA Alignment Subset-Size Sensitivity Sweep — FINAL REPORT

**Team B (Shanmukh) · QIntern 2026 · Mentors: Dr. Simranjit Singh, Dr. Mohit Sajwan**
**Config:** QSVM v3.1 (bug-fixed), XGBoost feature selection, q=8, seed=7, local kernel (g=2, non-overlapping pairs), C=1.0

This is the final version of the Day 5a deliverable and supersedes both
earlier passes in this repo (`Day5a_KTA_Subset_Sweep_Results.md` — original,
buggy; `Day5a_KTA_Subset_Sweep_Results_v3_1_CORRECTED.md` — bug-fixed sweep).
Both are kept for the audit trail. **This file adds a round of "can we do
better" diagnostics run after the corrected sweep**, at the team's request,
to check whether the flat/modest accuracy numbers could be improved with
more compute (landmarks, optimization budget, circuit depth) before
finalizing what goes in the paper.

**Bottom line up front: no. Three different attempts to push accuracy
higher (more Nyström landmarks, more KTA optimization iterations, a deeper
circuit) all made results flat or worse. 0.800 (CTGAN, n=10) stands as the
best legitimate number for this architecture — not 0.875, which came from
the now-fixed circuit bug.**

---

## Part 1 — Corrected 8-run sweep (unchanged from the previous pass)

| KTA subset n | Dataset | Accuracy | Macro-F1 | KTA (full test set) |
|---|---|---|---|---|
| 5  | CTGAN    | 0.800 | 0.803 | -0.2112 |
| 10 | CTGAN    | 0.800 | 0.801 | -0.2113 |
| 20 | CTGAN    | 0.775 | 0.769 | -0.2150 |
| 50 | CTGAN    | 0.800 | 0.795 | -0.2119 |
| 5  | Original | 0.450 | 0.440 | -0.2326 |
| 10 | Original | 0.450 | 0.449 | -0.2309 |
| 20 | Original | 0.425 | 0.421 | -0.2342 |
| 50 | Original | 0.425 | 0.419 | -0.2261 |

Both circuit and RNG-coupling bugs fixed and verified (see the CORRECTED
report for full detail). Accuracy is flat across n on both datasets, within
the ~0.06–0.08 standard-error band for a 40-point test set — genuinely
flat/robust, but not statistically distinguishable from noise at this
sample size.

---

## Part 2 — "Can we improve this?" diagnostic round

Three questions were tested, each targeting a different possible bottleneck.

### 2a. Is QSVM even competitive with classical, on the same features?

| | QSVM v3.1 | Classical RF (same 8 features) | Classical RF (full features) |
|---|---|---|---|
| CTGAN | 0.800 | **0.875** | **0.950** |
| Original | 0.425–0.450 | 0.425 | 0.400 |

**Finding:** on CTGAN, plain Random Forest beats QSVM even on the identical
8-feature input, and does much better with full features — there's no sign
of quantum advantage here, and a lot of the CTGAN signal is lost in the
8-qubit compression. On Original, classical performance is *also* low with
full features, suggesting the ~500-sample, 3-class family task is close to
a genuine data-difficulty ceiling regardless of method.

### 2b. Does more Nyström landmark coverage help? (CTGAN, n=10 fixed)

| n_landmarks | Accuracy | Wall time |
|---|---|---|
| 40 (baseline) | **0.800** | 158s |
| 80 | 0.750 | 431s |
| 150 | 0.775 | 1332s (~22 min) |

**Finding:** no. More landmarks made results *worse*, at up to 8x the
compute cost. Rules out "Nyström approximation too coarse" as the
bottleneck.

### 2c. Does more KTA optimization budget or circuit depth help? (CTGAN, n=10 fixed)

| Config | Accuracy | Macro-F1 | KTA (full test) | \|\|w_trained\|\| |
|---|---|---|---|---|
| align_iters=5 (baseline) | **0.800** | 0.801 | -0.2113 | 0.97 |
| align_iters=15 | 0.800 | 0.806 | -0.1925 | 2.97 |
| align_iters=30 | 0.750 | 0.750 | -0.1886 | 2.83 |
| L=3 (deeper circuit) | 0.775 | 0.777 | -0.1830 | 0.93 |

**Finding, and this is the most informative diagnostic of the three:** the
KTA *alignment score itself* keeps improving with more optimization budget
and depth (-0.211 → -0.193 → -0.189 → -0.183 — moving toward 0, i.e. genuinely
less anti-aligned). But downstream **accuracy does not follow it** — flat at
best, worse at iters=30. `||w_trained||` roughly triples from 0.97 to ~2.9-3.0
as iterations increase. This is the signature of **overfitting the 10-point
alignment subset**: COBYLA finds weights that align well with those specific
10 points but generalize worse to the actual 40-point test set. More
optimization makes the proxy metric look better while quietly hurting the
thing that matters.

### Verdict across all three

| Lever tried | Best result | vs. 0.800 baseline |
|---|---|---|
| More landmarks | 0.775 (n=150) | worse |
| More optimization iterations | 0.800 (tied, iters=15) | flat |
| Deeper circuit | 0.775 (L=3) | worse |

**Nothing beat the baseline.** Every direction that adds compute either did
nothing or actively hurt accuracy. That pattern — not "we haven't tried hard
enough," but "trying harder makes it worse" — is itself informative: it
points to the local pairwise-fidelity kernel design being the limiting
factor, not the amount of tuning or approximation quality behind it.

---

## Part 3 — What this means for the paper

- **Report 0.800 (n=10, CTGAN), not 0.875.** The 0.875 number came from a
  circuit that was mathematically unable to do what the methodology section
  describes (see Part 1 / the CORRECTED report for the full bug writeup).
  0.800 is the real, verified, best-effort number for this architecture.
- **Don't claim quantum advantage on CTGAN as currently configured** — classical
  RF beats QSVM on the identical feature set (0.875 vs 0.800) and by a wide
  margin with full features (0.950). If a quantum-advantage claim is
  important to the paper, it needs a different architectural angle (see
  below), not more tuning of this one.
- **The Original-dataset ceiling (~0.42–0.45) is not obviously a QSVM
  weakness** — classical baselines land in the same range on the same data,
  which is actually the more defensible framing: the paper's claim there
  can be about *comparable* performance to classical at this dataset, not
  about beating a classical ceiling that doesn't exist at this sample size.
- **KTA's overfitting-to-subset behavior (2c) is worth a sentence in the
  paper's limitations** — it's a real, somewhat interesting finding in its
  own right (a small-subset alignment objective can improve on its own
  metric while degrading generalization), and it's honest about why larger
  align_iters isn't a free win.

## Open item — not yet run

**Original-dataset sample-size scaling** (does accuracy improve with more
raw samples pulled from the much larger available pool — thousands per
family vs. the ~500 used here) was set up (`test_b_more_samples.py`, in the
zip) but not executed in this session. This is the one remaining lever that
hasn't been ruled out — worth running before concluding the Original-dataset
ceiling is a hard data-difficulty limit rather than a small-n effect.

---

## Recommended next steps, in priority order

1. Run the pending Original-dataset sample-size scaling test (`test_b_more_samples.py`).
2. Re-run `ablation.py` configs A–D under the fixed circuit + decoupled RNG
   (flagged in the CORRECTED report — same two bugs are present there,
   affecting the existing headline ablation table).
3. If a real accuracy improvement is still needed for the paper's claim,
   the next architectural lever to test is the feature/qubit budget itself
   (q=8 is a hard cap right now) or the projection method (`hybrid`
   LDA+PCA vs. `xgboost`) — not further tuning of the current local-kernel +
   KTA design, which this diagnostic round shows is close to its ceiling.
4. Multi-seed re-run before citing any specific number as stable, consistent
   with Week 5's finding that single-seed CTGAN numbers don't survive
   multi-seed correction.

---

## Compute notes

Corrected 8-run sweep: ~24 min. Diagnostic round (2a classical baselines,
2b landmark scaling, 2c optimization/depth scaling): ~40 min additional CPU
time. All on `lightning.qubit`, no GPU used at any point.
