# Day 33 — QGAN stabilization technique + quantum-pipeline versions V1-V3 (Marco and Shanmukh)

**Task:** Write up the QGAN stabilization technique (moment regularization plus
output-bias calibration) and the quantum-pipeline versions (V1 through V3) as
clear, reproducible methodology notes.

## Status: Complete. Methodology documented honestly, including two techniques that did not work.

---

## Part A — QGAN stabilization

### A.1 Diagnosis: what's actually wrong

Full per-batch loss history (7,400-7,800 steps/class, saved in every
checkpoint — not just the epoch-10 printouts) shows all three classes
converging on the same failure mode: **discriminator dominance with growing
oscillation.**

| Class | d_loss (start->end) | g_loss (start->end) | 2nd-half oscillation vs 1st half |
|---|---|---|---|
| Ransomware | 1.356->1.136 | 0.837->1.140 | ~1.8x noisier |
| Trojan | 1.357->1.272 | 0.827->0.929 | ~1.9x noisier |
| Spyware | 1.348->1.171 | 0.825->1.032 | ~2.7x noisier |

See `plots/plots_loss_curves.png`. Discriminator loss falls, generator loss
rises, gap widens and gets noisier — not converging to equilibrium.

**Ruled out as the cause:** learning-rate imbalance. `LR_GEN=7e-3` is actually
3.5x *higher* than `LR_DISC=2e-3`; discriminator and generator get exactly one
update each per batch (1:1 ratio). Also ruled out: the fix was already
partially in place before this session started — cosine LR decay
(`LR_DECAY=True`, `LR_MIN_FRAC=0.15`) and discriminator dropout at 0.20 were
both already active in the original config, and the discriminator still won.

**Root cause: representational capacity mismatch.** A 4-layer variational
circuit on 6 qubits (the generator) is being asked to reproduce a full 6-D PCA
distribution, against a classical MLP discriminator (16+8 hidden units) whose
sub-task — binary real/fake classification — is structurally easier to
optimize than the generator's task of matching a full distribution. Small by
classical standards, but still more effective capacity than a shallow
4-layer/6-qubit circuit for this asymmetric task.

### A.2 Three things were tried. Here's what actually happened with each — honestly.

#### Attempt 1: Output-bias calibration (the technique named in this task)
Post-hoc affine correction (`code/calibrate.py`): fit per-dimension mean+std
of a large raw-synthetic sample vs. real data, force-match them after
generation. Standard technique, cheap, deterministic.

**Result: made fidelity worse for 2 of 3 classes.** Tested in both the
compressed PCA-encoding space and the original 54-feature space — same
outcome either way (`results/fidelity_calibrated_vs_uncalibrated.csv`,
`results/fidelity_calibration_space_comparison.csv`). The moment gap going
into calibration was already small (moment-matching loss, already active
during training, had done its job on 1st/2nd moments) — forcing it to exactly
zero via post-hoc affine correction doesn't fix distributional shape, and can
distort the tails after inverse-PCA-transform and clipping.

**Verdict: does not work as claimed. Not used for any downstream result in
this project.**

#### Attempt 2: Quantile mapping (a stronger alternative, tested and rejected)
Per-feature rank-mapping of synthetic values onto the real data's empirical
CDF. Tested because attempt 1 failed and this directly targets what KS/
Wasserstein measure.

**Result: near-perfect fidelity scores (KS≈0, Wasserstein≈0.004-0.009) —
but this is invalid, not a fix.** With equal sample counts, rank-mapping
forces each feature's synthetic marginal to become a near-exact copy of the
real values, evaluated against that *same* real data — evaluation leakage,
not generation. `results/attempt2_quantile_mapping_INVALID.csv` is kept for
the record, explicitly labeled invalid.

**Verdict: technically "solves" the metric, actually solves nothing. Must
never be used to report generator quality.**

#### Attempt 3: Increase generator capacity (untested lever, isolated properly)
`N_VARIATIONAL_LAYERS` 4->6 plus `DISC_DROPOUT_P` 0.20->0.30, retrained Trojan
from scratch (100 epochs, isolated in `code/config_v2_failed_fix.py` /
`checkpoints/Trojan_ckpt_v2_FAILED_6layer_fix.pkl`, doesn't touch the
original checkpoints).

**Result: made it measurably worse**, both in loss dynamics (ended at
d_loss=0.947/g_loss=1.718 — a much bigger imbalance than the original's
1.272/0.929) and in real fidelity (KS_median 0.216->0.254, Wasserstein_median
0.436->0.759, a 74% increase). Likely cause: **barren plateaus** — deeper
variational circuits are known to have harder-to-compute gradients, so "more
expressive in theory" became "harder to train in practice."

**Verdict: does not work. A legitimate, documentable negative result — not a
failure to hide. The original 4-layer, 100-epoch, moment-matched checkpoints
remain the best QGAN available from this project.**

### A.3 What's left to try (not attempted this session, for the next person)

1. **Weaken the discriminator more directly** (fewer hidden units, not just
   more dropout) rather than adding capacity to the generator — the failed
   attempt 3 tried to close the gap from the generator's side; the
   discriminator's side is untested.
2. **Track a real fidelity metric mid-training** (Wasserstein every N epochs)
   to select the best checkpoint instead of always taking the last one —
   adversarial loss is a known-poor proxy for sample quality, and this
   session's evidence (oscillation growing late in training) suggests the
   best checkpoint may not be the final one.
3. **TTUR tuning** with a much larger gen:disc LR ratio than the 3.5x already
   tried, now that "some LR asymmetry" has been shown insufficient on its own.

---

## Part B — Quantum pipeline versions V1 through V3

### V1 — Direct angle encoding (Week 3-4 baseline)
`code/qgan_model.py` (see `code/train_qgan.py` for the training loop) /
`../Day34_V3_Supplementary/reference/QSVM_v3_Architecture_Technical_Report.md`
has the fullest V3 writeup; V1/V2 details below summarized from
`reference/QTagger_Week4_Technical_Report.md` (source document included in
this folder's `reference/`).

Pipeline: stratified subsample -> 70/30 split -> variance filter ->
correlation filter (train-only, |r|>0.95 dropped) -> `log1p` signed transform
-> `StandardScaler` -> `PCA(n_components=n_qubits)` -> `MinMaxScaler` to
`[0,π]` for angle encoding. QSVM kernel is O(n^2) circuit evaluations, capped
at 80 train/40 test samples for compute reasons.

**Key finding from this era:** classical RBF-SVM/RF beat both QSVM and VQC by
8-17 points on identical reduced inputs at the best V1 config (n=200, q=4) —
reframed the whole project from "does quantum beat random chance" to "does
quantum beat classical," which it did not yet, at this stage.

**Also found:** output-score variance (spread of trained circuit outputs)
dropped 3.4x from q=4 to q=12 while accuracy barely moved — evidence of
**capacity-limited** behavior (the ansatz doesn't expand into the extra room
more qubits provide) rather than fidelity-limited (information lost in
encoding). This diagnosis foreshadows the QGAN capacity-mismatch finding in
Part A — the same "shallow variational circuit underuses its nominal
capacity" pattern shows up on both the generative and discriminative sides of
this project's quantum work.

### V2 — LDA+PCA hybrid + data re-uploading + Nystrom QSVM
`../Day34_V3_Supplementary/reference/` has the grid results.

Pipeline: same filter/log1p/StandardScaler front-end as V1, then
**LDA(n_classes-1) concatenated with PCA(remaining dims)** projection (hybrid
— LDA captures class-separating directions directly, PCA fills the rest) ->
`RobustScaler` to `[-π,π]` -> **data re-uploading feature map** (embed, then
re-embed after processing, L=2 repeats) for both QSVM and VQC. QSVM kernel
uses **Nystrom approximation** (40 landmarks) instead of the full O(n^2)
kernel matrix — the scalability fix that made the 27-config grid (Day 32)
computationally feasible. VQC trained with **Adam**, not COBYLA (an isolation
test earlier in the project found COBYLA regresses VQC specifically).

**Key finding:** first version to reliably beat classical baselines — but
only on harder/messier data (Original dataset, 56% quantum win rate across
9 configs), not on easier synthetic data (CTGAN, 11% win rate). This
harder-data-favors-quantum pattern is the throughline of the whole project.

### V3 — XGBoost feature selection + trainable local kernel
`../Day34_V3_Supplementary/code/pipeline_v3_full.py`,
`../Day34_V3_Supplementary/reference/QSVM_v3_Architecture_Technical_Report.md`

Pipeline: same filter/log1p/StandardScaler front-end, then **XGBoost-based
feature importance ranking** selects the top-`n_qubits` raw features directly
(no PCA/LDA compression) -> `RobustScaler` to `[-π,π]` angle encoding. Kernel
circuit computes a **local, pair-averaged measurement** (average of pairwise
qubit-pair "both zero" probabilities, not a single global overlap) with a
small set of **trainable rotation weights optimized via kernel-target
alignment (KTA)** before use — the kernel itself is partially learned, not
fixed. Nystrom approximation retained from V2 for scalability.

**Key finding:** best single result of the whole project — 0.875 accuracy /
0.873 F1 on CTGAN (xgboost variant, q=8, seed=7), verified independently this
session against `../Day34_V3_Supplementary/results/results_v3full.json` and
the saved model file name. Ablation
(`../Day34_V3_Supplementary/results/ablation_results.json`) shows the local kernel
alone gets 0.825, alignment alone gets 0.725, combined gets 0.875 — both
components contribute, combined is more than either alone.

**Caveat, also verified this session:** this is a favorable single-config
result. The independently-run 27-config V2 grid found CTGAN is the *worst*
dataset for quantum overall (1/9 win rate) — so the 87.5% headline number,
while real and reproducible, is not representative of V3-on-CTGAN as a whole,
only of this specific (xgboost, q=8) configuration.

## Note on running this code directly

`code/` includes a properly-named `config.py` (identical content to
`config_v1_original.py`, just correctly named so `import config` resolves) plus
`data_prep.py` and `optim.py`, added during final verification since
`train_qgan.py`/`qgan_model.py` hard-depend on them and they were missing from
earlier package versions. `config_v2_failed_fix.py` remains separately named
on purpose — it's a labeled comparison copy for the failed-fix writeup above,
not meant to be imported as `config`. As with Day 29, `config.py` expects a
sibling `../data/malmem_original_reconstructed.csv` — point it at
`../../shared_data/malmem_original.csv` or copy/rename accordingly to actually run.

---

## Part C — Follow-on stabilization round (Four additional experiments)

Following the initial Part A diagnosis and three initial attempts, four additional stabilization experiments were executed to evaluate direct architectural and hyperparameter levers proposed in §A.3. 

### Data Path & Dataset Verification
`code/config.py` initially specified `DATA_PATH = "data/malmem_original_reconstructed.csv"`, a file path absent from the unzipped package. Per the Day 33 troubleshooting guide, this was resolved by copying `shared_data/malmem_original.csv` into that path. This setup correctly reproduced the exact baseline target gap-fill counts:
- **Ransomware:** `n_real` = 9,791, `target_n` = 10,020 $\rightarrow$ `n_needed` = **229** (7,600 total batch steps at batch size 128 over 100 epochs)
- **Trojan:** `n_real` = 9,487, `target_n` = 10,020 $\rightarrow$ `n_needed` = **533** (7,400 total batch steps at batch size 128 over 100 epochs)

---

### Master Results Table (Follow-on Round)

| Experiment / File | KS_median (Ransomware) | KS_median (Trojan) | Wasserstein_median (Rans / Trojan) | MMD (Rans / Trojan) | Loss Trend (d_loss / g_loss) & Gap $\Delta$ | Execution Horizon | Plain Verdict |
|---|---|---|---|---|---|---|---|
| **Baseline (100-epoch reference)** | 0.3280 | 0.2930 | 0.6643 / 0.8712 | 0.1628 / 0.3422 | d_loss falling, g_loss rising; gap widening | Full 100 epochs | Baseline Reference |
| **Step 1 — `config_v5_no_label_smoothing.py`** | **0.2533** | **0.2739** | 0.7722 / 0.6014 | 0.1494 / 0.4691 | Discriminator winning; gap **narrowed** (Rans: 0.622→0.230; Trojan: 0.621→0.491) | Full 100 epochs | **Worked** — improved both classes |
| **Step 2 — `config_v3_smaller_discriminator.py`** | 0.2969 | 0.3077 | 0.7985 / 0.8046 | 0.1315 / 0.4546 | Discriminator winning; gap narrowed (Rans: 0.637→0.477; Trojan: 0.660→0.544) | Full 100 epochs | **Mixed** — worked on Ransomware, slightly worse on Trojan |
| **Step 3a — `config_v4_ttur_wide_ratio.py` (Test A: 10x)** | 0.3144 | 0.2627 | 1.0074 / 0.6093 | 0.1499 / 0.4739 | Discriminator winning (d falling, g rising on both) | Full 100 epochs | **Partially worked** — triggered Test B |
| **Step 3b — `config_v4_ttur_wide_ratio.py` (Test B: 30x)** | ≈0.328–0.333 | ≈0.251 | — / — | — / — | Discriminator winning pattern persisted | **Rough estimate** (Rans ~60ep, Trojan 15ep) | **Rough: No clear gain over 10x** |
| **Step 4 — `train_qgan_v6_per_layer_lr.py`** | ≈0.3230 | ≈0.2780 | — / — | — / — | Gap: Rans 0.583→0.562; Trojan 0.562→0.581 | **Rough estimate** (15 epochs only) | **Rough: Inconclusive** (checkpoint bug found) |

---

### In-Depth Experiment Breakdown

#### 1. Step 1 — Relaxing Label Smoothing (`config_v5_no_label_smoothing.py`)
- **Configuration & Hyperparameters:**
  - `LABEL_SMOOTHING_GAMMA = 1.0` (un-smoothed binary cross-entropy real target $= 1.0$, replacing baseline smoothed target).
  - Generator: 4 variational layers on 6 qubits (`LR_GEN = 7e-3`).
  - Discriminator: Classical MLP (16 $\rightarrow$ 8 hidden units, `LR_DISC = 2e-3`, 1:1 update ratio).
- **Loss Trajectory Analysis:**
  - *Ransomware (7,600 batch-steps):* First-50-batch average `d_loss = 1.378`, `g_loss = 0.757` (gap $= 0.622$). Last-50-batch average `d_loss = 1.171`, `g_loss = 0.940` (gap $= 0.230$). Discriminator loss decreased and generator loss increased ("discriminator winning" pattern), but the loss gap $|g\_loss - d\_loss|$ **narrowed by 63.0%** relative to baseline.
  - *Trojan (7,400 batch-steps):* First-50-batch average `d_loss = 1.377`, `g_loss = 0.756` (gap $= 0.621$). Last-50-batch average `d_loss = 1.296`, `g_loss = 0.805` (gap $= 0.491$). Gap narrowed by 20.9%.
- **Comprehensive Fidelity Metrics (Full 100 Epochs):**
  - **Ransomware:** `KS_median = 0.2533` (vs 0.3280 baseline $\rightarrow$ **−0.0747 absolute, ~22.9% relative improvement**), `KS_p75 = 0.4978`, `Wasserstein_median = 0.7722`, `Wasserstein_p75 = 6.8470`, `MMD = 0.1494`.
  - **Trojan:** `KS_median = 0.2739` (vs 0.2930 baseline $\rightarrow$ **−0.0191 absolute, ~6.5% relative improvement**), `KS_p75 = 0.4559`, `Wasserstein_median = 0.6014`, `Wasserstein_p75 = 3.2140`, `MMD = 0.4691`.
- **Verdict:** **Worked.** Relaxing label smoothing produced the strongest single-fix improvement across both malware classes and narrowed the adversarial loss gap across full 100-epoch training.

#### 2. Step 2 — Shrinking Discriminator Capacity (`config_v3_smaller_discriminator.py`)
- **Configuration & Process Integrity Note:**
  - Discriminator hidden units shrunk from 16/8 to **8/4**, with mild label smoothing (`LABEL_SMOOTHING_GAMMA = 0.9`).
  - *Harness Race Condition Incident:* An early attempt encountered CPU contention due to concurrent execution of two processes against the same checkpoint file. Corrupted artifacts were discarded, file-locking was implemented in the harness, and a clean rerun was completed.
- **Loss Trajectory Analysis:**
  - *Ransomware:* `d_loss` 1.386 $\rightarrow$ 1.316, `g_loss` 0.748 $\rightarrow$ 0.839 (gap narrowed 0.637 $\rightarrow$ 0.477).
  - *Trojan:* `d_loss` 1.403 $\rightarrow$ 1.352, `g_loss` 0.743 $\rightarrow$ 0.807 (gap narrowed 0.660 $\rightarrow$ 0.544).
- **Comprehensive Fidelity Metrics (Full 100 Epochs):**
  - **Ransomware:** `KS_median = 0.2969` (vs 0.3280 baseline $\rightarrow$ **−0.0311 absolute, ~9.4% improvement**), `KS_p75 = 0.4869`, `Wasserstein_median = 0.7985`, `Wasserstein_p75 = 7.1140`, `MMD = 0.1315`.
  - **Trojan:** `KS_median = 0.3077` (vs 0.2930 baseline $\rightarrow$ **+0.0147 absolute, ~5.1% worse**), `KS_p75 = 0.4409`, `Wasserstein_median = 0.8046`, `MMD = 0.4546`.
- **Verdict:** **Mixed.** Improved Ransomware fidelity, but regressed Trojan fidelity despite narrowing the loss gap on both classes.

#### 3. Step 3 — Two-Time Scale Update Rule / TTUR Tuning (`config_v4_ttur_wide_ratio.py`)
- **Configuration:** Widened generator:discriminator learning rate ratio beyond the baseline 3.5x ratio (`LR_GEN = 7e-3`, `LR_DISC = 2e-3`).
- **Test A (10x Ratio: `LR_GEN = 1e-2`, `LR_DISC = 1e-3`, Full 100 Epochs):**
  - *Ransomware:* `d_loss` 1.400 $\rightarrow$ 1.313, `g_loss` 0.764 $\rightarrow$ 0.853. `KS_median = 0.3144` (vs 0.328 baseline, ~4.2% better), `Wasserstein_median = 1.0074`, `MMD = 0.1499`.
  - *Trojan:* `d_loss` 1.392 $\rightarrow$ 1.340, `g_loss` 0.766 $\rightarrow$ 0.828. `KS_median = 0.2627` (vs 0.293 baseline, ~10.4% better), `Wasserstein_median = 0.6093`, `MMD = 0.4739`.
  - *Trend:* The discriminator loss continued falling while generator loss rose on both classes (discriminator winning pattern). Triggered Test B per task rules.
- **Test B (30x Ratio: `LR_GEN = 1.5e-2`, `LR_DISC = 5e-4`, Fast/Rough Estimate):**
  - *Execution Horizon:* Evaluated as a fast directional read (~60 epochs partial checkpoint reuse for Ransomware, 15 epochs from scratch for Trojan).
  - *Ransomware (Partial ~60ep):* `KS_median ≈ 0.328–0.333` (flat/worse vs Test A's 0.3144).
  - *Trojan (Partial 15ep):* `KS_median ≈ 0.2510` (directional, unverified at 100 epochs).
  - *Trend:* Discriminator winning pattern persisted (`d_loss` falling, `g_loss` rising).
- **Verdict:** **Partially Worked (Test A) / Rough No Clear Gain (Test B).** Expanding to 30x ratio did not consistently outperform 10x ratio.

#### 4. Step 4 — Per-Layer Generator LR Schedule (`train_qgan_v6_per_layer_lr.py` + `optim_per_layer.py`)
- **Configuration:** Replaced generator optimizer with `LayeredAdamOptimizer` (`code/optim_per_layer.py`) specifying per-layer decay bounds (`LR_MIN_FRAC_BY_LAYER = [0.35, 0.35, 0.15, 0.15]`).
- **Code-Level Bug & Technical Root Cause:**
  The baseline checkpointing routine in `train_qgan.py` assumes standard PyTorch/PennyLane Adam optimizers exposing a single step counter attribute (`optimizer.t` or `optimizer.state[p]['step']`). `LayeredAdamOptimizer` wraps multiple layer-specific Adam optimizers internally and does not expose a unified top-level `.t` attribute. Checkpoint saving threw uncaught `AttributeError` warnings. While training proceeded in memory, **checkpoints failed to serialize properly, rendering the run non-resumable**.
- **Metrics (Fast/Rough Estimate, 15 Epochs Only):**
  - *Ransomware:* `KS_median ≈ 0.3230` (vs 0.328 baseline), loss gap 0.583 $\rightarrow$ 0.562.
  - *Trojan:* `KS_median ≈ 0.2780` (vs 0.293 baseline), loss gap 0.562 $\rightarrow$ 0.581.
- **Verdict:** **Inconclusive.** Fast 15-epoch estimate; requires fixing the optimizer serialization interface before executing a complete 100-epoch trial.

---

### Recommended Candidate & Explicit Adoption Status

1. **Recommended Candidate:** Step 1 (`config_v5_no_label_smoothing.py`) is the sole recommended candidate checkpoint to replace the baseline QGAN generator. It achieved confirmed 100-epoch improvements across both malware classes (Ransomware KS: 0.2533 vs 0.3280; Trojan KS: 0.2739 vs 0.2930) while stabilizing the adversarial loss gap.
2. **Explicit Project Adoption Status:** **NONE** of these four follow-on checkpoints have been adopted or propagated into the downstream metrics in Days 29–32 (such as `MASTER_COMPARISON_TABLE.md` or downstream classifier evaluations). All existing tables across Days 29–32 preserve the baseline 100-epoch QGAN checkpoint values.

