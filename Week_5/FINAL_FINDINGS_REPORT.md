# FINAL FINDINGS REPORT — QTagger+ Quantum vs. Classical Malware Classification

**What this document is:** the single place that explains everything in this
zip — every result, why each number is high or low, every variation tried,
and what's actually proven vs. still open. Read this first; every other
document in the zip is detail behind one row of the tables below.

---

## 1. The headline answer

**Does quantum beat classical? Yes — but only on one specific kind of data,
and that finding is now proven three independent ways, not asserted once.**

Quantum (QSVM) shows a real, low-variance, reproducible advantage over
classical models **only on the original, unaugmented, messy dataset**. On
every form of synthetic/augmented data tested (CTGAN, SMOTE, QGAN, all
architecture versions), quantum's apparent wins turned out to be high-variance
noise — real individual results, but not a stable advantage once tested
properly with multiple random seeds.

This was **not the original conclusion of this project**. Early results
(a single favorable seed) suggested quantum won on CTGAN too (0.875 accuracy).
That number was real and independently verified — but it did not survive
being re-tested with more seeds. Section 4 explains exactly how that
correction was found and why it's trustworthy.

---

## 2. Master comparison table — every dataset, every method, why each result is what it is

| Dataset | Method | Result | Why this number is high or low |
|---|---|---|---|
| **Original** | Classical (SVM/RF) | 0.36–0.38 avg (V2 grid, 9 configs) | **Low.** Real MalMem2022 data is genuinely hard — no class-balancing, natural noise, overlapping feature distributions between malware families. Classical models struggle to find a clean decision boundary. |
| **Original** | QSVM v2 | 0.45 avg, up to 0.558 at n=1000 | **Higher than classical, and the gap *grows* with sample size** — classical degrades 0.442→0.317 as n grows 250→1000, QSVM improves 0.367→0.558 in the same range. This is the single most consistent, largest quantum advantage in the entire project. |
| **Original** | QSVM v3, 3 seeds | 0.542 ± 0.031 mean, margin +3.3pp, **3/3 win rate** | **Robust win, low variance.** The only dataset where every single seed tested shows quantum ahead. |
| **CTGAN** | Classical (SVM/RF) | 0.68–0.70 avg (V2 grid) | **High.** CTGAN generates synthetic samples *conditioned on class label* — this produces tight, well-separated per-class clusters that are close to the easiest possible input for a classical kernel or tree ensemble. |
| **CTGAN** | QSVM v2 | 0.619 avg, **1/9 win rate** | **Loses most of the time.** Same reason in reverse — when the data is already easy to separate linearly/by trees, quantum's more complex decision boundary has no advantage to offer. |
| **CTGAN** | QSVM v3, 1 seed (headline) | **0.875**, margin +0.2pp | **This was the project's best single number — and it was misleading.** Real, verified, reproducible (matched independently to 4 decimal places by a second script) — but only at this one seed. |
| **CTGAN** | QSVM v3, 3 seeds (corrected) | 0.708 ± 0.170 mean, margin **-12.5pp**, **1/3 win rate** | **The honest number.** Accuracy swings from 0.875 to 0.475 across seeds on the *identical* config — a 40-point range. std=0.170 is enormous. The 0.875 was the high end of noise, not a stable result. |
| **SMOTE** | Classical (SVM/RF) | 0.39 avg (V2 grid) | **Low-moderate.** SMOTE interpolates between real neighbors, so it doesn't create CTGAN's artificially clean clusters, but it does correctly preserve the real data's underlying difficulty. |
| **SMOTE** | QSVM v2 | 0.394 avg, **3/9 win rate** | **Roughly tied with classical.** Middle ground between Original (hard, quantum wins) and CTGAN (easy, quantum loses) — consistent with the difficulty gradient. |
| **SMOTE** | QSVM v3, best-of-2-seed (headline) | **0.600**, margin +7.3pp | **Already a red flag in the original writeup**: this was *chosen* after seed=7 scored 0.350 (a loss). Best-of-2 is not a representative statistic. |
| **SMOTE** | QSVM v3, 4 seeds (corrected) | 0.444 ± 0.094 mean, margin **-7.7pp**, **1/4 win rate** | **The honest number — a net loss.** Confirms the best-of-2 selection was cherry-picking a favorable outlier, not reporting a real trend. |
| **QGAN** | Classical (SVM/RF) | ~0.55–0.59 (this project's own runs) | **Moderate.** QGAN's synthetic rows are close to real data in aggregate statistics (moment-matching loss was active during training) but not in full distribution shape — see Section 3 for why. |
| **QGAN** | QSVM v1 | 0.350, margin **-21.7pp** | **Badly loses.** V1's simple global-overlap kernel and PCA-only projection is the weakest architecture tested anywhere in this project — consistent with V1's only other documented result (Original dataset, Week 3-4 era: classical beat it by 8-17pp at its best config). Both times V1 was tested, it lost. |
| **QGAN** | QSVM v2 | 0.550, margin **-8.3pp** | **Loses.** Better than V1 but still behind classical — QGAN's synthetic data isn't clean enough for even the improved V2 architecture to compensate. |
| **QGAN** | QSVM v3, 5 seeds | 0.540 ± 0.049 mean, margin **-4.5pp**, **1/5 win rate** | **Net loss, but the *most consistent* variance of the three synthetic datasets** (std=0.049 vs. CTGAN's 0.170) — QGAN's numbers are more stable, just consistently mediocre rather than wildly noisy. |
| **QGAN synthetic data itself** | Fidelity (KS/Wasserstein/MMD) vs. real | KS_median 0.28–0.33, worst of the three generators | **High = bad here** (lower is better fidelity). QGAN ranks **worst of SMOTE/CTGAN/QGAN on every single fidelity metric, every class** — see Section 3, this is the root cause behind QGAN's weak downstream numbers. |
| **SMOTE synthetic data** | Fidelity | KS_median 0.010–0.013, **best of the three** | **Low = good.** SMOTE interpolates between literal real neighbors, so it's structurally guaranteed to be statistically close to real data — this is close to the best-case scenario for a fidelity metric, almost by construction. |
| **CTGAN synthetic data** | Fidelity | KS_median 0.091–0.133, **middle** | **Better than QGAN, worse than SMOTE.** CTGAN learns a genuine generative model (harder than SMOTE's interpolation) but produces the class-conditional over-clustering described above, which shows up as moderate — not perfect — fidelity. |
| **QGAN augmented data** | Downstream classifiers (RF/XGBoost/LightGBM/SVM F1) | ~flat, -0.06 to -0.5pp vs. no augmentation | **Flat, not negative or positive — because the effect is too small to measure.** QGAN's gap-fill scheme only adds synthetic rows up to 2–5% of training volume for the affected classes. Even a perfect generator couldn't move a classifier already at 87%+ accuracy by injecting that little data. |
| **SMOTE augmented data** | Downstream classifiers | **+2.9 to +8.6pp**, largest of any method | **High — the standout result of the whole project on this metric.** Combination of high fidelity (close to real data) *and* high volume (66–68% synthetic rows) — both matter together, not either alone (see CTGAN below). |
| **CTGAN augmented data** | Downstream classifiers | **-0.5 to -2.0pp, the only negative result** | **Slightly harmful.** Similar synthetic volume to SMOTE (62–66%) but worse fidelity — the class-conditional over-clustering that helped CTGAN look easy in isolated small-sample tests actively hurts a classifier trained on the full blended dataset, introducing distribution shift against real test data. |

---

## 3. QGAN stabilization — every variation tried, and why each one did or didn't work

| Variation | What changed | Result | Why |
|---|---|---|---|
| **30 → 100 epochs** | Just trained longer, same architecture | Ransomware fidelity improved (Wasserstein -38.6%); Trojan got worse (+53.4%) | **Mixed, not uniform.** Loss curves show discriminator winning steadily in all 3 classes (d_loss falling, g_loss rising, growing oscillation) — more epochs helped where the generator was still trending the right direction, didn't help where it had already plateaued. |
| **Output-bias calibration** (post-hoc mean/std matching) | Force synthetic data's statistics to exactly match real data after generation | Made fidelity **worse** for 2/3 classes | The moment gap was already small (moment-matching loss was active during training) — forcing it to exactly zero via affine correction distorted the distribution shape after inverse-PCA-transform, especially near the clipping boundary. |
| **Quantile mapping** (rank-based distribution matching) | Remap synthetic values onto real data's empirical CDF | Near-perfect fidelity scores (KS≈0) | **Invalid, not a real fix.** With matched sample sizes, this forces synthetic marginals to become literal copies of real values evaluated against that same real data — evaluation leakage. Explicitly excluded from every legitimate result in this package. |
| **Generator capacity increase** (4→6 variational layers) + **stronger discriminator regularization** (dropout 0.20→0.30) | Rebuilt and retrained Trojan from scratch with both changes | Made it **measurably worse** — fidelity KS_median 0.216→0.254, Wasserstein +74%; loss imbalance widened, not narrowed | Consistent with **barren plateaus** — deeper variational quantum circuits are known to have harder-to-compute gradients, so "more expressive in theory" became "harder to train in practice." A real, documented negative result, not a failure to hide. |
| **LR decay + moderate dropout (0.20)** | *(This was already active in the original uploaded code before this project started — not a new attempt)* | Discriminator still won despite this | Confirms the problem is a genuine **capacity mismatch** (generator structurally weaker than discriminator), not simply an untuned learning rate or missing regularization — those were already in place and insufficient. |

**Bottom line on QGAN: three real fix attempts were tried, all three failed,
and that's reported as a finding, not hidden.** The root cause (discriminator
dominance from a capacity mismatch between a shallow 4-6 layer quantum
generator and even a small classical discriminator) is diagnosed with actual
loss-curve evidence, not guessed at.

---

## 4. How the biggest correction of the project was found — the proof, step by step

This matters because it's the difference between "we found a number" and
"we found a number and checked it holds up":

1. **Original claim**: QSVM v3 beats classical on CTGAN (0.875), SMOTE (0.600),
   and Original (0.500) — based on 1 seed for CTGAN/Original, and a
   best-of-2-seed pick for SMOTE (seed 7 lost at 0.350, seed 1 won at 0.600,
   seed 1 was reported as the headline).
2. **Independent verification, not blind trust**: before accepting the 0.875
   CTGAN number, it was cross-checked against raw JSON output (not just the
   summary text) and against the saved model filename — both confirmed the
   number was real.
3. **The test that actually mattered**: rather than stopping at verification,
   2-5 *additional* seeds were run per dataset using the same validated code.
4. **Result**: CTGAN's accuracy swung from 0.875 down to 0.475 across just 3
   seeds — a 40-point range on the identical configuration. SMOTE similarly
   lost at 3 of 4 seeds tested. QGAN lost at 4 of 5. **Only Original won at
   every single seed tested (3/3), with by far the lowest variance
   (std=0.024 on the margin vs. 0.10–0.17 for the others).**
5. **Cross-check against a completely separate result**: this pattern —
   quantum wins reliably on Original, not on synthetic data — was already
   found independently by the V2 pipeline's 27-config grid (Section 2, top
   rows), built and run in a different session with a different architecture
   version. Two unrelated pipeline versions, two different sets of random
   seeds, the same conclusion. That agreement is what makes this finding
   trustworthy rather than a coincidence of one particular test.

---

## 5. What's proven vs. what's still open

**Proven, with evidence that would survive scrutiny:**
- Quantum reliably beats classical on the original, unaugmented dataset (2
  independent pipeline versions, multiple seeds, consistent low variance).
- QGAN's generator never fully stabilized — root-caused with actual loss
  curve data, not asserted.
- SMOTE is the best augmentation method for downstream classifier impact;
  CTGAN is mildly harmful; QGAN is neutral — all measured on an identical,
  verified-matching test split across all four methods.
- Fidelity ranks SMOTE > CTGAN > QGAN consistently, on every class and metric.

**Real, verified, but not representative on their own — kept for
transparency, not deleted:**
- CTGAN's 0.875 and SMOTE's 0.600 headline numbers. They happened. They just
  don't generalize past the specific seed that produced them.

**Still open if this needs to go further:**
- 3-5 seeds is enough to overturn a false headline and establish a direction
  — not enough for tight statistical confidence intervals. 10+ seeds per
  dataset would be the next step.
- QGAN has no VQC results (only QSVM v1/v2/v3) and no V1/V2-pipeline-matched
  fidelity comparison to CTGAN/SMOTE at those versions.
- Every downstream classifier number in this project is on capped/subsampled
  data (n≤500 for QSVM comparisons); full-dataset classifier numbers (Day 29)
  use a different, larger sample and shouldn't be directly compared to the
  QSVM accuracy numbers in Section 2.

---

## 6. Where to look for more detail

| Want to know... | Go to |
|---|---|
| Exact downstream classifier numbers, all 4 methods | `Day29_Downstream_Classifier_Comparison/` |
| Exact fidelity numbers, all 4 methods | `Day30_Fidelity_Reconciliation/` |
| Full seed-by-seed QSVM data behind Sections 2 and 4 | `Day31_QGAN_QSVM_Pipeline/results/SEED_EXPANSION_raw_results.csv` and `SEED_EXPANSION_STATISTICAL_CORRECTION.md` |
| The V2 27-config grid referenced throughout | `Day32_Master_Comparison_Tables/reference/V2_27config_CTGAN_SMOTE_Original_grid.md` |
| Full QGAN stabilization diagnosis + all 3 failed fixes | `Day33_Stabilization_Methodology/` |
| V1→V2→V3 architecture evolution, explained | `Day33_Stabilization_Methodology/README.md` Part B |
| The verified 0.875 result's origin and ablation | `Day34_V3_Supplementary/` |
| All code used to produce every number above | `code/` subfolders in Days 29-31, 33 (self-contained, runnable with a data path adjustment — see each folder's README) |
