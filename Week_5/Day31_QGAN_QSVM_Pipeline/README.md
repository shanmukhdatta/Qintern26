# Day 31 — QGAN Through the V3 QSVM/VQC Pipeline (Shanmukh)

## Objective
Run the fixed-QGAN-augmented data through the same quantum pipeline
(QSVM/VQC) already used for CTGAN and SMOTE, completing the full
augmentation-method x classifier-family comparison.

## Tasks Performed
- Ran QGAN's synthetic data through the V3 QSVM pipeline (XGBoost feature
  selection → RobustScaler → local pair-averaged quantum kernel with
  trainable rotation weights → Nystrom-approximated QSVM), the exact same
  architecture already used for CTGAN and SMOTE, with only the input data
  swapped — so any accuracy difference is attributable to the data, not to
  a different implementation.
- Discovered and fixed an unfair first comparison (see Methodology).
- Extended QGAN's coverage to all three QSVM architecture versions (V1, V2,
  V3), not just V3.
- Incorporated a comprehensive study (18 configs: QSVM v1/v2/v3 × VQC
  v1/v2/v3 × three datasets) and independently verified it before trusting
  it.
- Re-tested the headline single-seed numbers with several additional seeds,
  which overturned the original "quantum wins everywhere" conclusion.

## Methodology
- **Fairness check first:** the first QGAN run used the as-uploaded dataset,
  which was only ~2-5% synthetic (QGAN's gap-fill scheme). CTGAN's dataset,
  by contrast, was ~33-40% synthetic. Comparing these directly would have
  been comparing two very different amounts of synthetic data, not two
  augmentation *methods* — so this was caught before trusting the result,
  and QGAN's synthetic data was rebuilt at matched scale before re-running.
- **Same code, different data:** `code/pipeline_v3_qgan.py` is a verbatim
  copy of the CTGAN script (`code/pipeline_v3_CTGAN_ORIGINAL_reference.py`)
  with only the data-loading function changed, ruling out "different code"
  as an explanation for any accuracy gap.
- **Verification before trusting the comprehensive study:** the uploaded
  18-config study's CTGAN seed=7 result (0.875/0.873) was checked against
  this project's own independently-run Day 31 CTGAN result — it matched to
  4 decimal places, a genuine cross-check across two separate sessions.
- **Seed expansion as the real test:** rather than stopping at one favorable
  seed, 2-5 additional seeds were run per dataset using the same verified
  code, to check whether the original headline numbers were representative
  or just a lucky draw.

## Results

**Run 1 — QGAN as-uploaded (~2-5% synthetic, unfair comparison):**

| Seed | classical SVM | classical RF | QSVM v3 |
|---|---|---|---|
| 42 | 0.300 | 0.375 | 0.425 |
| 1 | 0.350 | 0.550 | 0.400 |
| 7 | 0.525 | 0.425 | 0.400 |

**Run 2 — QGAN matched-scale (~33-40% synthetic, fair comparison):**

| Seed | classical SVM | classical RF | QSVM v3 | synth_frac |
|---|---|---|---|---|
| 42 | 0.500 | 0.475 | 0.525 | 0.339 |
| 1 | 0.425 | 0.500 | 0.550 | 0.361 |
| 7 | 0.500 | 0.450 | 0.625 | 0.337 |

CTGAN reference, same protocol, seeds 42/1/7: QSVM 0.800 / 0.850 / 0.875.

**QGAN across all three architecture versions (seed=7):**

| Version | QGAN acc | Classical best | Margin |
|---|---|---|---|
| V1 (PCA + global kernel) | 0.350 | 0.567 | -21.7pp |
| V2 (LDA/PCA hybrid + Nystrom) | 0.550 | 0.633 | -8.3pp |
| V3 (XGBoost + local aligned kernel) | 0.625 | 0.587 | +3.8pp (single seed — see correction below) |

**The correction that overturned the original headline — expanded to
3-5 seeds per dataset:**

| Dataset | Headline (1-2 seeds) | Corrected mean ± std (3-5 seeds) | Win rate |
|---|---|---|---|
| CTGAN | 0.875 | 0.708 ± 0.170 | 1/3 |
| SMOTE | 0.600 (best-of-2) | 0.444 ± 0.094 | 1/4 |
| QGAN | 0.625 (best seed) | 0.540 ± 0.049 | 1/5 |
| **Original** | 0.500 | 0.542 ± 0.031 | **3/3** |

Full seed-by-seed data: `results/SEED_EXPANSION_raw_results.csv` and
`results/SEED_EXPANSION_STATISTICAL_CORRECTION.md`.

## Interpretation (in plain language)
- **QGAN loses at every architecture version tested**, except a single best
  seed at V3 — and that one win does not hold up once more seeds are tested
  (5-seed average: net loss). This is consistent with QGAN's generator never
  fully stabilizing during training (Day 33).
- **Why Original data wins reliably and synthetic data doesn't**: real,
  unaugmented data is genuinely messy — feature distributions overlap
  between malware families, there's no artificial cleanup, no clear
  separating line between classes. Classical models struggle to find a
  clean boundary in that mess, and the quantum kernel's different geometry
  gives it a small, *consistent* edge there. Synthetic data — whichever
  method generated it — tends to be more regular/separable than real data
  (CTGAN especially, since it generates data *conditioned on class label*,
  producing tight, well-separated per-class clusters). Once the data is
  already easy to separate, a classical model handles it just fine on its
  own, and there's no gap left for a more complex quantum boundary to fill.
  Any apparent quantum "win" on that kind of data (like CTGAN's 0.875) turns
  out to be noise riding on top of an already-easy problem, not a real edge —
  which is exactly what the 40-point swing (0.875 → 0.475) across seeds
  demonstrates.
- **Why verification mattered here**: the 0.875 CTGAN number was real and
  reproducible (matched independently to 4 decimal places) — but "real and
  reproducible at one seed" is not the same as "representative." Expanding
  the seed count is what actually tested whether the headline generalized,
  and it didn't.

## Conclusion
- Full augmentation-method × classifier-family comparison (CTGAN, SMOTE,
  QGAN, Original — all on the same V3 pipeline) is now complete.
- QGAN's original single-seed 0.625 "win" at V3 does not survive multi-seed
  testing (5-seed average: net loss, -4.5pp margin).
- The only dataset that shows a robust, low-variance quantum win across
  every seed tested is Original — synthetic data of any kind (CTGAN, SMOTE,
  QGAN) shows a net loss on average once properly seed-tested.
- Decision: carry this corrected result forward into Day 32's master table,
  explicitly retiring the original "QSVM v3 wins everywhere" claim.
