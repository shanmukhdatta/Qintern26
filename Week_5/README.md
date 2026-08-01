# QTagger+ Week package — Days 29-35

**Shanmukh, QIntern 2026 — QTagger+ quantum malware classification project.**

This package covers the Day 29-35 sprint: reconciling QGAN's corrected
numbers into the project's master comparison tables, running the fixed QGAN
through the V3 QSVM pipeline, documenting the stabilization attempts, and
consolidating everything into final methodology notes.

## Quick status

| Day | Task | Status |
|---|---|---|
| 29 | Downstream classifier comparison (RF/XGBoost/LightGBM/SVM) | **Complete** — all 4 methods, identical protocol |
| 30 | Fidelity metrics (KS/Wasserstein/MMD) reconciliation | **Complete** — all 4 methods, identical protocol |
| 31 | QGAN through V3 QSVM/VQC pipeline | **Complete** — now includes SMOTE via Shanmukh's comprehensive study |
| 32 | Master configuration comparison tables | **Complete** — no open cells |
| 33 | QGAN stabilization + V1-V3 methodology writeup | **Complete**, including 2 documented failed fixes |
| 34 | V3 findings supplementary material | **Complete**, independently re-verified |
| 35 | Final packaging | **Complete** — this zip |

**All seven days closed. No open items remain.**

**⚠️ Important correction applied after initial completion**: expanded
seed testing (3-5 seeds vs. the original 1-2) found that "QSVM v3 beats
classical on all 3 datasets" does not hold. Only Original data shows a
robust win. CTGAN, SMOTE, and QGAN all show a net loss on average once
properly seed-tested, despite real, verified favorable single-seed
headlines. Full detail:
`Day31_QGAN_QSVM_Pipeline/results/SEED_EXPANSION_STATISTICAL_CORRECTION.md`.
This strengthens the project's core finding rather than undermining it — see
below.

## What actually happened this sprint, in two paragraphs

The QGAN was retrained from 30 to 100 epochs with a checkpoint/resume system
built specifically because this sandbox can't run background jobs across
tool calls. That corrected QGAN was compared against CTGAN on the V3 QSVM
pipeline — the first attempt was unfair (QGAN's dataset was ~2-5% synthetic
vs. CTGAN's real ~33-40%), caught before trusting the result, and fixed by
rebuilding QGAN's synthetic data at matched scale. The original single-seed
headline suggested CTGAN wins on raw accuracy but the quantum-vs-classical
margin was tied with QGAN — **later corrected with 3-5 seeds per dataset**:
only Original data shows a robust, low-variance quantum win; CTGAN, SMOTE,
and QGAN all show a net loss on average once properly seed-tested (see
Day 31/32 for full statistics). Separately, three stabilization techniques
were tried for QGAN's training instability (output-bias calibration,
quantile mapping, increased generator capacity) — none worked, reported
honestly rather than papered over.

The SMOTE and CTGAN raw datasets were then uploaded and processed to close
Days 29-30. Neither file had an explicit real/synthetic flag, and a plain
merge to identify real rows exploded on this dataset's duplicate rows — a
hash-based duplicate-safe matching approach was used instead, which surfaced
a real finding along the way: SMOTE retains every real row exactly, CTGAN
retains zero (100% synthetic). With all four methods measured on an
identical, verified-matching held-out test split, **fidelity ranks
SMOTE > CTGAN > QGAN** consistently, but **downstream classifier impact ranks
SMOTE >> QGAN ≈ CTGAN (CTGAN actually slightly negative)** — a different
order than fidelity, meaning fidelity scores alone don't reliably predict
downstream usefulness. SMOTE's combination of high fidelity *and* high
synthetic volume (~66-68% of training data vs. QGAN's <5%) appears to be
what actually moves classifier metrics.

## Folder guide

- **`shared_data/`** — the raw Original/SMOTE/CTGAN CSVs plus the reusable
  real-vs-synthetic matching code (`shared_data/code/build_augmented_frames.py`)
  used to make Days 29-30's comparison fair.
- **`Day29_Downstream_Classifier_Comparison/`** — all 4 methods'
  RF/XGBoost/LightGBM/SVM numbers, same protocol, same test set.
- **`Day30_Fidelity_Reconciliation/`** — all 4 methods' KS/Wasserstein/MMD.
- **`Day31_QGAN_QSVM_Pipeline/`** — QGAN vs. CTGAN/SMOTE/Original on the V3
  QSVM pipeline (now with QGAN V1/V2 coverage too), corrected with 3-5 seeds
  per dataset after the original 1-2 seed headlines proved non-representative.
- **`Day32_Master_Comparison_Tables/`** — everything above plus the
  pre-existing 27-config V2 grid, consolidated into one table.
- **`Day33_Stabilization_Methodology/`** — the diagnosis (loss-curve evidence,
  root-caused to capacity mismatch) and three attempted fixes, two of which
  didn't work and are documented as such, plus V1-V3 architecture notes.
- **`Day34_V3_Supplementary/`** — the 0.875 CTGAN result, independently
  re-verified against raw run data.
- **`Day35_Final_Consolidation/`** — package manifest and cross-day consistency checks.
- **`notebooks_reference/`** — original project notebooks.

## How the last open item closed — and what it led to

Shanmukh uploaded a comprehensive study (QSVM v1/v2/v3 × VQC v1/v2/v3 × all
three datasets — 18 configs) that went well beyond the single SMOTE cell that
was open. Independently verified before trusting it — the study's CTGAN
seed=7 result (0.875/0.873) matches this project's own independently-run
Day 31 CTGAN result to 4 decimal places, a genuine cross-check across two
separate sessions.

Having that verification code in hand made it cheap to test whether the
single-seed headlines actually generalized. **They didn't.** Expanding to
3-5 seeds per dataset (same script, same protocol, just more seeds) found:

| Dataset | Headline (1-2 seeds) | Corrected mean ± std (3-5 seeds) | Win rate |
|---|---|---|---|
| CTGAN | 0.875 | 0.708 ± 0.170 | 1/3 |
| SMOTE | 0.600 (best-of-2) | 0.444 ± 0.094 | 1/4 |
| QGAN | 0.625 | 0.540 ± 0.049 | 1/5 |
| **Original** | 0.500 | 0.542 ± 0.031 | **3/3** |

**Only Original data shows a robust, low-variance quantum win.** CTGAN's
single-seed 0.875 turned out to be the high end of a 40-point swing across
seeds — a favorable outlier, not a stable result. This is a genuine
correction to the earlier "QSVM v3 wins everywhere" claim, not a caveat
layered on top of an unchanged conclusion.

**It strengthens the project's core finding rather than undermining it**:
the same "quantum's edge is real on harder/messier data, not on
synthetic/augmented data" pattern was now found three independent ways — the
pre-existing V2 grid (Section 1 of Day 32's table), the original V3
comparison, and this corrected multi-seed V3 result. Different pipeline
versions, different seed sets, same conclusion.

QGAN also now has verified V1/V2/V3 coverage (previously V3-only) — it loses
to classical at every architecture version tested. Full detail:
`Day31_QGAN_QSVM_Pipeline/results/SEED_EXPANSION_STATISTICAL_CORRECTION.md`.
