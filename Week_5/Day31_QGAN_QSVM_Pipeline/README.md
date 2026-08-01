# Day 31 — Run fixed-QGAN data through the V3 QSVM/VQC pipeline (Shanmukh)

**⚠️ READ FIRST: `results/SEED_EXPANSION_STATISTICAL_CORRECTION.md`** — the
"QSVM v3 wins on all 3 datasets" claim later in this README (and in Day 32)
was based on 1-2 seeds and does **not hold** once more seeds are tested.
Only Original data shows a robust win (3/3 seeds, low variance). CTGAN,
SMOTE, and QGAN all show a *net loss* on average with proper multi-seed
testing, despite real, verified favorable single-seed headlines (0.875,
0.600). Read the correction document for full statistics before trusting
anything below it at face value.

**Task:** Run the fixed-QGAN-augmented data through the same quantum pipeline
(QSVM/VQC) already used for CTGAN and SMOTE, completing the full
augmentation-method x classifier-family comparison.

## Status: COMPLETE. No blockers.

**Architecture in brief** (full detail: `../Day33_Stabilization_Methodology/README.md`
Part B, and `../Day34_V3_Supplementary/README.md`): V3 = XGBoost feature
selection (top-8 features by importance, no PCA/LDA compression) → RobustScaler
to `[-π,π]` angle encoding → a **local, pair-averaged kernel** (average of
pairwise qubit "both zero" probabilities, not a single global state overlap)
with a small set of **trainable rotation weights optimized via kernel-target
alignment (KTA)** before use → Nystrom-approximated QSVM (40 landmarks) for
scalability. Same architecture used for every dataset in this folder — only
the input data changes between runs.

`code/pipeline_v3_qgan.py` is the V3 QSVM pipeline (identical
kernel/alignment/Nystrom logic to the CTGAN script,
`code/pipeline_v3_CTGAN_ORIGINAL_reference.py`, copied verbatim) with only the
data-loading function swapped — so any accuracy difference is attributable to
the input data, not a reimplementation difference.

**Two runs were done**, because the first turned out to be an unfair
comparison (see `results/QGAN_vs_CTGAN_V3_COMPARISON.md` for the full story):

### Run 1 — as-uploaded QGAN dataset (~2-5% synthetic, gap-fill only)
`data/malmem_qgan_augmented.csv` -> `results/results_v3_qgan_gapfill.json`

| Seed | classical SVM | classical RF | QSVM v3 |
|---|---|---|---|
| 42 | 0.300 | 0.375 | 0.425 |
| 1 | 0.350 | 0.550 | 0.400 |
| 7 | 0.525 | 0.425 | 0.400 |

### Run 2 — matched-scale QGAN dataset (rebuilt to match CTGAN's actual ~33-40% synthetic composition)
`data/malmem_qgan_matched_ctgan_scale.csv` -> `results/results_v3_qgan_matched.json`

| Seed | classical SVM | classical RF | QSVM v3 | synth_frac |
|---|---|---|---|---|
| 42 | 0.500 | 0.475 | 0.525 | 0.339 |
| 1 | 0.425 | 0.500 | 0.550 | 0.361 |
| 7 | 0.500 | 0.450 | 0.625 | 0.337 |

CTGAN reference (same protocol, `results/results_v3_CTGAN_ORIGINAL_reference.json`):
seeds 42/1/7 -> QSVM 0.800/0.850/0.875.

## Update — QGAN now has V1/V2/V3 coverage (previously V3-only)

Using Shanmukh's own `seed_expansion_code/quantum_all_versions.py` (unmodified logic, only a
QGAN data-loading branch added — see `seed_expansion_code/`), QGAN was run
through all three QSVM architecture versions at seed=7:

| Version | QGAN acc | Classical best | Margin |
|---|---|---|---|
| V1 (PCA + global kernel) | 0.350 | 0.567 | -21.7pp |
| V2 (LDA/PCA hybrid + re-upload + Nystrom) | 0.550 | 0.633 | -8.3pp |
| V3 (XGBoost + local aligned kernel) | 0.625 | 0.587 | +3.8pp at this seed — see correction doc for the full 5-seed picture (net negative on average) |

QGAN loses at every architecture version except V3's single best seed.
Consistent with Day 33's diagnosis that the generator's adversarial training
never fully stabilized — a better downstream architecture can't fully
compensate for lower-fidelity generated input data.

### The headline finding — SUPERSEDED, see correction doc

The paragraph below reflects the original 3-seed (42/1/7) analysis and is
kept for provenance, but is **superseded by `results/SEED_EXPANSION_STATISTICAL_CORRECTION.md`**,
which reran these same 3 seeds via a second, independently-written script
and found the classical baseline shifts enough between implementations that
2 of 3 seeds (42, 1) actually flip to a loss — only seed 7 robustly wins
across both scripts. The "3/3 seeds, tied margin" framing below does not
hold once cross-checked against a second implementation, and gets worse once
2 more seeds (100, 123) are added on top.

<details>
<summary>Original (superseded) analysis</summary>

Raw accuracy: CTGAN wins (0.80-0.875 vs 0.525-0.625). But the **quantum-vs-classical
margin** — the thing this task is actually measuring — was originally reported
as statistically tied: QGAN wins 3/3 seeds by +6.7pp average, CTGAN wins 3/3
seeds by +7.5pp average. This does not hold under the seed expansion above.
</details>

Full writeup: `results/QGAN_vs_CTGAN_V3_COMPARISON.md` (also superseded in its
"3/3 wins" framing, kept for provenance).
Logs: `logs/run_gapfill_seeds.log`, `logs/run_matched_scale_seeds.log`.

## Update — SMOTE now included (comprehensive study, Shanmukh) — see correction at top of file

The one item left open in earlier versions of this package — a V3-pipeline
SMOTE run — is closed. Shanmukh uploaded a comprehensive study covering
QSVM v1/v2/v3 and VQC v1/v2/v3 across all three datasets, independently
verified against its raw JSON before trusting it (see
`comprehensive_study_SMOTE_addendum_README.md`).

**Its headline numbers were then re-tested with 2 additional seeds each — see
`results/SEED_EXPANSION_STATISTICAL_CORRECTION.md` for the full statistics.
Summary: only Original held up as a genuine win (3/3 seeds). CTGAN (1/3),
SMOTE (1/4), and QGAN (1/5) all showed net losses on average — the original
0.875/0.600/0.500 headlines below were real runs but not representative.**

| Dataset | Headline QSVM v3 (1-2 seeds) | Corrected mean ± std (3-5 seeds) | Corrected win rate |
|---|---|---|---|
| CTGAN | 0.875 | 0.708 ± 0.170 | 1/3 |
| SMOTE | 0.600 (best-of-2) | 0.444 ± 0.094 | 1/4 |
| Original | 0.500 | 0.542 ± 0.031 | **3/3** |
| QGAN | 0.625 (best seed) | 0.540 ± 0.049 | 1/5 |

Full detail, verification steps, and the corrected statistics:
`comprehensive_study_SMOTE_addendum_README.md` and
`results/SEED_EXPANSION_STATISTICAL_CORRECTION.md`. Raw data:
`comprehensive_study_SMOTE_addendum/`, `seed_expansion_code/`.
