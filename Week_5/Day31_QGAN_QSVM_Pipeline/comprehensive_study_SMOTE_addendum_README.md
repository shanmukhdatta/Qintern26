# Addendum — Comprehensive V1/V2/V3 study (QSVM + VQC), all 3 datasets (Shanmukh)

**⚠️ SUPERSEDED: the headline claim below ("QSVM v3 meets or beats classical
on all three datasets") does not hold once expanded to 3-5 seeds per
dataset — see `results/SEED_EXPANSION_STATISTICAL_CORRECTION.md` in this
folder's parent directory for the corrected statistics. Only Original data
holds up as a robust win. This document is kept unmodified for provenance —
it's a real, verified study, just not the final word on the underlying
question.**

This folder is Shanmukh's independently-run comprehensive study, uploaded to
close Day 31/32's one remaining open item (a V3-pipeline SMOTE run). It turned
out to be much bigger than just that — a full QSVM v1/v2/v3 × VQC v1/v2/v3 ×
{Original, SMOTE, CTGAN} grid (18 configs), with matched-feature classical
baselines for every cell. Kept in its original structure (`classical/`,
`quantum/`, `comparative_results/`, `images/`) rather than reorganized, so it
stays traceable to the source.

**Start with `comprehensive_study_SMOTE_addendum/comparative_results/COMPARATIVE_RESULTS.md`** — it's the main
analysis file and is genuinely well-disclosed (states its own goal upfront,
checks it explicitly rather than just asserting it, and reports an unfavorable
result — VQC losing on every dataset/version — as plainly as the favorable ones).

## What I independently verified before trusting it

- **CTGAN seed=7 result (0.875/0.873) matches our own Day 31 V3-CTGAN run
  exactly** — a genuine independent cross-check, not circular (two different
  sessions, same underlying protocol, same number).
- **Checked the raw JSON behind the disclosed seed note myself**: QSVM v3 on
  SMOTE really does score 0.350 at seed=7 (below classical's 0.46/0.52) and
  0.600 at seed=1 (above classical's 0.527/0.507) —
  `comprehensive_study_SMOTE_addendum/quantum/qsvm_v3/qsvm_v3_smote.json`
confirms both numbers exactly as
  reported in the markdown.
- **Confirmed the "classical (matched features)" baseline is the
  same-features-as-quantum SVM/RF embedded per-run** (e.g.
  `classical_svm_same_features` inside
`comprehensive_study_SMOTE_addendum/quantum/qsvm_v3/qsvm_v3_ctgan.json`), not the
  separately-tuned, more-featured LightGBM/RF in `classical/*.json` — those
  are a different, higher baseline (e.g. 0.913 LightGBM on CTGAN) kept only
  as additional context, explicitly not the comparison basis. This matches
  our own project's convention exactly (Day 31 used the same
  same-features-as-quantum discipline).

## One protocol caveat worth flagging plainly

**CTGAN and Original results here are single-seed (seed=7 only). SMOTE tried
two seeds (7, then 1) and reported the better one as the headline, with the
worse one disclosed alongside it — not hidden, but also not the same
3-seed-average protocol Day 31 used for our own QGAN-vs-CTGAN comparison
(seeds 42/1/7, margin averaged across all three, no seed dropped).** This
doesn't invalidate the result — the source document discloses it plainly, and
seed=1's 0.600 is a real, verified run, not a fabrication — but it does mean
"QSVM v3 beats classical on SMOTE" is currently resting on a single favorable
seed rather than a multi-seed average, unlike the CTGAN-vs-QGAN comparison
elsewhere in this project. Worth a 2-3 more seed run on SMOTE specifically if
this result needs to hold up to closer scrutiny later.

## Headline result, folded into Day 32's master table

| Dataset | QSVM v3 | Classical (matched) best | Margin | Seed(s) |
|---|---|---|---|---|
| CTGAN | 0.875 | 0.873 | +0.2pp | 7 only |
| SMOTE | 0.600 | 0.527 | +7.3pp | 1 (best-of-2; seed 7 gave 0.350, below classical) |
| Original | 0.500 | 0.467 | +3.3pp | 7 only |

**QSVM v3 is the only version/model that meets-or-beats classical on all
three datasets.** QSVM v1, QSVM v2, and every VQC version lose to classical
on every dataset tested — consistent with the capacity/architecture story
built up across this whole project (Day 33's diagnosis, V1's capacity-limited
finding, V2's harder-data-favors-quantum pattern) rather than a contradiction
of it.
