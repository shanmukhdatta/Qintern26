# Day 34 (Optional) — Version 3 findings as supplementary material

**Task:** Document XGBoost-based feature selection, trainable quantum kernel
via KTA, and local-kernel measurement as supplementary material, only if it
doesn't displace Days 29-33.

## Status: Complete. Did not displace any other day (done after Days 29-33 were handled).

This day is mostly a **verification + organization** pass over material that
already existed in the uploaded seminar package, not new experimentation —
consistent with the task being optional supplementary documentation.

### What was verified this session (not just copied)

The 0.875 CTGAN accuracy claim was cross-checked directly against raw run
data, not taken on faith:

- `results/results_v3full.json` — found the exact matching run:
  `qubits=8, seed=7, variant=xgboost, acc=0.875, f1_macro=0.8732974910394264`
- `model/model_bundle_xgboost_q8_seed7_acc0875.pkl` — filename encodes the
  same seed/qubit/accuracy combination independently
- `results/ablation_results.json` — local kernel alone: 0.825, KTA alignment
  alone: 0.725, combined: 0.875 (both components needed, neither alone
  suffices)
- Cross-referenced against `reference/MASTER_SUMMARY.md`, which — notably —
  already flags this result's own limitations candidly (CTGAN-only, not yet
  verified on SMOTE/original, hyperparameters not grid-searched)

### Architecture, in brief (full detail in Day 33's V1-V3 methodology notes)

1. **XGBoost feature selection**: `XGBClassifier` fit on the full preprocessed
   feature set, top-`n_qubits` features selected by importance — replaces
   V2's PCA/LDA compression with direct feature selection.
2. **Local-kernel measurement**: instead of a single global state-overlap
   (standard fidelity kernel), the circuit computes an average over pairwise
   qubit "both zero" probabilities — a different, cheaper-to-estimate
   similarity measure.
3. **Trainable kernel via KTA**: a small set of rotation weights, applied
   between the forward and adjoint embedding halves of the kernel circuit,
   optimized via kernel-target alignment (COBYLA, maximizing alignment
   between the kernel matrix and the label-similarity matrix on a small
   subset) before the kernel is used for the full landmark-based QSVM.

### Contents

- `code/pipeline_v3_full.py`, `code/ablation.py` — full implementation
- `notebooks/QSVM_v3_CTGAN_LocalKernel_XGBoost.ipynb` — original working notebook
- `reference/QSVM_v3_Architecture_Technical_Report.md` — detailed architecture writeup
- `reference/Proof_and_Seminar_Report_87pct_Result.md` — the seminar-facing summary
- `model/` — the actual trained model bundle + a load/predict example script
- `results/` — raw JSON for both the full grid and the ablation study

No new experiments were run for this day specifically — Day 31 (this
session's actual new V3 experimentation, running QGAN data through this same
pipeline) is where the new work happened. This day packages and verifies the
pre-existing V3 material so it stands alongside Day 31's results as
supplementary context.

### `comprehensive_study_classical_vs_quantum/` — Shanmukh's full uploaded study, included as-is

This is the complete, unmodified folder from Shanmukh's
`Comprehensive_Study_Classical_vs_Quantum.zip` upload — QSVM v1/v2/v3 and
VQC v1/v2/v3, all three datasets, 18 configs, with matched-feature classical
baselines and comparative analysis. Included here in full (not just excerpted
into other days) since it's squarely Version 3 / multi-version supplementary
material, matching this day's task.

**Important:** this study's own headline claims (documented in its
`comparative_results/COMPARATIVE_RESULTS.md`) were subsequently re-tested
with additional seeds in Day 31 and **partially overturned** — see
`../Day31_QGAN_QSVM_Pipeline/results/SEED_EXPANSION_STATISTICAL_CORRECTION.md`
for the corrected numbers, and the top-level `FINAL_FINDINGS_REPORT.md`
(package root) for the full explanation. This folder is kept exactly as
uploaded for provenance/traceability — read it alongside the correction, not
instead of it. A condensed, analysis-focused version of this same study
(with the verification steps and seed-expansion context built in) also lives
at `../Day31_QGAN_QSVM_Pipeline/comprehensive_study_SMOTE_addendum/`, used
there because that's where it was originally needed to close Day 31's open
SMOTE item — this folder here is the complete, original, unannotated version.
