# Day 35 — Final consolidation (whole team)

**Task:** Package all notebooks, reports, and repo links; finalize the master
comparison tables and written methodology.

## Status: Packaged. Core pipeline complete. QGAN candidate fix identified (Step 1: no label smoothing) but unadopted in downstream tables (open candidate item).

### Package manifest

```
week_package/
├── README.md                              <- start here, top-level index
├── shared_data/                            raw + derived data for SMOTE/CTGAN/Original, reusable prep code
├── Day29_Downstream_Classifier_Comparison/  COMPLETE — all 4 methods, identical protocol
├── Day30_Fidelity_Reconciliation/           COMPLETE — all 4 methods, identical protocol
├── Day31_QGAN_QSVM_Pipeline/                COMPLETE — no blockers
├── Day32_Master_Comparison_Tables/          COMPLETE — no open items
├── Day33_Stabilization_Methodology/         COMPLETE — includes 2 failed-fix writeups + 4 follow-on experiments
├── Day34_V3_Supplementary/                  COMPLETE — verified against raw run data
├── Day35_Final_Consolidation/               this folder
└── notebooks_reference/                     original .ipynb files, shared context
```

### What closed since the last package version

Days 29 and 30 were blocked on the SMOTE and CTGAN raw dataset files — those
were uploaded and processed this round. Two things worth flagging about how
they were handled:

1. **Neither file has an explicit real/synthetic flag.** A plain merge
   against the original data to identify real rows exploded (this dataset has
   duplicate rows), so a hash-based duplicate-safe counting approach was used
   instead (`../shared_data/code/build_augmented_frames.py`), verified against
   known real class counts before trusting any downstream result.
2. **This surfaced a real finding, not just a technical fix**: SMOTE retains
   every real row exactly; CTGAN retains zero — every CTGAN row is synthetic.
   Worth knowing when interpreting either method's numbers.

Both days now report identical-protocol, identical-test-split results across
all four methods (Original/QGAN/SMOTE/CTGAN), cross-validated against each
other (all four methods' "original" baseline F1 numbers match to 6 decimal
places, confirming the shared test split is genuinely identical).

### The headline findings from completing 29/30

- **Fidelity ranks: SMOTE > CTGAN > QGAN**, consistently, every class, every metric.
- **Downstream impact ranks: SMOTE >> QGAN ≈ CTGAN (CTGAN slightly negative)**
  — a different order than fidelity. CTGAN's better fidelity than QGAN did
  *not* translate into better downstream impact; SMOTE's combination of high
  fidelity *and* high synthetic volume (~66-68% of training data, vs QGAN's
  <5%) is what actually appears to move classifier metrics.
- This means **fidelity metrics alone are not a reliable proxy for downstream
  usefulness** — a genuinely useful methodological finding for future
  augmentation-method decisions on this project, not just a filled-in table.

### What was still open, and how it closed

The V3-pipeline SMOTE run closed when Shanmukh uploaded a comprehensive study
(`Day31/comprehensive_study_SMOTE_addendum/`) covering QSVM v1/v2/v3 and VQC
v1/v2/v3 across all three datasets — 18 configs, far more than the single
SMOTE cell that was open. Independently verified before trusting it: the
study's CTGAN seed=7 result (0.875/0.873) matches this project's own
independently-run Day 31 CTGAN result to 4 decimal places — genuine
cross-validation, not circular, since the two runs happened in different
sessions.

**That verification led to a bigger correction.** Having independent
verification code (`quantum_all_versions.py`) in hand made it easy to test
whether the single-seed headlines generalized — they didn't. Expanding to
3-5 seeds per dataset found: **only Original data shows a robust,
low-variance quantum win (3/3 seeds). CTGAN, SMOTE, and QGAN all show a net
loss on average**, despite real, verified favorable single-seed numbers
(0.875, 0.600, 0.625) that turned out to be high-variance outliers, not
stable results. Full statistics:
`../Day31_QGAN_QSVM_Pipeline/results/SEED_EXPANSION_STATISTICAL_CORRECTION.md`.

This correction **strengthens rather than undermines** the project's
throughline: the same "quantum wins on harder/messier data, not on
synthetic/augmented data" pattern was now found three independent ways —
the V2 grid, the original V3 comparison, and the corrected multi-seed V3
results — different pipeline versions, different seed sets, same conclusion.

QGAN also now has verified V1/V2/V3 coverage (previously V3-only) — it loses
to classical at every architecture version.

### Follow-on QGAN Stabilization Round & Adoption Status

Following the original Day 33 write-up, a follow-on stabilization round evaluated four additional QGAN variants: (1) removing label smoothing (`config_v5_no_label_smoothing.py`), (2) shrinking discriminator hidden units (`config_v3_smaller_discriminator.py`), (3) widening TTUR learning rate ratios to 10x and 30x (`config_v4_ttur_wide_ratio.py`), and (4) per-layer generator learning rates (`train_qgan_v6_per_layer_lr.py`).

**Key Finding:** Removing label smoothing (Step 1) was the single fix that produced confirmed 100-epoch improvements across both malware classes, dropping `KS_median` from 0.3280 to **0.2533** for Ransomware and from 0.2930 to **0.2739** for Trojan. Other variants yielded mixed or incomplete results (and surfaced a non-fatal optimizer checkpoint attribute bug in Step 4).

**Project Adoption Status:** This improved candidate checkpoint has **not** been adopted or propagated into Days 29–32 downstream classifier metrics or master tables. Adopting this candidate for downstream pipelines remains an open candidate task.

### Consistency checks performed across days

- Day 31's finding (QGAN and CTGAN have statistically tied quantum-vs-classical
  margins, ~7pp either way) was cross-checked against the independently-derived
  V2 27-config grid (Day 32) and found consistent.
- Day 29/30's newly-completed baseline numbers (Original, no augmentation)
  match QGAN's originally-computed baseline to 6 decimal places — confirms the
  identical-test-split guarantee held across two separate work sessions.
- Day 30's fidelity ranking (SMOTE > CTGAN > QGAN) and Day 29's downstream
  ranking (SMOTE >> QGAN ≈ CTGAN) were compared directly rather than reported
  in isolation — the mismatch between the two rankings (Section 5 of Day 32's
  master table) is itself flagged as a finding, not smoothed over.

### Repo / file links

All code, data, checkpoints, logs, and notebooks referenced across every
day's README are included directly in this package (not external links).

