# Day 5a — KTA Alignment Subset-Size Sensitivity — FINAL

**Read `Day5a_KTA_Subset_Sweep_FINAL_REPORT.md` first — it has the full
writeup, tables, and conclusions.** Everything else in this folder is
supporting code/data/figures for that report.

## Bottom line for the paper

- The original headline number (**0.875**, CTGAN, n=10) was produced by a
  circuit bug (`RY(w)` then `RY(-w)` cancelled to identity — the "trainable
  alignment" step never actually trained anything). **Use the corrected
  number instead: 0.800.**
- With the bug fixed, accuracy is flat across KTA subset size n∈{5,10,20,50}
  on both CTGAN (~0.78–0.80) and Original (~0.43–0.45) — consistent with
  "n=10 wasn't a lucky pick," but the effect size is within the statistical
  noise floor for a 40-point test set, so word it as suggestive, not confirmed.
- Three follow-up attempts to push accuracy higher (more Nyström landmarks,
  more KTA optimization iterations, a deeper circuit) all made results flat
  or worse — see the FINAL_REPORT's diagnostic section. **0.800 is the
  ceiling for this architecture as currently designed**, not a number that
  just needs more tuning/compute.
- Classical Random Forest beats QSVM on the identical feature set on CTGAN
  (0.875 vs 0.800) — worth stating plainly rather than implying a quantum
  advantage that isn't there in this configuration.

## Files

- `Day5a_KTA_Subset_Sweep_FINAL_REPORT.md` — the deliverable.
- `pipeline_v3_1_fixed.py` — corrected QSVM pipeline (both bugs fixed).
- `kta_subset_sweep_v3_1_fixed.py` — sweep script that produced the main table.
- `config.yaml` — exact seeds/settings used.
- `results/` — raw JSON: the 8-run sweep, plus the diagnostic-round runs
  (landmark scaling, optimization-budget/depth scaling).
- `figures/` — the three figures actually worth using in the paper:
  - `kta_subset_sweep_accuracy_v3_1_fixed.png` — main result (accuracy vs n).
  - `kta_old_vs_new_headline.png` — why 0.875 → 0.800 (bug correction).
  - `diagnostic_round_summary.png` — why more compute didn't help.

## Still open (not yet run)

Original-dataset sample-size scaling — does accuracy improve with more raw
samples pulled from the larger available pool (thousands per family vs. the
~500 used here)? This is the one lever not yet ruled out; worth running
before treating the Original-dataset ceiling (~0.43–0.45) as final.
