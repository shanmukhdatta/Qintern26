# Day 30 — Fidelity Metrics Reconciliation

## Objective
Reconcile KS (Kolmogorov-Smirnov), Wasserstein distance, and MMD (Maximum Mean Discrepancy) across all four augmentation methods (Original, SMOTE, CTGAN, QGAN) into one final comparison table, now that QGAN's numbers have changed completely since Week 3's broken run.


## Tasks Performed
- Separated real rows from synthetic rows for SMOTE and CTGAN (same
  hash-based matching method built for Day 29, reused here so both days are
  directly comparable).
- Computed KS statistic, Wasserstein distance, and MMD for every method,
  per malware class (Ransomware, Spyware, Trojan), using one identical
  scoring function (`code/evaluate.py`'s `fidelity_report()`) for all four
  methods — same parameters, same code, run once per method per class.
- Compared QGAN's 30-epoch (broken) numbers against its new 100-epoch
  (corrected) numbers.
- Consolidated everything into one master table.

## Methodology
Fidelity here means: *how statistically close does the synthetic data's
shape look to the real data's shape, feature by feature?*
- **KS statistic**: per feature, compares the real data's histogram shape to
  the synthetic data's histogram shape. Lower = more similar.
- **Wasserstein distance**: measures how much "work" it would take to
  reshape the synthetic distribution into the real distribution. Lower =
  more similar.
- **MMD**: a single distribution-level score (not per-feature) using an RBF
  kernel; 0 = identical distributions.
All three metrics were computed per feature (55 features), then summarized
as median and 75th percentile, and run through the exact same function for
every method — so any difference between methods reflects the augmentation
method itself, not a difference in how it was measured.

## Results
| Method | Class | KS_median | Wasserstein_median | MMD |
|---|---|---|---|---|
| SMOTE | Ransomware | **0.012** | **0.073** | **0.001** |
| CTGAN | Ransomware | 0.091 | 0.251 | 0.049 |
| QGAN | Ransomware | 0.328 | 0.664 | 0.163 |
| SMOTE | Spyware | **0.010** | **0.084** | **0.001** |
| CTGAN | Spyware | 0.133 | 0.284 | 0.005 |
| QGAN | Spyware | *(no synthetic generated — real count already met target)* | | |
| SMOTE | Trojan | **0.013** | **0.061** | **0.001** |
| CTGAN | Trojan | 0.112 | 0.143 | 0.081 |
| QGAN | Trojan | 0.293 | 0.871 | 0.342 |

Full table: `results/MASTER_fidelity_4way.csv`. QGAN's before/after
(30-epoch broken vs. 100-epoch corrected): `results/comparison_30ep_vs_100ep.md`.

**Consistent ranking, every class, every metric: SMOTE > CTGAN > QGAN.**
Not close — SMOTE's KS score is roughly 10x smaller than CTGAN's, which is
roughly 3x smaller than QGAN's, across the board.

## Interpretation (in plain language)
- **SMOTE looks the most "real" almost by construction.** It builds new
  rows by interpolating directly between two real neighboring rows — so it
  can't help but land close to the real data's shape.
- **CTGAN looks "real" too, but slightly less so.** It learns a genuine
  generative model of the data rather than just interpolating, which is a
  harder task — the result is noticeably good but not as tight a match as
  SMOTE.
- **QGAN looks the least "real" of the three.** Its generator never fully
  stabilized during training (root-caused with actual loss-curve evidence in
  Day 33 — a capacity mismatch between a shallow 4-layer quantum circuit and
  a much more capable classical discriminator), so its synthetic rows are the
  furthest from matching real data's shape on every single metric tested.
- **The finding that connects Day 29 and Day 30:** fidelity rank (SMOTE >
  CTGAN > QGAN) does **not** fully predict downstream classifier impact
  (Day 29 found SMOTE >> QGAN ≈ CTGAN, with CTGAN actually *negative*).
  CTGAN has meaningfully better fidelity than QGAN, yet worse downstream
  impact than QGAN's roughly-neutral result. In simple terms: looking
  statistically similar to real data (fidelity) is related to, but not the
  same as, actually helping a classifier learn (downstream usefulness) —
  CTGAN's synthetic rows are shaped close to real data on a feature-by-feature
  basis, but something about how those features relate to each other (the
  decision-relevant structure) still confuses a classifier trained on the
  blend.

## Conclusion
- Fidelity ranking is clean and fully consistent: **SMOTE > CTGAN > QGAN**,
  on every class and every metric measured.
- This ranking does not, by itself, predict which method actually helps a
  downstream classifier — that has to be measured separately (done in Day 29).
- QGAN's poor fidelity is consistent with, and explained by, its training
  instability documented in Day 33 — this is not a measurement error, it is
  the expected downstream consequence of that instability.
- Decision: treat fidelity and downstream classifier impact as two separate,
  necessary checks for any future augmentation method — neither one alone is
  a reliable stand-in for the other.
