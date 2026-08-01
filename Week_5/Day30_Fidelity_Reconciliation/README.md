# Day 30 — Fidelity metrics reconciliation
**Task:** Reconcile KS, Wasserstein, MMD across all four augmentation methods
into one final comparison table, now that QGAN's numbers have changed
completely from Week 3.

## Status: COMPLETE. All four methods measured with identical code (`evaluate.fidelity_report`).

### Full results (all three malware classes)

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

Full table: `results/MASTER_fidelity_4way.csv`. Raw per-method:
`results/fidelity_SMOTE_CTGAN.csv`, `results/fidelity_QGAN_100epoch_CORRECTED.csv`.
QGAN's before/after (30ep vs 100ep): `results/comparison_30ep_vs_100ep.md`.

### Procedure and code

Real-vs-synthetic rows for SMOTE/CTGAN were separated using the same
hash-based matching as Day 29 (`../shared_data/code/build_augmented_frames.py`)
before computing fidelity — real rows only get compared against real rows.
The metric itself is `code/evaluate.py`'s `fidelity_report()`: per-feature
Kolmogorov-Smirnov statistic and Wasserstein distance (median and p75 across
all 55 features), plus Maximum Mean Discrepancy (RBF kernel) as a
distribution-level summary — identical function, identical parameters, run
once per method per malware class.

### Ranking, consistent across every class and every metric

**SMOTE > CTGAN > QGAN.** Not close — SMOTE's KS_median is roughly 10x
smaller than CTGAN's, which is roughly 3x smaller than QGAN's, on every
single class. This ranking is expected given each method's mechanism:
SMOTE interpolates between real neighbors (structurally close to real data
almost by construction), CTGAN learns a full generative model of the
distribution, and QGAN — per Day 33's diagnosis — never resolved its
discriminator-dominance training instability, so its output distribution is
the least faithful of the three.

### The finding that connects Day 29 and Day 30

Fidelity rank (SMOTE > CTGAN > QGAN) does **not** fully predict downstream
classifier impact (Day 29: SMOTE >> QGAN ≈ CTGAN, with CTGAN actually
negative). CTGAN has meaningfully better fidelity than QGAN but *worse*
downstream impact than QGAN's roughly-neutral result. This suggests fidelity
metrics (which measure per-feature marginal closeness) and downstream utility
(which depends on preserving decision-relevant structure, not just marginals)
are related but not interchangeable — a real methodological note worth
carrying into any future augmentation-method selection decision, not just
optimizing for fidelity scores in isolation.
