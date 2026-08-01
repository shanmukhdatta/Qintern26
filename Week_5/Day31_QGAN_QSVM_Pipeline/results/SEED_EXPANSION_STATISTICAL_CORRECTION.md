# Statistical correction — expanded seed testing overturns the "QSVM v3 beats classical on all 3 datasets" headline

**This document corrects a claim made earlier in this package.** The earlier
Day 31/32 headline — "QSVM v3 is the only model/version that meets or beats
classical on all three datasets" — was based on 1-2 seeds per dataset (CTGAN:
seed 7 only; SMOTE: best-of-2, seed 1 chosen after seed 7 lost; Original:
seed 7 only). That headline does **not hold** once more seeds are tested.

## What was actually run

Using Shanmukh's own validated `../seed_expansion_code/quantum_all_versions.py` script (unmodified
logic, only a QGAN data-loading branch added), 2-5 additional seeds were run
per dataset for QSVM v3, using the identical `n_qubits=8, n_total=500`
protocol as every existing result in this package.

## Corrected results, with proper multi-seed statistics

| Dataset | Seeds tested | Mean accuracy ± std | Mean margin vs. best classical ± std | Win rate |
|---|---|---|---|---|
| CTGAN | 7, 100, 123 | 0.708 ± 0.170 | **-0.125 ± 0.127** | 1/3 |
| SMOTE | 7, 1, 100, 123 | 0.444 ± 0.094 | **-0.077 ± 0.099** | 1/4 |
| QGAN | 42, 1, 7, 100, 123 | 0.540 ± 0.049 | **-0.045 ± 0.049** | 1/5 |
| **Original** | 7, 100, 123 | 0.542 ± 0.031 | **+0.033 ± 0.024** | **3/3** |

Full per-seed numbers, both new runs and the original single/best-of-2-seed
runs merged: `SEED_EXPANSION_raw_results.csv`.

## What this actually means

**Only Original data shows a robust, consistent quantum win** — positive
margin in all 3 seeds tested, and by far the lowest variance (std=0.024 on
the margin, vs. 0.10-0.17 for the other three). This is a real, low-noise
signal.

**CTGAN, SMOTE, and QGAN all show a *net loss* on average** once more than
1-2 seeds are sampled, with enormous seed-to-seed variance — CTGAN's
accuracy alone swings from 0.875 (seed 7) to 0.475 (seed 123), a 40-point
range on the *same* dataset, config, and architecture. The original 0.875
and 0.600 headlines were real, verified numbers (not fabricated) — but they
were favorable draws from a high-variance distribution, not a stable,
reproducible advantage. This is exactly the failure mode small-n testing
predicts, and exactly why this correction was worth doing before this
package went in front of a mentor.

## Why this doesn't contradict the rest of the project — it confirms it

This result is consistent with, and strengthens, the throughline that runs
through the entire project (V1's capacity-limited finding, V2's 27-config
grid, Day 33's stabilization diagnosis): **quantum shows its most reliable
edge on harder, messier, less-processed data — not on synthetic/augmented
data**, regardless of which generator produced it. The V2 grid found this
independently (Original: 56% win rate; CTGAN: 11%). This seed-expanded V3
result finds the same pattern a second, independent way: Original 3/3,
everything synthetic 1/3-1/5. Two different pipeline versions, two different
seed sets, same conclusion — that's a genuine convergent finding, not
circular reasoning.

## What changes in the rest of this package

- Day 31 and Day 32's "QSVM v3 wins everywhere" framing is **superseded by
  this document**. Read this one for the corrected claim.
- The original single-seed numbers (CTGAN 0.875, SMOTE 0.600, Original 0.500)
  are **not wrong or fabricated** — they're real runs, kept in the package for
  provenance — they're just not representative on their own, which is now
  disclosed with the statistics to prove it rather than asserted in prose.
- **QGAN now has verified V1/V2/V3 coverage** (previously V3-only), closing
  that specific gap. V1 loses badly (-21.7pp), V2 loses (-8.3pp), V3's
  multi-seed margin is -4.5pp on average — QGAN doesn't outperform classical
  reliably at any architecture version, consistent with Day 33's diagnosis
  that its generator never fully stabilized.

## What still isn't fully resolved

5 seeds (QGAN) and 3-4 seeds (others) is a real improvement over 1-2, but
still not enough for tight confidence intervals — the margin std values above
(0.02-0.13) mean a 95% CI would still be wide at this sample size. If this
needs to hold up to a statistics-focused reviewer, 10+ seeds per dataset
would be the next step. Given the direction of the finding (net loss for 3/4
datasets) is now consistent across multiple independent seeds and matches
the V2 grid's independent finding, more seeds are more likely to sharpen
this conclusion than reverse it — but that's a reasoned expectation, not a
guarantee.
