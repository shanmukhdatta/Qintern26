# Paired Significance Testing: QSVM v3 vs. Classical SVM

Replaces the win-rate counts in the multi-seed correction table with formal
paired statistics. Same test batch scored by both models at each seed (the
RNG subsampling in both `run_qsvm_v3` and `run_classical` draws the identical
40-point test set given the same seed and dataset), so this is a genuine
paired design, not two independent seed sweeps.

## Original (unaugmented) — extended from 3 to 8 seeds

| Seed | QSVM v3 | Classical SVM | Diff |
|---|---|---|---|
| 0 | 0.675 | 0.525 | +0.150 |
| 1 | 0.650 | 0.600 | +0.050 |
| 2 | 0.675 | 0.425 | +0.250 |
| 3 | 0.575 | 0.425 | +0.150 |
| 4 | 0.650 | 0.525 | +0.125 |
| 5 | 0.475 | 0.425 | +0.050 |
| 6 | 0.600 | 0.450 | +0.150 |
| 7 | 0.550 | 0.600 | **−0.050** |

**QSVM: 0.606 ± 0.070. Classical: 0.497 ± 0.076.**
**Mean advantage: +10.9pp, 95% CI [+3.4pp, +18.5pp]. Cohen's d = 1.21 (large).**
**Paired t-test: t=3.42, p=0.011. Wilcoxon signed-rank: p=0.016.**
**Win rate: 7/8 (87.5%)** — down from the original 3-in-3 claim, but now
backed by a confidence interval and two significance tests instead of a raw
count. Both tests agree the advantage is real at the conventional α=0.05
threshold, on 8 independent seeds.

## CTGAN — 8 seeds, for comparison

| Seed | QSVM v3 | Classical SVM | Diff |
|---|---|---|---|
| 0 | 0.750 | 0.675 | +0.075 |
| 1 | 0.875 | 0.725 | +0.150 |
| 2 | 0.575 | 0.525 | +0.050 |
| 3 | 0.650 | 0.625 | +0.025 |
| 4 | 0.675 | 0.775 | −0.100 |
| 5 | 0.650 | 0.700 | −0.050 |
| 6 | 0.775 | 0.650 | +0.125 |
| 7 | 0.875 | 0.675 | +0.200 |

**QSVM: 0.728 ± 0.110. Classical: 0.669 ± 0.074.**
**Mean advantage: +5.9pp, 95% CI [−2.5pp, +14.4pp] — crosses zero.**
**Paired t-test: p=0.140. Wilcoxon: p=0.164. Neither reaches significance.**
**Win rate: 6/8 (75%).**

## What this means for the paper

**Original now has what it needs for a formal claim.** "3-in-3 win rate" was
never a statistical claim — three data points can't support one. At n=8, the
paired t-test and Wilcoxon test agree (p=0.011, p=0.016), the 95% CI on the
mean advantage excludes zero by a comfortable margin (+3.4 to +18.5 points),
and the effect size is large (d=1.21). This is the number to lead with: *"QSVM
v3 outperforms a classical RBF-SVM baseline on the Original dataset by a mean
of 10.9 percentage points (95% CI [3.4, 18.5], paired t-test p=0.011,
Wilcoxon p=0.016, n=8 seeds)."* That sentence will hold up in review; "3-in-3"
would not have.

**One seed (7) lost on Original** even though it was the standout seed for
CTGAN's headline number. Worth stating explicitly in the paper: it's a useful,
honest data point that argues against reading anything into any single seed,
CTGAN or Original, in isolation — the aggregate is what carries the claim now,
not any individual run.

**CTGAN still doesn't clear significance at n=8** (p=0.14 / p=0.16, CI crosses
zero). This matches Week 5's original caution and now has a formal test behind
it rather than just a win-rate description — CTGAN is not a dataset this paper
should lean on for a significance claim, and now there's a citable reason why.

## Files
- `code/run_paired.py` — paired-testing runner (t-test, Wilcoxon, CI, Cohen's d).
- `results/paired_original.json`, `results/paired_ctgan.json` — raw per-seed pairs.
- `figures/paired_seed_comparison.png` — QSVM vs. classical, per seed, both datasets.
