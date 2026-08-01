# Does QGAN beat the 87.5% CTGAN result on V3 QSVM? — Honest answer: it's complicated, and the fair comparison says "not on raw accuracy, but yes on quantum-vs-classical margin"

**⚠️ SUPERSEDED: the "3/3 seeds, tied margin" conclusion in this document was
based on only 3 seeds (42/1/7) and does not hold once cross-checked against a
second script and expanded to 5 seeds — 2 of the 3 seeds here flip to a loss
under re-test, and the corrected 5-seed average margin is negative (-4.5pp),
not tied with CTGAN. See
`SEED_EXPANSION_STATISTICAL_CORRECTION.md` for the current, correct numbers.
Kept below for provenance/traceability only.**

## Two runs were done, because the first one wasn't a fair comparison

### Run 1 — "as-is" QGAN dataset (gap-fill only, ~2-5% synthetic)
Our QGAN pipeline was built to *fill the gap* up to the real majority class
count (10,020) — Ransomware needed 229 synthetic rows, Trojan 533, Spyware 0.
That's only 2-5% synthetic composition.

| Seed | classical SVM | classical RF | **QSVM v3** |
|---|---|---|---|
| 42 | 0.300 | 0.375 | 0.425 |
| 1 | 0.350 | 0.550 | 0.400 |
| 7 | 0.525 | 0.425 | 0.400 |

Compared to CTGAN's 0.800 / 0.850 / 0.875 at the same seeds — QGAN looked like
a clear loss. **But this comparison is confounded**, not clean: I checked the
actual `malmem_ctgan__1_.csv` composition (via the original package's own 3-way
comparison notebook) and it's **33-40% synthetic per class** (~15,000-16,000
rows/class, vs. our ~10,000 gap-filled). Comparing a lightly-augmented dataset
against a heavily-augmented one isn't a test of generator quality — it's a test
of how much synthetic data was added, full stop.

### Run 2 — matched-scale QGAN dataset (rebuilt to match CTGAN's actual composition)
Regenerated synthetic data from the trained checkpoints (sampling is cheap —
no retraining needed) to hit the **exact same per-class targets as the real
CTGAN file**: Ransomware 15,422 / Spyware 14,986 / Trojan 15,849 (this required
also training a Spyware generator from scratch, since the original gap-fill run
never needed one).

| Seed | classical SVM | classical RF | **QSVM v3** | synth_frac |
|---|---|---|---|---|
| 42 | 0.500 | 0.475 | **0.525** | 0.339 |
| 1 | 0.425 | 0.500 | **0.550** | 0.361 |
| 7 | 0.500 | 0.450 | **0.625** | 0.337 |

## The honest read

**On raw accuracy, CTGAN still wins** (0.80-0.875 vs 0.525-0.625). But there's
a documented reason this isn't really about generator quality: the original
package's own `QuantumVsClassical_3Way_Showcase.ipynb` already found that
**CTGAN produces the *easiest* dataset for every model, quantum included** —
"CTGAN generates synthetic samples conditioned on class label, producing tight
per-class clusters — close to ideal input for an RBF kernel or tree ensemble."
That same notebook found quantum's real edge shows up on the *original, messy,
unaugmented* data, and that CTGAN is actually the *worst* dataset variant for
quantum in their full grid (quantum won only 1/9 CTGAN configs there — the
xgboost/q8/seed7 config used for the "87.5%" headline number was a favorable
outlier, not representative of CTGAN as a whole).

**On the question that actually matters — does quantum beat classical? —
QGAN and CTGAN tell the *same* story:**

| Dataset | Seeds where QSVM beat best classical | Avg. QSVM margin over best classical |
|---|---|---|
| CTGAN (xgboost/q8) | 3/3 | +7.5 points |
| QGAN matched-scale (xgboost/q8) | 3/3 | +6.7 points |

Quantum wins in every seed on both datasets, by a statistically similar
margin. QGAN-augmented data doesn't inflate everyone's accuracy the way CTGAN's
tight synthetic clusters do — but it preserves the same real relative quantum
advantage, on data that's closer to the real, messy distribution the package's
own broader study says quantum is actually good at.

## So: does QGAN "win" or "lose"?

- **Loses on absolute accuracy** — but that's measuring dataset easiness, not
  a fair quantum-vs-quantum generator comparison, and the original team's own
  analysis says CTGAN's absolute numbers are inflated by unrealistically clean
  synthetic clusters.
- **Matches CTGAN on the quantum-over-classical margin** (~7pp either way, 3/3
  seed win rate either way) — which is the actual "does quantum GAN win"
  question, and the answer there is a genuine tie, not a loss.
- **Caveat**: n=3 seeds, ~40-sample test sets (2.5 points per single flipped
  prediction, matching the original package's own stated caveat on config-level
  noise) — read seed-level numbers as directional, the aggregate margin
  comparison as the more robust finding.
