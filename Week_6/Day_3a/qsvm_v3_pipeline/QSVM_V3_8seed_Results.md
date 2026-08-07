# QSVM V3 Pipeline — Fixed Circuit, 8-Seed Test (CTGAN)

## Verdict: seed 7 is favorable, not a fluke, and not unique

| Seed | Accuracy | Macro-F1 |
|---|---|---|
| 0 | 0.750 | 0.737 |
| 1 | **0.875** | 0.864 |
| 2 | 0.575 | 0.562 |
| 3 | 0.650 | 0.644 |
| 4 | 0.675 | 0.672 |
| 5 | 0.650 | 0.643 |
| 6 | 0.775 | 0.772 |
| **7** | **0.875** | 0.873 |

**Mean = 0.728, std = 0.103, range 0.575–0.875 (q=8, CTGAN, fixed circuit).**

Seed 7's 0.875 is the best result of the 8, sitting about 1.4 standard
deviations above the mean — that's a real, meaningful favorable-seed effect,
not something you should report as "the" accuracy without qualification. If
you'd drawn a random seed instead of 7, the expected result is closer to 73%,
not 87.5%.

But it's not an isolated fluke either: **seed 1 hits the exact same 0.875**,
independently. Two different seeds out of eight reaching the same top value is
a meaningfully different story than one lucky outlier — it suggests 0.875 is a
real *achievable* ceiling for this configuration, reached ~25% of the time
across arbitrary seeds, not a one-in-eight-million coincidence. The honest
characterization for the paper: **QSVM V3 on CTGAN achieves up to 87.5%
accuracy depending on seed, with a mean of 72.8% ± 10.3% across 8 seeds** — not
"QSVM achieves 87.5%." This is consistent with the project's own Week 5
finding that CTGAN results don't survive multi-seed correction, now confirmed
independently on the corrected circuit.

## What this doesn't tell you
This is CTGAN only, 8 seeds, one qubit count (8), one config (align
subset=10, XGBoost feature selection). It doesn't test whether Original shows
the same spread — worth running the same 8 seeds there if you want the number
that anchors your paper's actual claim, since Original is what Week 5 flagged
as the seed-robust dataset. I can run that next if useful.

## Files
- `code/pipeline_v3_full.py` — the V3 pipeline, circuit fix applied (trainable
  rotation applied once, between forward/adjoint halves — not cancelled).
- `code/data_original.py` — loader for the Original dataset, if you want to
  extend this test there.
- `code/run_seeds.py` — the multi-seed runner used here. Saves incrementally
  after each seed so a long run can be resumed.
- `results/qsvm_v3_ctgan_seeds.json` — raw per-seed results backing the table above.

## Reproduce / extend
```bash
pip install --break-system-packages pennylane pennylane-lightning xgboost scikit-learn scipy pandas numpy
cd code
# already-run seeds are skipped automatically
python3 run_seeds.py CTGAN 8 0 1 2 3 4 5 6 7
# to test Original instead:
python3 run_seeds.py Original 8 0 1 2 3 4 5 6 7
```
Each seed takes ~90–200s (kernel Gram construction dominates). Data
(`malmem_ctgan.csv`, `MalMem2022.csv`) must be placed in `data/`.
