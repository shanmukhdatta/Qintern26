# Week 4 — Task Completion & Blockers

## Day 22 — Confirm CTGAN 87% config against Day 5 table
**Done.** Traced 87% to Random Forest, accuracy 0.8725, full-scale CTGAN retrain (no row cap, 320 epochs/class), all 55 raw features, real held-out test set. Classical result, not quantum -- corrected that mix-up along the way. Also found the CTGAN-track QSVM number while confirming this: 69% at n=1000, Nystrom kernel, but 3-class only (Benign excluded) -- flagged as not directly comparable to 4-class SMOTE numbers.

## Day 23 — QSVM + VQC on SMOTE, matched scales (n=200, n=1000)
**Done, extended beyond scope.** v1 pipeline run at 3 qubit widths (4/6/8) x 2 scales. Best: VQC 51.7% (n=200), 44.7% (n=1000). Classical baseline beat both quantum models by 8-17 points on identical inputs -- went back to the literature after that and built Version 2 (see Day 26-27), which closed and then reversed that gap.

## Day 24-25 — Qubit-capacity analysis beyond 12 qubits
**Partially done.** q=4 vs q=12 fixed-budget sweep: output-score variance dropped 3.4x while accuracy barely moved -- capacity-limited signature. q=16 attempted, did not finish (simulator wall, exponential cost). Would need GPU-backed simulator to go further.

## Day 26-27 — SMOTE vs CTGAN vs original three-way comparison, cross-check QGAN numbers
**Mostly done.** Full 27-config three-way comparison completed (q=8/10/12 x n=250/500/1000) once all three datasets were uploaded. Finding: CTGAN easiest for classical models, quantum wins most (56% of configs) on real/unaugmented data, with the clearest single result being QSVM beating classical by 24 points at n=1000 on Original data. Updated QGAN numbers were never provided this week -- that cross-check remains open.

## Day 28 — Write up findings, sync with Team C on pipeline consistency
**Done for own pipeline; cross-team sync still open.** Full technical report and notebooks completed covering all math, code, and results. Pipeline consistency within own experiments (v1 -> v2 -> v3) is documented and consistent. Did not sync with Team C this week -- no input received from their side.

## Beyond assigned scope this week
Built Version 3: local trainable quantum kernel + XGBoost feature selection, on CTGAN. Ablation-confirmed the local kernel fix (not just alignment) is what solves the qubit-scaling degradation seen in v1/v2. Best result of the entire project: **0.875 accuracy**, beating classical by up to 20 points.

## Blockers going into next week
1. Updated QGAN numbers still not received -- Day 26-27 cross-check incomplete without them.
2. No sync with Team C on shared PCA/qubit-budget conventions yet.
3. Version 3 (local kernel + XGBoost) has only been tested on CTGAN -- not yet run on SMOTE or Original.
4. Qubit-count simulation still walled at ~12-14 qubits on current hardware -- need GPU-backed simulator to extend further.
5. Version 3's hyperparameters (landmark count, alignment subset size, pair-grouping topology) were not swept -- room to improve further before calling this final.
