## Week 4 Progress — Notion Notes

### Day 22 — Confirm CTGAN 87% config against Day 5 table
**Done.**
Traced the 87% to its exact source: Random Forest, accuracy 0.8725, trained on the full-scale CTGAN retrain (no row cap, 320 epochs per minority class, balanced up to the real Benign count), on all 55 raw features, evaluated on a 100% real held-out test set. It's a classical result, not a quantum one — I'd mixed that up earlier, so this also corrected the framing of what I originally needed to cross-check.

Also pulled the CTGAN-track QSVM result while I was in there: 69% accuracy at n=1000, Nyström kernel, q=8, 50 landmarks. Classical RBF-SVM on the same subsample got 71%. Catch: that run only covers 3 of 4 classes (Benign wasn't included), so it wasn't a clean comparison against my SMOTE numbers at the time — resolved later once I standardized every dataset to the same 3-class scope.

### Day 23 — QSVM + VQC on SMOTE data, matched scales (n=200, n=1000)
**Done, went further than the original scope.**
Ran the pipeline (variance filter → correlation filter, train-only → log1p → StandardScaler → PCA → angle encoding), tried 3 qubit widths (4/6/8) at both scales, kept the best per scale. Best: n=200 → VQC 51.7%, n=1000 → VQC 44.7%.

Checked these against classical (RBF-SVM, RF) on the identical subsample and classical beat both quantum models by 8-17 points. So the honest read at that point: quantum beat random chance, but not classical.

Went back to the literature after that — pulled 4 papers, adapted 4 techniques (supervised LDA projection, deeper "data re-uploading" feature map, Nyström-approximated kernel, gradient-free optimizer for VQC). Rebuilt as Version 2.

### Day 24–25 — Extend qubit-capacity analysis beyond 12 qubits
**Partially done.**
Fixed-budget sweep at q=4 and q=12: output-score variance dropped 3.4x while accuracy barely moved and F1 fell — PCA preserved more information (0.80→0.99 explained variance) but the circuit didn't use the extra room. Capacity-limited, not fidelity-limited.

Tried q=16, didn't finish — simulator cost is exponential in qubit count, hit a wall around 12-14 qubits. Would need a GPU-backed simulator to go further.

### Day 26–27 — Three-way SMOTE vs CTGAN vs original comparison
**Done, once all three datasets were in hand.**
Version 2 (LDA+PCA hybrid, re-upload feature map, Nyström kernel, Adam not COBYLA for VQC) beat classical outright on SMOTE at n=500 (67.5%/72.5% vs classical's 45-62.5%).

Once the CTGAN and Original CSVs were uploaded, ran the full 27-config three-way comparison (q=8/10/12 × n=250/500/1000, all three datasets restricted to the same 3-class scope). Finding: CTGAN is easiest for classical models by a wide margin (mean 0.70 accuracy), quantum's relative advantage grows as the data gets harder — only wins 11% of CTGAN configs but 56% of configs on the real, unaugmented Original data. Sharpest single result: QSVM beats classical by 24 points on Original at n=1000 (0.558 vs 0.317).

Updated QGAN numbers were never provided this week — that specific cross-check is still open.

### Day 28 — Write-up
**Done for my own pipeline; cross-team sync still open.**
Wrote up the full week's findings — every method, why, the math, all results — in a technical report and notebooks. Didn't sync with Team C this week; no input came from their side on shared conventions.

### Beyond the assigned scope this week
Kept pushing QSVM after Version 2 closed the classical gap on SMOTE. Built Version 3: tested XGBoost feature selection head-to-head against the LDA+PCA hybrid (XGBoost won clearly — mean 0.837 vs 0.716, beat classical in 9/9 configs vs 6/9), added a trainable kernel (kernel-target alignment) and switched from a global to a local kernel measurement to fix kernel concentration — a documented issue where global similarity measurements lose discriminative power as qubit count rises.

Best result of the whole project: **0.875 accuracy**, beating classical by up to 20 points on CTGAN. Ran an ablation to check what actually caused the improvement rather than assume: the local-kernel change did roughly 5x more of the work than the trainable-kernel change alone.

---

## Blockers going into next week
1. Updated QGAN numbers still not received — Day 26-27 cross-check incomplete without them.
2. No sync with Team C yet on shared PCA/qubit-budget conventions.
3. Version 3 has only been tested on CTGAN — not yet run on SMOTE or the Original dataset.
4. Qubit-count simulation still walled at ~12-14 qubits — need a faster/GPU simulator to push further.
5. Version 3's hyperparameters (landmark count, alignment subset size, qubit-pairing topology) weren't swept — room to improve further before calling this final.
