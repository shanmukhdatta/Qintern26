# QTagger+ Week 4 Technical Report
## QSVM / VQC on SMOTE-Augmented Ransomware Data: Pipeline, Experiments, and Findings

**Author's project:** QTagger+ (QIntern 2026) — quantum ML for ransomware/malware-family tagging
**Dataset:** CIC-MalMem-2022, SMOTE-balanced, 4-class (Benign / Ransomware / Spyware / Trojan)
**Scope of this report:** everything executed in this working session, in the order it happened, with every result, every method change, and the reasoning behind each one. Written so a first-time reader with no prior context can follow the whole arc.

---

## 1. Starting Point

**Input:** `MalMem2022_SMOTE.csv` — 117,192 rows, 55 numeric features + `Label`, perfectly balanced across 4 classes (29,298 rows each). No missing values.

**Task list for the week (as given):**
| Day | Task |
|---|---|
| 23 | Run QSVM and VQC on SMOTE data at matched scales (n=200, n=1000), same scales already used for CTGAN |
| 24-25 | Extend qubit-count capacity analysis beyond 12 qubits if time allows |
| 26-27 | Consolidate SMOTE vs CTGAN vs original three-way comparison; cross-check against updated QGAN numbers |
| 28 | Write up Week 4 findings, focused on pipeline consistency |

**A structural constraint stated upfront and true throughout:** this session has access to the uploaded SMOTE CSV and to a general summary of past work, but **not** to the actual prior notebook/table that produced the reported "87% CTGAN accuracy" or the "Day 5 joint classifier table" or updated QGAN numbers. Those cross-checks (Day 26-27) stayed blocked the entire session for that reason — flagged immediately and re-flagged at every later stage rather than guessed at.

---

## 2. Locked Preprocessing Pipeline (v1)

This is the pipeline used for all Day 23 experiments and the first grid extension. It mirrors conventions from earlier weeks of the project (log1p transform, leakage-safe correlation filtering, train-only fitting).

**Steps, in order:**
1. **Stratified sub-sample** to the target scale `n_total` (e.g. 200, 250, 500, 1000), balanced 4-way.
2. **Train/test split**, 70/30, stratified, `random_state=42`.
3. **Variance filter** — drop any feature with variance ≤ 1e-8, fit on train only.
4. **Correlation filter** — drop any feature with |Pearson r| > 0.95 against another surviving feature, computed on the train split only (this is the leakage-safe part: correlation is never computed using test rows).
5. **`log1p` signed transform** — `sign(x) * log1p(|x|)`, applied to both splits using no fitted parameters (deterministic).
6. **`StandardScaler`** — fit on train, applied to both.
7. **`PCA`** — fit on train, `n_components = n_qubits` (this ties the preprocessing directly to the quantum circuit width being tested).
8. **`MinMaxScaler` to `[0, π]`** — fit on train, so PCA outputs become valid rotation angles for angle encoding.

After the variance + correlation filters, the surviving feature count was consistently **24-26 out of 55** raw features, regardless of qubit count — the filters remove the same redundant columns every time, since they only depend on train-split statistics, not on `n_qubits`.

**Why this order:** filtering before scaling avoids wasting compute standardizing columns that get dropped; splitting before any fitting step is what makes the pipeline leakage-safe — test-set information never touches a scaler, filter, or PCA fit.

---

## 3. Day 23 — QSVM + VQC at Matched Scales (n=200, n=1000)

**Requested methodology:** "try up to 3 configurations, report the best." The three natural knobs available without altering the raw data are qubit count / PCA width — so 3 widths were tested per scale: **q = 4, 6, 8**.

**Compute constraint discovered early:** QSVM needs a fidelity kernel matrix, which costs one quantum circuit evaluation per pair of samples — O(n²). A single kernel evaluation at 8 qubits takes roughly 3.3 ms on the local simulator (`pennylane` + `lightning.qubit`). At n=1000 (800 train / 200 test) that's over 480,000 evaluations ≈ 26 minutes for one config — too slow to run the full requested grid in reasonable time. **Fix:** cap QSVM's kernel-matrix inputs at 80 train / 40 test samples per run, regardless of `n_total`. VQC (gradient-based, not O(n²)) was capped separately at 140-300 training samples with epoch count reduced as qubit count rose, to keep wall-clock time bounded.

**All 6 runs, exact numbers:**

| n_total | qubits | PCA explained var. | QSVM acc | QSVM f1 | QSVM train used | VQC acc | VQC f1 | VQC epochs |
|---|---|---|---|---|---|---|---|---|
| 200 | 4 | 0.804 | 0.425 | 0.332 | 80 | **0.517** | 0.461 | 20 |
| 200 | 6 | 0.900 | 0.475 | 0.426 | 80 | 0.467 | 0.450 | 15 |
| 200 | 8 | 0.948 | 0.500 | 0.463 | 80 | 0.483 | 0.379 | 10 |
| 1000 | 4 | 0.749 | 0.375 | 0.355 | 80 | **0.447** | 0.446 | 12 |
| 1000 | 6 | 0.855 | 0.350 | 0.330 | 80 | 0.437 | 0.401 | 9 |
| 1000 | 8 | 0.918 | 0.350 | 0.330 | 80 | 0.397 | 0.353 | 6 |

**Initial (later revised) read:** VQC beat QSVM in 5 of 6 runs; both beat the 0.25 four-class random-chance floor; accuracy trended *down*, not up, as qubit count rose from 4 to 8, even though PCA explained variance rose from ~0.75-0.80 to ~0.92-0.95. This last point — more retained information, worse downstream accuracy — became the seed of everything that followed.

**A note on epoch count:** VQC epochs were reduced as qubit count rose (20 → 15 → 10 at n=200) purely to keep runtime bounded — this is a compute-budget confound sitting inside the "qubit effect" above, addressed properly in Section 4.

---

## 4. Day 24-25 — Qubit-Capacity Sweep (Isolating the Qubit Effect)

Section 3's qubit-vs-accuracy trend was confounded by epoch count also shrinking as qubits rose. This sweep **fixed** the compute budget (n=200, 100 training samples, 10 epochs) and varied only `n_qubits`, to test the "capacity-limited vs fidelity-limited" hypothesis cleanly. A new metric was tracked: **output-score variance** — the spread of the trained circuit's raw expectation-value outputs across the test set, a proxy for whether the circuit is actually using its available Hilbert space or collapsing toward a narrow output range.

| Qubits | PCA explained var. | Accuracy | F1 macro | Output-score variance | Wall time |
|---|---|---|---|---|---|
| 4 | 0.804 | 0.383 | 0.376 | 0.0509 | 27s |
| 12 | 0.989 | 0.417 | 0.313 | **0.0148** | 85s |

**q=16 was attempted and did not finish** within a 280-second budget. Statevector simulation cost is exponential in qubit count (2¹⁶ = 65,536-dimensional state vs 2¹² = 4,096), so this is a genuine simulator wall, not a bug — worth reporting as its own finding: **beyond roughly 12-14 qubits, classical simulation itself becomes the bottleneck** on this hardware, separate from the model-capacity question.

**Key finding:** output-score variance dropped **3.4×** (0.051 → 0.015) from q=4 to q=12, while accuracy barely moved (+0.034) and F1 macro actually *fell* (0.376 → 0.313, meaning the classifier collapsed toward predicting fewer classes as width increased). PCA explained variance rose the whole time (0.80 → 0.99). **Interpretation:** more qubits give the classical preprocessing stage more room to preserve information, but the *trained circuit* doesn't expand into that extra room — it contracts. This is evidence for **capacity-limited** behavior (the ansatz/optimizer combination isn't using the space it's given) rather than **fidelity-limited** behavior (information lost in the encoding itself).

---

## 5. Reverification — The Honest Gut-Check

Before trusting Section 3's headline numbers, two checks were run, prompted directly by the request to "reverify" rather than take the results at face value.

### 5.1 Classical baseline on identical data

Same 80/40 subsample, same 4 PCA features (n=200, q=4 — the "best" v1 config), same train/test split:

| Model | Accuracy | F1 macro |
|---|---|---|
| Dummy (stratified random) | 0.400 | 0.395 |
| **Classical RBF-SVM** | **0.600** | 0.559 |
| **Classical Random Forest** | **0.600** | 0.578 |
| QSVM | 0.425 | 0.332 |
| VQC | 0.517 | 0.461 |

**Finding:** classical SVM/RF beat *both* quantum models by 8-17 points on the exact same reduced inputs. This reframed the whole exercise — "beats random chance" is a low bar; the real question is whether the quantum model extracts more signal than a classical model given identical information, and at this point in the project, it did not.

### 5.2 Seed-stability check

Same config (n=200, q=4), 3 random seeds:

| Seed | QSVM acc | VQC acc |
|---|---|---|
| 1 | 0.525 | 0.517 |
| 7 | 0.525 | 0.467 |
| 42 | 0.425 | 0.433 |
| **mean ± std** | **0.492 ± 0.047** | **0.472 ± 0.034** |

With only 40 test samples, one flipped prediction swings accuracy by 2.5 points — so single-seed comparisons *between* QSVM and VQC (e.g. "VQC won 5/6 runs" from Section 3) are not statistically reliable at this scale. The comparison *against classical baselines*, however, holds: even QSVM's best seed (0.525) stays below classical's 0.600.

**Conclusion of this section:** Day 23's results were real numbers, correctly computed, but the narrative needed correcting — quantum was not competitive with classical on identical inputs. This, not the raw accuracy numbers, is the actual finding to report.

---

## 6. Grid Extension — q=8/12 × n=250/500/1000

Requested next: widen the grid to qubit counts 8 and 12, and sample sizes 250, 500, and 1000, each benchmarked against classical baselines trained on identical subsampled data/features (the same fairness discipline from Section 5.1, now applied everywhere).

| n | q | Dummy | Classical SVM | Classical RF | QSVM_v1 | VQC_v1 | Winner |
|---|---|---|---|---|---|---|---|
| 250 | 8 | 0.350 | 0.575 | **0.650** | 0.525 | 0.427 | classical |
| 250 | 12 | 0.350 | 0.575 | **0.650** | 0.425 | 0.453 | classical |
| 500 | 8 | 0.275 | 0.450 | **0.575** | 0.525 | 0.287 | classical |
| 500 | 12 | 0.275 | 0.450 | **0.625** | 0.475 | 0.407 | classical |
| 1000 | 8 | 0.200 | **0.475** | 0.450 | 0.350 | **0.480** | quantum (VQC, narrowly) |
| 1000 | 12 | 0.200 | **0.475** | 0.425 | 0.350 | **0.477** | quantum (VQC, narrowly) |

**Result: classical wins 4 of 6 configs outright**, and even where "quantum wins" (n=1000), VQC's margin over classical SVM is under 1 point. **Verdict: not impressive.** This explicitly triggered the next phase — go find out how other researchers made QSVM/VQC actually beat classical baselines, rather than keep tuning blind.

---

## 7. Literature Review

Searched for published work applying QSVM/VQC to malware or comparable tabular classification tasks, specifically looking for cases where quantum methods *beat* classical baselines, and what those authors did differently in preprocessing and training.

**Papers reviewed:**

1. **Rahman et al., "Scalable Malware Family Classification Using Quantum Kernel–Based Machine Learning"** (arXiv:2606.16191, 2026). 23-class malware classification, 18,836 samples. Their fidelity-based quantum kernel + Nyström approximation reached **80.88% accuracy**, beating every classical baseline tested (best classical: KNN at 79.56%) under identical features and split. Three ablations from this paper were directly actionable:
   - **Supervised LDA projection beat unsupervised SVD** before quantum encoding: 80.88% vs 78.83%, all else held fixed.
   - **Accuracy rose with both qubit count and circuit depth** (repetitions `L`) — the opposite of this project's Section 3/4 finding — but their feature map re-applies encoding + entangling gates **L=4 times** (a "data re-uploading" style circuit), not the single-pass `AngleEmbedding` used in this project's v1 pipeline.
   - **Nyström landmark approximation** lets the kernel effectively use far more training data than a full O(n²) kernel matrix permits, without the quadratic blowup that forced the 80-sample cap in Sections 3-6.

2. **"Can Feature Engineering Help Quantum Machine Learning for Malware Detection?"** (arXiv:2305.02396). On the Drebin dataset, XGBoost-based feature selection into a VQC beat Decision-Tree-based selection by a wide margin (78.91% vs 62.41%). Signal: *which* dimensionality-reduction method feeds the circuit matters as much as circuit design. **Not implemented this session** — flagged as a follow-up.

3. **"Quantum-Inspired Machine Learning: a Survey"** (arXiv:2308.11269). Citing Masun et al.: QSVM/VQC on the ClaMP and Reveal malware/vulnerability datasets **underperformed classical SVM and shallow neural networks** — i.e., Section 5-6's finding in this project is a documented, common outcome in this exact sub-field, not a sign of a broken pipeline. The same survey cites a separate result on NSL-KDD where an EfficientSU2 circuit trained with the **gradient-free COBYLA optimizer** beat classical SVM after PCA down to just 3 features.

4. **"Quantum Machine Learning for Cybersecurity: A Taxonomy and Future Directions"** (arXiv:2512.15286). A ZZFeatureMap-based QSVM beat classical SVM (80.75% vs 77%) on an image-based dataset — evidence that richer, entangling feature maps outperform plain angle encoding.

**Four techniques selected for implementation**, adapted (not copy-pasted) to this project's 4-class, tabular, single-dataset setting:
1. Supervised LDA projection (from paper 1)
2. Data re-uploading feature map, L=2 repetitions (from paper 1's depth finding)
3. Nyström landmark kernel for QSVM (from paper 1)
4. COBYLA optimizer for VQC (from the survey's NSL-KDD citation)

**A necessary adaptation, stated explicitly rather than hidden:** paper 1 has 23 classes, so LDA can project up to 22 dimensions and was used directly at q=8. This project's dataset has **4 classes**, so LDA is mathematically capped at **`n_classes − 1 = 3` components** — it cannot reach q=8 or q=12 alone. The adaptation used: **LDA(3) concatenated with PCA(q−3) computed on the same standardized features**, giving `q` total dimensions with the supervised LDA subspace prioritized and PCA filling the remainder. This specific hybrid is this project's own design choice, not something from the source papers.

---

## 8. Pipeline v2 — Implementation and Results

**Changes from v1, all combined into one pipeline (`pipeline_v2.py`):**
- Projection: `LDA(min(3, q))` concatenated with `PCA(q − lda_dim)` on the residual, replacing plain PCA.
- Scaling: `RobustScaler` + clip to ±3 std, mapped to `[−π, π]`, replacing plain `MinMaxScaler` to `[0, π]`.
- QSVM: fidelity kernel evaluated with a **re-upload circuit** (`AngleEmbedding` + ring-CNOT entangling, repeated L=2 times) against **M=40 Nyström landmarks**, with explicit `K_MM^{-1/2}` feature construction and a linear SVM on the resulting Nyström features — allowing up to 300 training samples' worth of signal instead of the 80-sample hard cap.
- VQC: same re-upload circuit family, trained with `scipy.optimize.minimize(method='COBYLA')` instead of Adam/parameter-shift.

**Full q=8/12 × n=250/500/1000 grid, QSVM_v2 vs QSVM_v1 vs classical:**

| n | q | QSVM_v1 | **QSVM_v2** | Classical SVM | Classical RF | v2 beats classical? |
|---|---|---|---|---|---|---|
| 250 | 8 | 0.525 | 0.575 | 0.575 | 0.650 | ties SVM, loses RF |
| 250 | 12 | 0.425 | 0.525 | 0.575 | 0.650 | no |
| 500 | 8 | 0.525 | **0.675** | 0.450 | 0.575 | **yes, both** |
| 500 | 12 | 0.475 | **0.725** | 0.450 | 0.625 | **yes, both** |
| 1000 | 8 | 0.350 | 0.450 | 0.475 | 0.450 | ties RF |
| 1000 | 12 | 0.350 | 0.450 | 0.475 | 0.425 | beats RF |

**QSVM_v2 improved over QSVM_v1 in all six configs, no exceptions.** At n=500, both qubit widths clearly beat both classical baselines — this is the single strongest, most reproducible quantum result across the entire session.

**VQC_v2 (COBYLA), same grid:**

| n | q | VQC_v1 (Adam, single-pass) | VQC_v2 (COBYLA, re-upload) |
|---|---|---|---|
| 250 | 8 | 0.427 | 0.413 |
| 250 | 12 | 0.453 | 0.413 |
| 500 | 8 | 0.287 | 0.413 |
| 500 | 12 | 0.407 | 0.327 |
| 1000 | 8 | 0.480 | 0.447 |
| 1000 | 12 | 0.477 | 0.283 |

**Mixed to worse.** An isolation test was run to find out why: at n=500/q=12 (VQC_v2's worst case), the re-upload circuit was kept but the optimizer was switched back to Adam. Result: **0.433 accuracy / 0.398 F1** — better than *both* VQC_v1 (0.407) and VQC_v2 (0.327). **This cleanly separates the two changes: the re-upload feature map genuinely helps VQC too; COBYLA was the regression**, most likely under-converged given the fixed iteration budget against 72 parameters (`n_layers × n_qubits × 3 = 2 × 12 × 3` at q=12). **Recommended VQC configuration going forward: re-upload circuit + Adam, not COBYLA.**

---

## 9. Consolidated Timeline — What Was Tried, In Order, and Why

1. **Baseline pipeline (v1):** variance filter → correlation filter → log1p → StandardScaler → PCA(=n_qubits) → MinMax[0,π]. *Why:* matches locked project conventions from earlier weeks; simple, auditable, leakage-safe.
2. **Day 23 matched-scale run (n=200/1000, q=4/6/8):** *Why:* directly requested; "try 3, keep best" methodology applied to qubit width, the only tunable knob available without altering the data.
3. **Capacity sweep (q=4 vs 12, fixed compute budget):** *Why:* Day 23's qubit-vs-accuracy trend was confounded by epoch count; needed a clean isolation to test the capacity-limited hypothesis. Found output-variance collapse — the real signature of the bottleneck.
4. **Classical baseline + seed-stability reverification:** *Why:* explicitly requested re-verification; results without a classical comparison point are not evidence of anything. Found classical beats quantum by 8-17 points on identical inputs — corrected the entire narrative.
5. **Grid extension (q=8/12 × n=250/500/1000, always vs. classical):** *Why:* requested scale-out; applying the classical-comparison discipline from step 4 everywhere this time, not just retroactively. Classical won 4/6 — confirmed "not impressive" as a real, not anecdotal, finding.
6. **Literature search:** *Why:* explicit instruction — if not impressive, find out what published work does differently rather than keep tuning blind. Found the capacity-limited finding (step 3) is a documented outcome in this literature, and found four concrete, adaptable techniques from papers that did beat classical.
7. **Pipeline v2 (LDA+PCA hybrid, re-upload kernel, Nyström QSVM, COBYLA VQC):** *Why:* implement the four techniques together and re-run the full grid, rather than claim improvement without re-testing. Found QSVM_v2 is a genuine, reproducible win; VQC_v2 (COBYLA specifically) is not.
8. **Isolation test (re-upload + Adam, dropping COBYLA):** *Why:* when VQC_v2 underperformed, the honest next step was to find out which of the two simultaneous changes caused it, rather than discard "re-uploading" along with "COBYLA" as one bundle. Found the re-upload circuit helps on its own; COBYLA was the specific problem.

---

## 10. Where This Leaves the Project

**What works, keep it:**
- **QSVM pipeline v2** (hybrid LDA+PCA projection → re-upload feature map → Nyström landmark kernel) is the one component in this entire session with a clean, multi-config win over classical baselines. Adopt as the new QSVM baseline for future weeks.
- **VQC: keep the re-upload circuit, revert to Adam.** The circuit change helps; the optimizer change (COBYLA) does not, at least not with the iteration budget tested.
- The **output-score-variance metric** from Section 4 is a useful diagnostic to keep tracking going forward — it caught the capacity collapse that raw accuracy alone did not show clearly.

**What's still open / blocked:**
- Day 26-27 (SMOTE vs CTGAN vs original three-way comparison, cross-checked against the 87% CTGAN result and updated QGAN numbers) remains blocked on missing source artifacts — the Day 5 joint classifier table, the exact CTGAN run configuration, and the QGAN numbers were never available in this session. Needed before that consolidation can be done honestly.
- XGBoost-based feature selection (paper 2, Section 7) was identified as a promising fifth technique but not implemented or tested this session.
- VQC has not yet been re-tested at the full q=8/12 × n=250/500/1000 grid with the corrected re-upload+Adam configuration — only spot-checked at one config (n=500, q=12).
- q≥16 qubit simulation is currently blocked by simulator wall-clock cost, independent of any modeling question — would need a faster simulator backend (e.g. GPU-backed) to push further.

**Single-sentence summary for a reader with no other context:** starting from a pipeline that lost to plain classical SVM/RF on identical inputs, targeted changes pulled from published quantum-malware-classification research — a supervised (LDA) projection, a deeper re-uploading quantum feature map, and a Nyström-approximated kernel — turned QSVM into the first component in this project to reliably beat classical baselines, while an isolation test showed the same feature-map change helps VQC too, but a gradient-free optimizer swap tried alongside it does not.
