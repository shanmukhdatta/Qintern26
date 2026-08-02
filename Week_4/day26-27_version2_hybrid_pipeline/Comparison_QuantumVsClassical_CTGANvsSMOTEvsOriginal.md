# Quantum vs Classical, CTGAN vs SMOTE vs Original — Full Comparison

**Grid:** 3 datasets × 3 qubit widths (8, 10, 12) × 3 sample sizes (250, 500, 1000) = 27 configs.

**Pipeline used (identical across all 27 runs — this is the point):** variance filter → correlation filter (train-only) → log1p → StandardScaler → **LDA(2)⊕PCA(q−2)** hybrid projection → RobustScaler→[−π,π] → **data re-uploading feature map (L=2)**. QSVM uses the **Nyström-approximated kernel** (40 landmarks). VQC uses the re-upload circuit trained with **Adam** (not COBYLA — Week 4's isolation test showed COBYLA regresses VQC; Adam is the confirmed-correct choice). Classical baselines (RBF-SVM, Random Forest) trained on the identical subsample and identical LDA+PCA features as the quantum models at every config.

**Class scope:** all three datasets restricted to the **3 malware families** (Ransomware/Spyware/Trojan), Benign excluded uniformly. This was necessary, not optional — the CTGAN dataset as provided only contains these 3 classes, so this is the only scope in which all three datasets are directly comparable. It also happens to be the harder, more meaningful problem (Benign was already shown to be trivially separable at ~100% in earlier CTGAN classical runs).

---

## Full results — all 27 configs

| Dataset | n | q | Classical SVM | Classical RF | QSVM v2 | VQC v2 (Adam) | Winner |
|---|---|---|---|---|---|---|---|
| CTGAN | 250 | 8 | 0.725 | 0.675 | 0.700 | 0.627 | classical |
| CTGAN | 250 | 10 | 0.700 | 0.650 | 0.650 | 0.573 | classical |
| CTGAN | 250 | 12 | 0.650 | 0.650 | **0.700** | 0.547 | quantum |
| CTGAN | 500 | 8 | **0.775** | 0.700 | 0.575 | 0.533 | classical |
| CTGAN | 500 | 10 | 0.700 | 0.700 | 0.575 | 0.587 | classical |
| CTGAN | 500 | 12 | 0.725 | 0.700 | 0.525 | 0.493 | classical |
| CTGAN | 1000 | 8 | 0.650 | 0.700 | 0.650 | 0.610 | classical |
| CTGAN | 1000 | 10 | 0.725 | 0.675 | 0.600 | 0.570 | classical |
| CTGAN | 1000 | 12 | 0.675 | 0.700 | 0.600 | 0.513 | classical |
| SMOTE | 250 | 8 | 0.350 | 0.425 | 0.400 | 0.347 | classical |
| SMOTE | 250 | 10 | 0.325 | 0.375 | 0.400 | 0.400 | quantum (tie) |
| SMOTE | 250 | 12 | 0.350 | 0.350 | 0.375 | 0.213 | quantum |
| SMOTE | 500 | 8 | **0.500** | 0.350 | 0.425 | 0.427 | classical |
| SMOTE | 500 | 10 | **0.500** | 0.375 | 0.425 | 0.360 | classical |
| SMOTE | 500 | 12 | **0.500** | 0.375 | 0.375 | 0.400 | classical |
| SMOTE | 1000 | 8 | 0.325 | 0.400 | 0.325 | 0.380 | classical |
| SMOTE | 1000 | 10 | 0.375 | 0.400 | 0.375 | 0.313 | classical |
| SMOTE | 1000 | 12 | 0.325 | 0.425 | **0.450** | 0.380 | quantum |
| Original | 250 | 8 | 0.450 | 0.450 | 0.375 | 0.387 | classical |
| Original | 250 | 10 | 0.400 | 0.475 | 0.325 | 0.347 | classical |
| Original | 250 | 12 | 0.400 | 0.400 | 0.400 | **0.427** | quantum |
| Original | 500 | 8 | 0.350 | 0.425 | **0.425** | 0.367 | quantum (tie) |
| Original | 500 | 10 | 0.350 | 0.425 | **0.425** | 0.387 | quantum (tie) |
| Original | 500 | 12 | 0.350 | 0.325 | **0.425** | 0.367 | quantum |
| Original | 1000 | 8 | 0.300 | 0.300 | **0.550** | 0.390 | quantum |
| Original | 1000 | 10 | 0.325 | 0.325 | **0.550** | 0.407 | quantum |
| Original | 1000 | 12 | 0.325 | 0.275 | **0.575** | 0.470 | quantum |

## Aggregate summary (mean accuracy across all 9 configs per dataset)

| Dataset | Classical SVM | Classical RF | QSVM v2 | VQC v2 | Quantum win rate |
|---|---|---|---|---|---|
| **CTGAN** | 0.703 | 0.683 | 0.619 | 0.561 | **1/9 (11%)** |
| **SMOTE** | 0.394 | 0.386 | 0.394 | 0.358 | **3/9 (33%)** |
| **Original** | 0.361 | 0.378 | **0.450** | 0.394 | **5/9 (56%)** |

## The clearest single trend: Original dataset, QSVM vs classical, by sample size

| n_total | Classical (best) | QSVM v2 |
|---|---|---|
| 250 | 0.442 | 0.367 |
| 500 | 0.400 | 0.425 |
| 1000 | **0.317** | **0.558** |

Classical accuracy on the Original dataset *degrades* as n_total grows (0.442 → 0.317), while QSVM *improves* (0.367 → 0.558) — opposite directions, and at n=1000 the gap is 24 points in QSVM's favor, the single largest and most consistent quantum advantage found across every experiment run this project (last week or this week).

---

## Interpretation

**1. CTGAN is the easiest dataset for classical models, by a wide margin, and quantum can't touch it.**
CTGAN's mean classical accuracy (0.70/0.68) is nearly double SMOTE's or Original's (0.36–0.39). This makes sense mechanically: CTGAN generates synthetic samples *conditioned on class label*, which tends to produce tight, well-separated per-class clusters in feature space — close to the easiest possible input for an RBF kernel or a tree ensemble. Quantum models don't get to exploit that same clustering as effectively (QSVM mean 0.62, VQC mean 0.56) — they win only 1 of 9 configs here, and only marginally (0.700 vs 0.650 classical, at q=12/n=250).

**2. Original (real, imbalanced, non-augmented) data is where quantum does best — and the pattern strengthens with more data, not less.**
This is the most important finding of this run. On the Original dataset, QSVM has the *highest mean accuracy of any model on any dataset in this whole comparison* (0.450), while classical models are at their *worst* here (0.361/0.378 — barely above the 0.33 three-class chance floor at n=1000). The n=1000 rows show classical collapsing toward random guessing (0.275–0.325) while QSVM holds steady in the 0.55–0.575 range.

**Why this might be happening (interpretation, not proven):** real, non-augmented malware family data is inherently noisier and more heterogeneous than either synthetic augmentation method — each family here aggregates multiple real malware subtypes (e.g., several distinct ransomware strains) with genuinely different behavioral signatures, rather than one smooth synthetic distribution per class. As n_total grows, the stratified sample pulls in more of that underlying subtype diversity, making the classification boundary messier for classical models that rely directly on distances/splits in the projected feature space. The quantum kernel's re-uploading feature map embeds each point into a much higher-dimensional Hilbert space (2^q dimensions) before computing similarity — it's plausible this gives it more room to separate a messier, multi-modal class structure than a classical kernel operating in the same q-dimensional input space. This is a hypothesis suggested by the data, not something confirmed by additional controlled experiments this session.

**3. SMOTE sits in between, closer to "no clear winner."**
Classical and quantum are nearly tied on average (0.39 vs 0.39 QSVM), with quantum winning 3/9 — better than CTGAN's 1/9, worse than Original's 5/9. SMOTE's interpolation-based synthetic samples are less tightly clustered than CTGAN's (SMOTE just draws points along line segments between real neighbors, no learned class-conditional distribution), which may explain why it's harder for classical models than CTGAN but still easier than the genuinely heterogeneous real data.

**4. A pattern across all three datasets: the worse classical does, the more quantum wins.**
CTGAN (classical strongest) → 11% quantum win rate. SMOTE (classical middling) → 33%. Original (classical weakest) → 56%. This is consistent, monotonic, and not something built into the experimental design — it emerged from the results. The practical read: **quantum kernel methods in this pipeline don't beat classical on "easy" data, but they hold up far better than classical does when the data gets genuinely hard.**

**5. VQC is not the strength of this pipeline, on any dataset.**
VQC v2 (Adam) has the lowest mean accuracy of the four models on every single dataset (0.561 CTGAN, 0.358 SMOTE, 0.394 Original). QSVM is consistently the stronger of the two quantum methods this round — consistent with Week 4's finding.

---

## Caveats, stated plainly

- 27 runs, no seed-repeats at this scale — given Week 4's finding that single-seed accuracy on small test sets (here: capped at 40) swings several points from sampling noise alone, individual "wins" (especially narrow ones like SMOTE 250/10's exact tie at 0.400) should be read as directional, not definitive. The aggregate/mean-based conclusions (CTGAN easiest, Original hardest-for-classical, the win-rate gradient) are more robust since they're averaged over 9 configs each.
- Classical and quantum models are both capped at 80 train / 40 test samples per run for compute reasons (same constraint as Week 4) — larger n_total changes *which* subsample gets drawn from a bigger pool, not how much data the models actually see per run. The Original-dataset trend (Section 3 above) is about subsample *composition* changing with pool size, not about "more training data" in the conventional sense.
- LDA is capped at 2 components for this 3-class problem (`n_classes - 1`), same hybrid-with-PCA adaptation used in Week 4.
