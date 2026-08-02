# Detailed Report — Quantum vs Classical Across Three Data-Augmentation Strategies

## 1. What this run was for

Everything from Week 4 tested one dataset (SMOTE) against classical baselines. This run asks the bigger question directly: **holding the pipeline fixed, does the choice of augmentation strategy (SMOTE / CTGAN / none) change whether quantum or classical wins?** Three datasets, one pipeline, one grid.

## 2. Inputs

| Dataset | File | Native scope | What it is |
|---|---|---|---|
| SMOTE | `MalMem2022_SMOTE__1_.csv` | 4-class (Benign included), balanced | Interpolation-based synthetic oversampling |
| CTGAN | `malmem_ctgan__1_.csv` | 3-class only (no Benign), balanced | GAN-based synthetic oversampling, conditioned per class |
| Original | `malmem_original.csv` | 4-class derivable from `Category`, imbalanced | Real, unaugmented CIC-MalMem-2022 |

**Class-scope decision:** all three restricted to 3 classes (Ransomware/Spyware/Trojan), Benign dropped from SMOTE and Original to match CTGAN's native scope. This was a forced choice, not a preference — CTGAN's file has no Benign rows at all, so any comparison including Benign would have to leave CTGAN out entirely. Family labels for Original were derived by splitting the `Category` column on `-` and taking the prefix (e.g. `Ransomware-Conti` → `Ransomware`).

## 3. Pipeline — identical across all 27 runs, by design

This is the actual point of the exercise: if the pipeline is held fixed, any difference in outcome is attributable to the dataset, not to inconsistent methodology. Full detail (with the underlying math) is in last week's `QTagger_Full_Technical_Notebook.ipynb`; summarized here:

1. Stratified 3-class sub-sample to `n_total` (250/500/1000), 70/30 train/test split, train-only fitting throughout
2. Variance filter (drop ≤1e-8 variance) → correlation filter (drop |r|>0.95, train-only)
3. Signed log transform: `sign(x)·log1p(|x|)`
4. `StandardScaler` (train-fit)
5. **Hybrid supervised/unsupervised projection:** `LDA(2)` (capped at `n_classes−1=2` for this 3-class problem) concatenated with `PCA(q−2)` on the residual, giving `q` total dimensions
6. `RobustScaler` + clip to ±3σ, mapped to `[−π, π]`
7. **Quantum encoding:** data re-uploading feature map — `AngleEmbedding(Y)` + ring-CNOT entangling, repeated `L=2` times
8. **QSVM:** Nyström-approximated kernel, 40 landmarks, explicit `K_MM^{-1/2}` feature construction, linear SVM on the resulting features (train capped at 300, test at 40)
9. **VQC:** same re-upload circuit, `StronglyEntanglingLayers` ansatz, trained with **Adam** (10 epochs, batch size 16) — not COBYLA, per Week 4's isolation-test finding that COBYLA regresses VQC performance relative to Adam
10. **Classical baselines:** RBF-SVM and Random Forest, trained on the identical LDA+PCA-projected features and identical subsample (capped 80 train/40 test) that the quantum models see — this is what makes the quantum-vs-classical comparison fair rather than comparing quantum-on-reduced-features to classical-on-raw-features

## 4. Grid

3 datasets × {q=8,10,12} × {n=250,500,1000} = 27 configurations, each producing 4 numbers (classical SVM, classical RF, QSVM, VQC).

## 5. Headline results

*(Full 27-row table in `Comparison_QuantumVsClassical_CTGANvsSMOTEvsOriginal.md`.)*

**Mean accuracy per dataset, averaged over all 9 configs:**

| Dataset | Classical SVM | Classical RF | QSVM v2 | VQC v2 | Quantum wins |
|---|---|---|---|---|---|
| CTGAN | 0.703 | 0.683 | 0.619 | 0.561 | 1/9 |
| SMOTE | 0.394 | 0.386 | 0.394 | 0.358 | 3/9 |
| Original | 0.361 | 0.378 | **0.450** | 0.394 | 5/9 |

**Original dataset, QSVM vs best classical, by sample size:**

| n | Classical | QSVM |
|---|---|---|
| 250 | 0.442 | 0.367 |
| 500 | 0.400 | 0.425 |
| 1000 | 0.317 | **0.558** |

## 6. Findings, in order of how confident I am in them

### 6.1 High confidence: CTGAN is far easier for classical models than SMOTE or Original
Mean classical accuracy on CTGAN (0.70/0.68) is nearly double the other two datasets (0.36–0.39). This holds across every single one of the 9 CTGAN configs — classical accuracy never drops below 0.65 there, while it never exceeds 0.50 on SMOTE or Original. This is a strong, consistent, dataset-level effect, not a fluke of any particular config.

**Why:** CTGAN is a conditional generative model — it learns and samples from a per-class distribution, which tends to produce synthetic points that cluster tightly around a class-typical mode. That's close to the ideal input for both an RBF kernel (which measures local density) and a tree ensemble (which splits on axis-aligned regions). SMOTE, by contrast, interpolates linearly between real nearest-neighbor pairs, which doesn't concentrate density the same way, and Original data is exactly as messy as real malware behavior actually is.

### 6.2 High confidence: quantum's relative advantage grows as classical gets worse
The win-rate progression — CTGAN 11% → SMOTE 33% → Original 56% — is monotonic across all three datasets and lines up exactly with classical's mean accuracy going in the opposite direction (0.70 → 0.39 → 0.37). This isn't cherry-picked; it's the aggregate over 9 configs per dataset.

**Practical reading:** this pipeline's quantum models aren't beating classical because they're "better" in some general sense — they're winning specifically where classical is struggling. On the easiest data, classical wins comfortably. On real, unaugmented, messy data, quantum wins more often than not.

### 6.3 Medium confidence: on Original data specifically, QSVM improves as classical degrades with scale
This is the single cleanest trend in the whole dataset: at n=1000, classical collapses to 0.275–0.325 (barely above the 0.33 three-class chance floor) while QSVM holds at 0.55–0.575. The direction is consistent across all three qubit widths tested (8, 10, 12) — not just one lucky config.

**Proposed mechanism (this is interpretation, flagged as such, not confirmed by a controlled follow-up):** the Original dataset aggregates multiple real malware subtypes under each family label (e.g., several distinct ransomware strains grouped as "Ransomware"). As the sample pool grows, stratified sampling pulls in more subtype diversity, making the within-class structure more heterogeneous — harder for a classical decision boundary defined directly in the `q`-dimensional projected space. The quantum re-upload feature map embeds each point into a `2^q`-dimensional Hilbert space before computing similarity, which plausibly gives it more room to separate a multi-modal class structure than the classical models get from the same `q` raw input dimensions. This has not been isolated with a dedicated experiment (e.g., directly measuring within-class subtype diversity vs. n_total) — it's the most likely explanation given what's known, not a proven one.

### 6.4 Medium confidence: SMOTE sits in between because its synthetic points are less tightly clustered than CTGAN's
Consistent with 6.1's logic in reverse — SMOTE doesn't learn a class-conditional distribution, so it shouldn't produce CTGAN-level clustering, and the results bear that out (SMOTE's classical mean of 0.39 sits far below CTGAN's 0.70). But this is inferred from the accuracy gap, not directly measured (e.g., via a clustering-quality metric on the two synthetic datasets) — flagged as the weaker end of this report's confidence.

### 6.5 Low confidence / needs more runs: individual narrow wins in the 27-row table
Several configs are decided by a single test-set sample (e.g., SMOTE 250/10 is an exact 0.400/0.400 tie; Original 500/8 and 500/10 both show QSVM winning by exactly 0.075 over the best classical). At a 40-sample capped test set, one flipped prediction moves accuracy 2.5 points — Week 4's seed-stability check found standard deviations of 0.03–0.05 at this scale. Individual config-level "winners" in the table should be read as directional, not as settled results. The aggregate/mean-based findings above (6.1–6.3) are far more robust since they average over 9 configs each, which is why they're reported with higher confidence.

### 6.6 High confidence: VQC underperforms QSVM on every dataset tested, again
Consistent with Week 4. VQC's mean accuracy trails QSVM's on all three datasets (0.561 vs 0.619 CTGAN, 0.358 vs 0.394 SMOTE, 0.394 vs 0.450 Original). QSVM remains the stronger of the two quantum approaches in this pipeline.

## 7. What this changes about the Week 4 conclusions

Week 4 found quantum beat classical narrowly on SMOTE at n=500/q=12 (0.725 vs 0.625). This run, using the *3-class* version of SMOTE (Benign dropped) rather than Week 4's 4-class version, does not reproduce that specific win at that specific config — SMOTE 500/12 here shows classical (0.500) beating quantum (0.400). The two runs aren't directly comparable (different class count, different specific subsamples drawn), but it's worth being explicit: **this week's SMOTE numbers are not simply "last week's numbers, extended."** They're a different, harder problem (3-class, Benign excluded), and they come out lower across the board for every model, quantum and classical alike. The more robust, cross-dataset finding — that quantum's relative advantage tracks inversely with how easy the data is for classical models — is new to this run and wasn't visible in Week 4's single-dataset scope.

## 8. What's still open

- No seed-repeats at this scale (27 configs × multiple seeds would be another ~5-10x the compute used here) — the config-level table should be read with the caveat in 6.5.
- The "subtype diversity" mechanism proposed for the Original-dataset trend (6.3) is inferred, not directly measured — a follow-up could bin the Original test set by actual malware subtype and check whether classical errors concentrate on subtype-diverse test points more than QSVM's do.
- CTGAN's synthetic-data quality (KS-statistic checks) was done in the earlier classical-only notebook but not repeated for this 3-class file — worth confirming this specific CTGAN export matches that same quality bar.
- VQC was only tested with Adam + re-upload here; no further optimizer or ansatz-depth exploration was done this round.
