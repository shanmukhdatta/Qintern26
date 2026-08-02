# Week 4 — Master Results Table

Every result from this week's work, one place. "Classes" column matters — 3-class and 4-class numbers are not directly comparable (see notes).

## SMOTE track — Day 23, pipeline v1

Pipeline: variance filter → correlation filter (train-only) → log1p → StandardScaler → PCA(=q) → MinMax[0,π] → single-pass AngleEmbedding (no entanglement)

| Dataset | Model | n | q | Accuracy | F1 macro | Classes |
|---|---|---|---|---|---|---|
| SMOTE | QSVM | 200 | 4 | 0.425 | 0.332 | 4 |
| SMOTE | QSVM | 200 | 6 | 0.475 | 0.426 | 4 |
| SMOTE | QSVM | 200 | 8 | 0.500 | 0.463 | 4 |
| SMOTE | QSVM | 1000 | 4 | 0.375 | 0.355 | 4 |
| SMOTE | QSVM | 1000 | 6 | 0.350 | 0.330 | 4 |
| SMOTE | QSVM | 1000 | 8 | 0.350 | 0.330 | 4 |
| SMOTE | VQC | 200 | 4 | **0.517** | 0.461 | 4 |
| SMOTE | VQC | 200 | 6 | 0.467 | 0.450 | 4 |
| SMOTE | VQC | 200 | 8 | 0.483 | 0.379 | 4 |
| SMOTE | VQC | 1000 | 4 | **0.447** | 0.446 | 4 |
| SMOTE | VQC | 1000 | 6 | 0.437 | 0.401 | 4 |
| SMOTE | VQC | 1000 | 8 | 0.397 | 0.353 | 4 |

## SMOTE track — capacity sweep (fixed compute budget: n=200, 100 train, 10 epochs)

| Dataset | Model | q | Accuracy | F1 macro | Output-score variance | Classes |
|---|---|---|---|---|---|---|
| SMOTE | VQC | 4 | 0.383 | 0.376 | 0.0509 | 4 |
| SMOTE | VQC | 12 | 0.417 | 0.313 | 0.0148 | 4 |
| SMOTE | VQC | 16 | *did not finish — simulator wall* | — | — | 4 |

## SMOTE track — classical baseline + reverification (n=200, q=4, identical 80/40 subsample)

Pipeline: same v1 pipeline, classical models trained on the same PCA-reduced features

| Dataset | Model | Accuracy | F1 macro | Classes | Notes |
|---|---|---|---|---|---|
| SMOTE | Dummy (stratified) | 0.400 | 0.395 | 4 | noise floor at n=40 test |
| SMOTE | Classical RBF-SVM | **0.600** | 0.559 | 4 | beats both quantum models |
| SMOTE | Classical Random Forest | **0.600** | 0.578 | 4 | beats both quantum models |
| SMOTE | QSVM (v1) | 0.425 | 0.332 | 4 | seed-mean 0.492±0.047 across 3 seeds |
| SMOTE | VQC (v1) | 0.517 | 0.461 | 4 | seed-mean 0.472±0.034 across 3 seeds |

## SMOTE track — grid extension, pipeline v1 (q=8/12 × n=250/500/1000), vs classical

Pipeline: same v1 pipeline; classical baselines on identical subsample/features at each config

| n | q | Dummy | Classical SVM | Classical RF | QSVM v1 | VQC v1 | Winner |
|---|---|---|---|---|---|---|---|
| 250 | 8 | 0.350 | 0.575 | 0.650 | 0.525 | 0.427 | classical |
| 250 | 12 | 0.350 | 0.575 | 0.650 | 0.425 | 0.453 | classical |
| 500 | 8 | 0.275 | 0.450 | 0.575 | 0.525 | 0.287 | classical |
| 500 | 12 | 0.275 | 0.450 | 0.625 | 0.475 | 0.407 | classical |
| 1000 | 8 | 0.200 | 0.475 | 0.450 | 0.350 | 0.480 | quantum (narrow) |
| 1000 | 12 | 0.200 | 0.475 | 0.425 | 0.350 | 0.477 | quantum (narrow) |

All 4-class. Classical wins 4/6.

## SMOTE track — pipeline v2 (literature-informed: LDA+PCA hybrid, re-upload feature map, Nyström QSVM, COBYLA VQC)

Pipeline: variance filter → correlation filter → log1p → StandardScaler → **LDA(3)⊕PCA(q−3)** → RobustScaler→[−π,π] → **re-upload AngleEmbedding+ring-CNOT (L=2)**

| Dataset | Model | n | q | Accuracy | F1 macro | Classes | vs classical (from grid above) |
|---|---|---|---|---|---|---|---|
| SMOTE | QSVM v2 (Nyström) | 250 | 8 | 0.575 | 0.566 | 4 | ties SVM (0.575), loses RF |
| SMOTE | QSVM v2 (Nyström) | 250 | 12 | 0.525 | 0.517 | 4 | loses both |
| SMOTE | QSVM v2 (Nyström) | 500 | 8 | **0.675** | 0.651 | 4 | **beats both** (0.450/0.575) |
| SMOTE | QSVM v2 (Nyström) | 500 | 12 | **0.725** | 0.691 | 4 | **beats both** (0.450/0.625) |
| SMOTE | QSVM v2 (Nyström) | 1000 | 8 | 0.450 | 0.487 | 4 | ties RF |
| SMOTE | QSVM v2 (Nyström) | 1000 | 12 | 0.450 | 0.452 | 4 | beats RF |
| SMOTE | VQC v2 (COBYLA) | 250 | 8 | 0.413 | 0.402 | 4 | below v1 |
| SMOTE | VQC v2 (COBYLA) | 250 | 12 | 0.413 | 0.382 | 4 | below v1 |
| SMOTE | VQC v2 (COBYLA) | 500 | 8 | 0.413 | 0.406 | 4 | above v1 |
| SMOTE | VQC v2 (COBYLA) | 500 | 12 | 0.327 | 0.289 | 4 | below v1 |
| SMOTE | VQC v2 (COBYLA) | 1000 | 8 | 0.447 | 0.432 | 4 | below v1 |
| SMOTE | VQC v2 (COBYLA) | 1000 | 12 | 0.283 | 0.217 | 4 | well below v1 |

**QSVM v2 beat QSVM v1 in all 6 configs — the single strongest, most reproducible quantum result of the week.** VQC v2 mixed to worse.

## SMOTE track — optimizer isolation test (n=500, q=12)

| Model configuration | Accuracy | F1 macro | Classes |
|---|---|---|---|
| VQC v1 (single-pass encoding, Adam) | 0.407 | 0.291 | 4 |
| VQC v2 (re-upload encoding, COBYLA) | 0.327 | 0.289 | 4 |
| VQC re-upload encoding + **Adam** (isolation test) | **0.433** | 0.398 | 4 |

Confirms: the re-upload feature map helps VQC; COBYLA was the regression, not the map.

## CTGAN track — classical (full-scale retrain)

Pipeline: raw 55 features, no PCA, no scaling reduction — full CTGAN-balanced training set (23,438/class), tested on a 100% real held-out set

| Dataset | Model | Accuracy | F1 macro | Classes | Notes |
|---|---|---|---|---|---|
| CTGAN (full-scale) | Random Forest | **0.8725** | 0.8082 | 4 | this is the "87%" |
| CTGAN (full-scale) | XGBoost | 0.8487 | 0.7725 | 4 | |
| CTGAN (full-scale) | LightGBM | **0.8795** | **0.8183** | 4 | actual best model in that run |

## CTGAN track — QSVM (Nyström, n=1000, q=8, 50 landmarks)

| Dataset | Model | Accuracy | F1 macro | Classes | Notes |
|---|---|---|---|---|---|
| CTGAN | QSVM (Nyström) | 0.69 | 0.6829 | **3 (no Benign)** | not directly comparable to 4-class rows above |
| CTGAN | Classical RBF-SVM (same subsample) | 0.71 | 0.7118 | 3 (no Benign) | closest quantum-classical gap all week (2 pts) |

## What's still missing for the full three-way comparison

| Arm | Status |
|---|---|
| SMOTE (augmented) | ✅ done, extensive (v1 + v2, multiple scales/qubits) |
| CTGAN (augmented) | ✅ done for classical + one QSVM point; VQC on CTGAN not run |
| **Original (no augmentation)** | ❌ not run by either track — this is the missing piece |
| QGAN | ❌ numbers not available |
| Reconciled class count (3 vs 4) | ❌ CTGAN QSVM excludes Benign; SMOTE track includes it |
