# Week 3 Summary — July 22, 2026

## Objective
To package the CTGAN-v2 augmented memory malware dataset through the shared quantum preprocessing pipeline, benchmark Quantum Support Vector Machine (QSVM) and Variational Quantum Classifier (VQC) models against classical controls, evaluate the specific impact of data augmentation on quantum performance, and diagnose quantum capacity vs. data fidelity constraints.

## Tasks Completed
- **Day 1:** Packaged CTGAN-v2 dataset through the shared pipeline (variance filter dropping 11 near-constant features, 0.95 correlation filter, StandardScaler, PCA to 8-qubit budget); saved fitted transform.
- **Day 2:** Evaluated QSVM with exact fidelity kernel at $n=200$ (63.3% accuracy / 0.625 Macro-F1), outperforming a matched-sample classical control (60.0% accuracy / 0.592 Macro-F1).
- **Day 3:** Built and trained VQC (8 qubits, StronglyEntanglingLayers, Adam optimizer); resolved barren plateau issues by reducing circuit depth from 2 reps to 1 rep (43.3% accuracy / 0.409 Macro-F1 at $n=200$).
- **Day 4:** Scaled models to $n=1000$; implemented Nyström approximation (50 landmark points) for QSVM (69.0% accuracy / 0.683 Macro-F1) and evaluated VQC (43.0% accuracy / 0.360 Macro-F1).
- **Day 5:** Conducted three-way comparison (augmented vs. original imbalanced vs. classical matched control at $n=200$); demonstrated augmentation prevents class collapse (Trojan recall 0% → active prediction).
- **Day 6:** Performed capacity vs. fidelity diagnosis; demonstrated that qubit scaling (4 → 8 → 12 qubits) increases accuracy (42% → 58% → 68%) in direct alignment with retained PCA variance (45.6% → 54.6% → 62.0%).
- **Day 7:** Documented augmented-data quantum comparison write-up for joint publications and synchronized pipeline standards across teams.

## Challenges
- **Barren Plateaus in Deep VQC:** 2-rep StronglyEntanglingLayers suffered from vanishing gradients; resolved by reducing to 1 layer rep.
- **QSVM Kernel Scalability:** Exact $O(n^2)$ fidelity kernel matrix computation was intractable at $n=1000$; successfully approximated via 50-landmark Nyström method.
- **Information Loss via Qubit Compression:** 8-qubit PCA retains only 54.6% cumulative feature variance, imposing a strict capacity ceiling on quantum models.

## Results
- **Matched Quantum Advantage ($n=200$):** QSVM surpassed the classical control by **+3.3% accuracy** (**63.3% vs. 60.0%**) and **+0.033 Macro-F1** (**0.625 vs. 0.592**).
- **Scaled QSVM Performance ($n=1000$):** Nyström-QSVM achieved **69.0% accuracy** and **0.683 Macro-F1**.
- **Augmentation Lift on Quantum Models:** Augmentation provided a **+26.6% accuracy boost** to quantum models (36.7% → 63.3%), outperforming the +15.0% lift observed in classical models at matched sample size.
- **Capacity Ceiling Diagnosis:** Qubit scaling (4-qubit: 42%, 8-qubit: 58%, 12-qubit: 68%) confirmed underperformance stems from qubit qubit/variance capacity constraints rather than synthetic sample fidelity.

## Next Week's Plan
- Test advanced feature selection strategies to retain higher cumulative variance within limited qubit budgets (10–12 qubits).
- Address VQC Trojan class prediction bias through loss-weighting and alternative ansatz designs.
- Prepare combined quantum-augmentation findings for joint paper write-ups.
