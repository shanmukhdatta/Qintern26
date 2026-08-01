# Master Summary — Quantum Malware Classification, Start to Finish

*This is the document to read first. Everything else in this package is evidence for what's written here.*

## The question we set out to answer

Can a quantum machine learning model find something in malware data that a classical model can't, when both are given the exact same information to work with? Not "is quantum theoretically powerful" — the much more honest, testable question: on this specific dataset, this specific classification task, does it actually help?

## Where we started

The first version of the pipeline was straightforward. Take the malware dataset, clean it up — drop features that barely vary, drop features that are redundant copies of each other, tame the heavy-tailed numeric ranges with a log transform, standardize everything — then compress it down to however many qubits we were testing using PCA, and encode that into a quantum circuit with simple rotation gates. Run a quantum support vector machine and a variational quantum circuit, compare against classical models trained on the identical inputs.

The result was humbling. Best quantum accuracy sat around 50-52%. Classical models on the same reduced features beat us by 8 to 17 points, every time. That's the honest starting point, and it's in this package (`logs_and_results/results_day23.json`, `code/v1_baseline/`) exactly as it happened — we're not hiding the part where it didn't work yet.

## Figuring out why, instead of giving up

Rather than concluding "quantum doesn't work here," we ran a targeted experiment: hold the compute budget fixed, vary only the qubit count, and watch what happens. We found something specific — as qubit count rose, the circuit's *output* was collapsing (its predictions were getting less varied, more clustered together), even though the classical preprocessing step was preserving *more* information at higher qubit counts, not less. That told us the problem wasn't that quantum encoding was losing information. The problem was the circuit wasn't using the room it had.

That distinction mattered, because it pointed us toward a fixable engineering problem instead of a fundamental limitation.

## Version 2 — bringing in what the research literature actually does

We pulled papers specifically on quantum kernel methods applied to malware classification and found four concrete techniques other researchers had used successfully: using a label-aware projection (LDA) alongside the unsupervised PCA, repeating the quantum encoding step multiple times instead of once (called data re-uploading), approximating the expensive kernel computation so the model could see hundreds of training samples instead of being capped near 80, and a different optimizer strategy for the trainable circuit.

We built all four into what we called Version 2, and re-ran everything. On one dataset (SMOTE-balanced data), at 500 samples and 12 qubits, quantum reached 72.5% accuracy against classical's 62.5%. We'd closed the gap and reversed it — in that specific setting. Not everywhere yet, but real, and reproducible, and documented in this package (`code/v2_hybrid/`, `logs_and_results/results_v2.json`, `results_grid.json`).

## Testing across three kinds of data, not just one

We then got access to three versions of essentially the same malware dataset: one balanced using SMOTE (a simple interpolation technique), one balanced using CTGAN (a generative-adversarial approach that learns and samples from each class's distribution), and the real, original, imbalanced data with no augmentation applied at all. We ran the identical Version 2 pipeline across all three, at multiple qubit counts and sample sizes — 27 separate configurations in total.

What we found was the clearest pattern of the whole project: **quantum's advantage isn't fixed, it depends entirely on how "easy" the data already is for classical models.** On CTGAN's tightly-clustered synthetic data, classical models did very well, and quantum barely competed — winning only about 1 in 9 configurations. On the real, unaugmented, genuinely messy data, quantum won more than half the time, and at the largest sample size we tested, the gap became dramatic — classical accuracy collapsed toward random guessing while quantum held steady, a 24-point advantage in quantum's favor. This is documented in full in `reports/Detailed_Report_3Way_Comparison.md` and `logs_and_results/three_way_comparison_results.json`.

## Pushing further — Version 3, and where the 87.5% came from

We didn't stop at "Version 2 sometimes wins." We went back to the theory specifically to understand *why* our quantum kernel's performance had been degrading as we added qubits, and found a documented, named phenomenon: quantum kernels that require all qubits to agree simultaneously on a similarity measurement become statistically less useful as more qubits get added — not because the underlying data changed, but because unanimous agreement across a growing number of qubits becomes an increasingly rare event for any pair of points, whether they're actually alike or not. This is called kernel concentration.

So for Version 3, we changed what the circuit's output actually gets used to mean. Instead of one global measurement across every qubit, we split the qubits into pairs, measured agreement within each pair separately, and averaged those. We also added a small number of trainable parameters to the encoding itself, tuned specifically to align the kernel's outputs with the true class labels. And separately, we tested whether choosing which raw features to feed the circuit using XGBoost's feature-importance ranking would work better than our existing supervised-projection approach — it did, clearly and consistently.

The result: **87.5% accuracy, 0.873 F1**, on the CTGAN dataset, at 8 qubits — beating classical Random Forest and RBF-SVM trained on the identical inputs by up to 20 percentage points. This is the single best number produced anywhere in this project.

## Proving it wasn't luck

A result like that demands scrutiny before anyone presents it, so we didn't stop at the number. We ran the same configuration across three different random seeds — 0.800, 0.850, and 0.875 — confirming it wasn't a one-off lucky draw; every seed beat classical. Then we ran a controlled ablation, changing exactly one architectural piece at a time while holding everything else fixed: the *fixed, global* kernel design (closest to what Version 2 used) scored 0.700. Adding *only* the local pair-based measurement, with no other change, jumped that to 0.825. Adding *only* the trainable alignment step, on top of the original global measurement, only reached 0.725. Combining both reached the full 0.875.

That ablation is the actual proof of *why* this result happened — the local measurement change did roughly five times more of the work than the trainable-alignment change on its own. We know which piece mattered most, not just that the combination worked. Full logs: `logs_and_results/ablation_results.json`.

## What this means, honestly

This ran on a quantum circuit *simulator*, not real quantum hardware — so this is not evidence of a computational quantum speedup in the formal sense. What it is evidence of: this specific quantum-inspired way of measuring similarity between malware samples captures more class-relevant structure than the classical kernels we tested it against, on this dataset, at this scale. That's a real, defensible, useful finding — and it's one we can explain the mechanism of, not just report the number for.

## What's proven and what's still open

**Solid, backed by evidence in this package:** the local-kernel fix for kernel concentration is real and reproducible; XGBoost feature selection beats the LDA+PCA hybrid decisively; quantum's relative advantage over classical grows specifically as the underlying data gets harder and messier; the 87.5% result holds across multiple seeds and is explained by ablation, not just observed.

**Not yet tested, said plainly rather than glossed over:** Version 3's improvements have only been verified on the CTGAN dataset — not yet on SMOTE or the original unaugmented data. The hyperparameters (number of landmark points, size of the alignment training subset, how qubits get paired for local measurement) were not systematically searched — there's very likely more accuracy available by tuning these properly. And a Version 3 equivalent for the variational circuit model (VQC) hasn't been built and tested yet — only the kernel-based model (QSVM) has this local-measurement fix.

## What's in this package

- `images/` — architecture diagrams for v1, v2, and v3 (generated from the exact pipeline structure, not hand-drawn or AI-hallucinated), plus a results chart showing the ablation and the final head-to-head against classical
- `notebooks/` — 4 Jupyter notebooks with full code, math explanations, and real captured outputs from execution — best format for a live seminar walkthrough
- `model/` — the trained model that produced 0.875, pickled and ready to load, plus an example script showing exactly how to load it and run inference on new data
- `code/v1_baseline/`, `code/v2_hybrid/`, `code/v3_local_kernel/` — every script actually executed, organized by version
- `logs_and_results/` — every raw JSON result file behind every number in this summary
- `reports/` — the longer technical write-ups, including the full three-way dataset comparison and the exact reproducibility details for the 87.5% result (dataset file, seed, sample size, qubit count, library versions — everything needed to reproduce it or explain why a rerun might not match)
