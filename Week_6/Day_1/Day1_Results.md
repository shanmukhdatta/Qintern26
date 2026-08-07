# Day 1 — Kernel-Concentration Diagnostics: Results

## Paper-ready summary

Local-pair measurement reduces kernel concentration relative to the global
fidelity kernel on the CTGAN-augmented data (Gram off-diagonal std 0.159 vs.
0.129) but shows no measurable difference on the Original (unaugmented) data
(0.280 vs. 0.281); on both datasets the local kernel's alignment-with-labels
signal above a permutation chance floor is lower than the global kernel's
(Original: 0.155 vs. 0.182; CTGAN: 0.050 vs. 0.097). Kernel concentration and
label-discriminative signal do not move together under local measurement —
locality reduces concentration on CTGAN specifically, while the KTA-above-floor
metric favors the global kernel on both datasets tested.

## Method

Per-qubit trainable rotation `RY(w_i)` is applied once, between the forward
encoding of `x1` and the adjoint decoding of `x2`, so it participates directly
in the fidelity overlap; `w` is optimized via gradient-free KTA maximization
(COBYLA) on a 10-point subset of the training fold. Local kernel: qubits
grouped into non-overlapping pairs, `P(00)` measured per pair and averaged.
Global kernel: full all-qubits-agree fidelity (`w=0`, no alignment). 8 qubits,
seed 7, XGBoost feature selection, `n_total=500`, held-out test batch (n=40,
identical subsample for v2/v3 via the same RNG sequence).
`kta_score(K, y) = <K, Y>_F / (||K||_F ||Y||_F)`, `Y_ij = +1` if same class
else `-1`; chance floor = mean KTA over 100 label permutations.

## Results table

| Model | Dataset | Gram off-diag std | KTA | Chance floor | KTA − floor |
|---|---|---|---|---|---|
| v2 (global) | CTGAN | 0.1290 | 0.1018 | 0.0048 | **0.0970** |
| v3 (local)  | CTGAN | 0.1592 | −0.2148 | −0.2653 | **0.0505** |
| v2 (global) | Original | 0.2812 | −0.0185 | −0.2007 | **0.1822** |
| v3 (local)  | Original | 0.2803 | −0.2331 | −0.3881 | **0.1550** |

## Interpretation

**Gram off-diagonal std (concentration):** local measurement de-concentrates
the kernel on CTGAN (+23% relative to global) but shows no detectable effect
on Original — the dataset the paper's central claim depends on. This is a real
gap in the mechanistic story, not a minor caveat, and should be stated as such
rather than generalized from the CTGAN result alone.

**KTA − chance floor (signal above noise):** the global kernel shows higher
signal-above-floor than the local kernel on both datasets. Local (pair-
averaged) measurement discards some joint-qubit information the global
fidelity kernel retains, and on this diagnostic test batch that costs more
label-signal than the reduced concentration recovers.

**Net read:** local kernel measurement gives a real, dataset-dependent
concentration benefit, but does not come with a matching gain in
label-discriminative signal by this diagnostic. The accuracy improvement V3
shows over V2 is better attributed to concentration reduction alone than to
the kernel becoming more label-aligned — those are two different things, and
only one of them moved in V3's favor here.

## Files
- `code/diagnostics.py` — `gram_offdiag_std`, `kta_score`, `kta_chance_floor`.
- `code/run_diagnostics.py` — hooks into the v2/v3 kernel code, builds the Gram
  matrix on the shared test batch (no new training run).
- `results/day1_kernel_diagnostics.json` — raw output backing the table above.
- `figures/day1_kernel_diagnostics.png` — bar charts for both metrics.
