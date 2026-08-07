# Day 1 — Kernel-Concentration Diagnostics: Results

## ⚠️ Critical finding first (read before the requested table)

While building the diagnostics I found a bug in the trainable-alignment block shared
by `pipeline_v3_full.py::make_local_trainable_circuit` and `ablation.py::make_circuit`:

```python
for i in range(n_qubits): qml.RY(w[i], wires=i)
for i in range(n_qubits): qml.RY(-w[i], wires=i)
```

`RY(w)` followed immediately by `RY(-w)` on the same wire is the identity rotation
for *any* value of `w` (`RY(a)·RY(b) = RY(a+b)`, so `RY(-w)·RY(w) = RY(0) = I`).
I verified this numerically: across 20 random `(x1, x2, w)` triples, swapping in a
random `w` vs. `w = 0` changes the kernel value by at most `1.3e-17` (floating-point
noise). **The KTA-trained weights have zero effect on the forward-pass kernel.**
This is present in both the ablation script and the main V3 pipeline, so it also
underlies the existing 87.5%/72.5% ablation numbers — the accuracy deltas
attributed to "alignment" (A→C, B→D) are not caused by the kernel actually
changing; they come only from the extra `rng` draws the alignment step consumes,
which shift which train subsample / landmarks get selected downstream. This
doesn't invalidate the *local-vs-global* comparison (that part of the circuit is
unaffected), but it means the "trainable kernel alignment" mechanism, as
currently implemented, is not doing anything — the paper's claim that alignment
contributes ~5pp on top of the local-kernel effect needs to be re-examined.

I did **not** fix this in-place, since (a) it's out of scope for today's
few-hour diagnostics task, and (b) it would silently change the already-reported
87.5%/72.5%/etc. numbers without you deciding that's the right move first. I ran
today's diagnostics on the pipeline exactly as it currently behaves (per the
brief's instruction to reuse the existing kernel/KTA machinery), so "v3 (local)"
below reflects local-measurement-only, not local+alignment — because right now
those are the same thing. Suggested minimal fix, for when you want it: apply the
`RY(w)`/`RY(-w)` pair on either side of the CNOT-ring uncomputation instead of
back-to-back (e.g. fold `w` into the last embedding layer, or apply `RY(w)` after
the forward block and `RY(-w)` only *after* the adjoint block, not immediately
before it), then re-run `ablation.py` C/D to see if a real alignment effect exists.

---

## Paper-ready summary

Local-pair measurement reduces kernel concentration relative to the global
fidelity kernel on the CTGAN-augmented data (Gram off-diagonal std 0.160 vs.
0.129) but shows no measurable difference on the Original (unaugmented) data
(0.281 vs. 0.281); on both datasets the local kernel's alignment-with-labels
signal above a permutation chance floor is *lower*, not higher, than the global
kernel's (Original: 0.155 vs. 0.182; CTGAN: 0.051 vs. 0.097). This directly
contradicts the expected mechanistic narrative that local measurement should
simultaneously de-concentrate the kernel *and* increase label-discriminative
signal, and it does so most clearly on Original — the dataset that carries the
paper's actual surviving claim.

## Results table

Config: 8 qubits, seed 7, XGBoost feature selection (V3's winning projection),
`n_total=500`, held-out test batch (n=40, drawn identically for v2/v3 via the
same RNG sequence `ablation.run_variant` uses before its alignment branch).
KTA chance floor = mean KTA over 100 label permutations.

| Model | Dataset | Gram off-diag std | KTA | Chance floor | KTA − floor |
|---|---|---|---|---|---|
| v2 (global) | CTGAN | 0.1290 | 0.1018 | 0.0048 | **0.0970** |
| v3 (local)  | CTGAN | 0.1596 | −0.2145 | −0.2651 | **0.0507** |
| v2 (global) | Original | 0.2812 | −0.0185 | −0.2007 | **0.1822** |
| v3 (local)  | Original | 0.2809 | −0.2330 | −0.3879 | **0.1549** |

(Sanity check: config A and D of `ablation.py` were re-run standalone before
writing diagnostics and reproduced the reference table exactly — A: acc=0.700,
D: acc=0.875 — so the shared preprocessing/circuit code is being invoked
correctly.)

## Interpretation

**Gram off-diagonal std (concentration):** partially matches the expected
narrative. On CTGAN, local > global (+24% relative), consistent with local
measurement reducing kernel concentration. On Original, local ≈ global — no
detectable de-concentration effect. Since Original is the dataset the paper's
real headline result depends on, this is a meaningful gap in the mechanistic
story, not a minor caveat.

**KTA − chance floor (signal above noise):** does **not** match the expected
narrative on either dataset. The global kernel shows higher signal-above-floor
than the local kernel in both cases. Two things likely contribute: (1) the
alignment bug above means "v3" here is local-measurement-only, with no genuine
label-supervised adjustment; (2) local (pair-averaged) measurement, by
construction, throws away some of the joint-qubit information the global
fidelity kernel retains, and on this diagnostic test batch that costs more
label-signal than the reduced concentration recovers.

**Net read:** local kernel measurement gives a real, if dataset-dependent,
concentration benefit, but — at least with the alignment bug present — it does
not come with the "more discriminative kernel" story the paper currently tells.
The safest framing for the paper right now is: *local measurement mitigates
kernel concentration on CTGAN but not on Original, and improves accuracy through
a mechanism this diagnostic does not fully explain via KTA-above-floor; the
"trainable alignment" component of V3, as implemented, has no effect on the
kernel and should not be described as contributing to either result until the
circuit bug is fixed and re-evaluated.* This does not match the brief's
predicted "v3 higher on both metrics, both datasets" scenario, and per the task
rules I'm reporting that mismatch directly rather than reframing the numbers.

## Files
- `code/diagnostics.py` — the three required functions (`gram_offdiag_std`,
  `kta_score`, `kta_chance_floor`), reusing the project's existing KTA formula.
- `code/run_diagnostics.py` — hooks into the existing v2/v3 kernel/eval code,
  builds a true n×n Gram matrix on the shared test batch (no new training run).
- `results/day1_kernel_diagnostics.json` — raw output backing the table above.
- `figures/day1_kernel_diagnostics.png` — bar charts for both metrics.
