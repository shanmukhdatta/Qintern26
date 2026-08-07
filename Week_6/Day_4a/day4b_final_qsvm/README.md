# Day 4b — Hybrid Readout Stacking, Final (fixed circuit, QSVM-framed)

Re-run of Day 4b's readout-head comparison using the corrected circuit from
the Day 1 bugfix. **Read `Day4b_FINAL_QSVM_Readout_Report.md` first.**

## Setup
```bash
pip install --break-system-packages pennylane pennylane-lightning xgboost \
    scikit-learn scipy pandas numpy matplotlib
```
Data bundled in `data/`.

## Config
q=8, seed=7 (+1, 42 for the CTGAN robustness check), `n_total=500`,
`max_train=300`, `max_test=40`, 40 Nyström landmarks, alignment subset=10,
COBYLA maxiter=max(5, n_qubits+2).

## What's different from the original Day 4b zip
`code/pipeline_v3_full.py` has the `RY(w)`/`RY(-w)` no-op bug fixed (see the
inline comment at `make_local_trainable_circuit`). `code/hybrid_readout.py`
itself is unchanged — it was already correctly built on top of the pipeline's
circuit, so fixing the circuit was sufficient.

## Run
```bash
cd code
python3 hybrid_readout.py
```

## Files
| File | Purpose |
|---|---|
| `code/pipeline_v3_full.py` | V3 pipeline with the circuit bug fixed. |
| `code/hybrid_readout.py` | Interception point + 4 readout heads (unchanged). |
| `results/day4b_hybrid_readout_seed7_FIXED.json` | Full sweep, fixed circuit. |
| `results/day4b_ctgan_3seed_check_FIXED.json` | CTGAN 3-seed check, fixed circuit. |
| `figures/day4b_hybrid_readout_fixed.png` | QSVM-highlighted comparison chart. |
