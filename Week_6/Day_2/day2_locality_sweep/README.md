# Day 2 — Locality Dose-Response Sweep

QTagger+ Consolidation Week, Team B (Shanmukh). Generalizes V3's local-kernel
measurement (previously hardcoded to pairs, g=2) to arbitrary group size g,
and sweeps g ∈ {1, 2, 4, 8} on CTGAN and Original datasets with no KTA
alignment, per `Day2_Locality_Dose_Response_Brief.md`.

## Reproduce

```bash
pip install pennylane pennylane-lightning xgboost scikit-learn pandas numpy scipy matplotlib
cd code
python3 locality_sweep.py   # runs all 8 (g, dataset) configs, ~15-20 min total
python3 make_figure.py      # regenerates figures/locality_dose_response.png
```

No paths are hardcoded to a personal machine: `code/locality_sweep.py` resolves
`data/ctgan.csv` and `data/original.csv` relative to its own location.

## Folder structure

```
day2_locality_sweep/
├── README.md
├── code/
│   ├── quantum_all_versions.py     # original V1/V2/V3 pipeline (unmodified, for reference)
│   ├── locality_sweep.py           # Day 2 generalized-g sweep (main script)
│   └── make_figure.py              # dose-response figure generator
├── data/
│   ├── ctgan.csv                   # malmem_ctgan.csv, renamed to match DATA_PATHS
│   └── original.csv                # MalMem2022.csv, renamed to match DATA_PATHS
├── results/
│   ├── locality_sweep_results.json # raw results, all 8 runs
│   └── Day2_Locality_Dose_Response_Results.md   # paper-ready table + interpretation
└── figures/
    └── locality_dose_response.png  # accuracy & Gram-std vs g, both datasets
```

## Config (fixed, per brief)

- 8 qubits, seed 7, n_total=500 (balanced 3-class: Ransomware/Spyware/Trojan)
- XGBoost top-8 feature selection (V3's projection), re-upload embedding L=2
- Nystrom landmark QSVM (40 landmarks, 300 train / 40 test), **no KTA alignment**
- g ∈ {1, 2, 4, 8}: qubits partitioned into n_qubits/g non-overlapping groups;
  joint |0...0⟩ probability computed per group and averaged (generalization of
  V3's pairwise g=2 formula — see docstring in `locality_sweep.py`)

## Headline result

g=2 (82.5%) and g=8 (70.0%) on CTGAN both reproduce the brief's reference
values exactly, confirming the generalized function correctly reduces to the
existing cases. The full sweep, however, shows a **non-monotonic** ("optimal
group size") pattern on CTGAN and a **plateau-then-drop** pattern on Original
— not the clean monotonic dose-response the paper's mechanism claim would
ideally show. Full interpretation in
`results/Day2_Locality_Dose_Response_Results.md`.
