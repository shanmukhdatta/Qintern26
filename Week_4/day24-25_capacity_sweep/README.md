# Day 24-25 — Qubit-capacity analysis beyond 12 qubits

Fixed-budget sweep (n=200, 100 train, 10 epochs) isolating qubit-count effect from compute-budget confounds. q=4 -> q=12: output-score variance dropped 3.4x (0.051 -> 0.015) while accuracy barely moved and F1 fell -- capacity-limited, not fidelity-limited. q=16 attempted, did not finish (exponential simulator cost wall, ~12-14 qubit ceiling on this hardware).

Files: run_capacity_config.py, results_capacity.json.
