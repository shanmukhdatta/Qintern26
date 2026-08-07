"""Generate the dose-response figure (accuracy & Gram off-diag std vs g)
for both datasets, per Section 4 of the Day 2 brief."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "..", "results", "locality_sweep_results.json")
FIG_PATH = os.path.join(HERE, "..", "figures", "locality_dose_response.png")

with open(RESULTS_PATH) as f:
    results = json.load(f)

by_ds = {"ctgan": {}, "original": {}}
for r in results:
    by_ds[r["dataset"]][r["g"]] = r

g_values = [1, 2, 4, 8]
colors = {"ctgan": "#1f77b4", "original": "#d62728"}
labels = {"ctgan": "CTGAN", "original": "Original (unaugmented)"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
for ds in ["ctgan", "original"]:
    acc = [by_ds[ds][g]["accuracy"] * 100 for g in g_values]
    ax.plot(g_values, acc, marker="o", linewidth=2, color=colors[ds], label=labels[ds])
ax.set_xscale("log", base=2)
ax.set_xticks(g_values)
ax.set_xticklabels([str(g) for g in g_values])
ax.set_xlabel("Group size g (measurement locality)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Accuracy vs. group size g\n(no KTA alignment, 8 qubits, seed 7)")
ax.axhline(33.3, color="gray", linestyle=":", linewidth=1, label="chance (3-class)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
for ds in ["ctgan", "original"]:
    gstd = [by_ds[ds][g]["gram_offdiag_std"] for g in g_values]
    ax.plot(g_values, gstd, marker="s", linewidth=2, color=colors[ds], label=labels[ds])
ax.set_xscale("log", base=2)
ax.set_xticks(g_values)
ax.set_xticklabels([str(g) for g in g_values])
ax.set_xlabel("Group size g (measurement locality)")
ax.set_ylabel("Gram off-diagonal std (landmark submatrix)")
ax.set_title("Kernel concentration vs. group size g")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle("Day 2: Locality Dose-Response Sweep — g \u2208 {1,2,4,8}", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
print(f"Saved figure to {FIG_PATH}")
