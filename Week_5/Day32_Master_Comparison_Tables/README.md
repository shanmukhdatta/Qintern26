# Day 32 — Master configuration comparison (Shanmukh)

**Task:** Consolidate the full configuration comparison (augmentation methods
x classifier families x sample scales) into final master tables, checked for
consistency.

## Status: COMPLETE, with an important correction applied. See `results/MASTER_COMPARISON_TABLE.md` top banner.

`results/MASTER_COMPARISON_TABLE.md` is the actual deliverable — five
sections: (1) the existing V2 27-config quantum-vs-classical grid across
CTGAN/SMOTE/Original, extended with this session's QGAN V3 numbers and the
full V1/V2/V3 comprehensive study; (2) V3 pipeline specifically — **corrected
with 3-5 seed statistics after the original 1-2 seed headline numbers proved
non-representative on expanded testing**; (3) fidelity by method; (4)
downstream classifiers by method; (5) consolidated interpretation tying all
of it together.

**The single most important finding in this table**: expanded seed testing
(Day 31) found that "QSVM v3 beats classical on all 3 datasets" — the
original headline — does not hold. Only Original data shows a robust,
low-variance win (3/3 seeds). CTGAN, SMOTE, and QGAN all show a net loss on
average once 3+ seeds are tested, despite real, verified favorable
single-seed numbers (0.875, 0.600, 0.625). This is a genuine correction, not
a caveat added on top of an unchanged conclusion.

**Consistency check performed**: this correction, found via expanded seeds on
the V3 pipeline, independently confirms the same pattern the pre-existing V2
grid found on a completely different pipeline version (Section 1: CTGAN 11%
win rate, Original 56%). Two different pipeline versions, two different seed
sets, same conclusion — a genuine convergent finding.

**No dedicated code for most of this day** — Day 32 is primarily consolidation.
The exception is the seed-expansion correction, whose code
(`../Day31_QGAN_QSVM_Pipeline/seed_expansion_code/quantum_all_versions.py`)
lives in Day 31 alongside the runs it produced. Every other number here
traces back to Day 29 (`main.py`, `evaluate.py`), Day 30 (`evaluate.py`),
Day 31 (`pipeline_v3_qgan.py`), or the pre-existing V2 grid / comprehensive
study (kept as reference material below).

`reference/` contains the source materials this table draws from — the
original V2 grid report, its detailed writeup, and raw JSON results — kept
alongside for anyone who wants to re-verify the numbers.
