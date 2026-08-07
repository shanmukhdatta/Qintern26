"""
Paired QSVM-v3 vs. classical-SVM (RBF) comparison across seeds, for formal
significance testing (paired t-test + Wilcoxon signed-rank) in place of raw
win-rate counts.

Both run_qsvm_v3 and run_classical subsample the test set via
np.random.default_rng(seed).choice(len(Xte), max_test, ...) -- called with the
same seed and the same Xte length, so both draw the *same* 40-point test batch.
That's what makes the comparison a genuine paired design (same held-out points
scored by both models at each seed), not just two independent seed sweeps.
"""
import sys
import os
import json

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_v3_full import load_ctgan, base_filter, project_xgboost, angle_scale, run_qsvm_v3, run_classical
from data_original import load_original


def prep(dataset, n_qubits, seed, n_total=500):
    if dataset == 'CTGAN':
        X, y = load_ctgan(n_total, seed=seed)
    elif dataset == 'Original':
        X, y = load_original(n_total, seed=seed)
    else:
        raise ValueError(dataset)
    Xtr_s, Xte_s, ytr_i, yte_i, classes = base_filter(X, y, seed=seed)
    Xtr_p, Xte_p = project_xgboost(Xtr_s, Xte_s, ytr_i, n_qubits, seed=seed)
    Xtr_r, Xte_r = angle_scale(Xtr_p, Xte_p)
    return Xtr_r, Xte_r, ytr_i, yte_i


def get_classical_only(dataset, n_qubits, seed):
    Xtr_r, Xte_r, ytr_i, yte_i = prep(dataset, n_qubits, seed)
    svm_r, rf_r = run_classical(Xtr_r, Xte_r, ytr_i, yte_i, seed=seed)
    return svm_r, rf_r


def get_paired(dataset, n_qubits, seed):
    Xtr_r, Xte_r, ytr_i, yte_i = prep(dataset, n_qubits, seed)
    svm_r, rf_r = run_classical(Xtr_r, Xte_r, ytr_i, yte_i, seed=seed)
    qsvm_r = run_qsvm_v3(Xtr_r, Xte_r, ytr_i, yte_i, n_qubits=n_qubits, seed=seed)
    return {'seed': seed, 'dataset': dataset, 'classical_svm_acc': svm_r['acc'],
            'classical_rf_acc': rf_r['acc'], 'qsvm_v3_acc': qsvm_r['acc']}


def paired_tests(qsvm_accs, classical_accs):
    qsvm_accs = np.array(qsvm_accs)
    classical_accs = np.array(classical_accs)
    diffs = qsvm_accs - classical_accs
    t_stat, t_p = stats.ttest_rel(qsvm_accs, classical_accs)
    try:
        w_stat, w_p = stats.wilcoxon(qsvm_accs, classical_accs)
    except ValueError as e:
        w_stat, w_p = float('nan'), float('nan')  # e.g. all-zero differences
    n = len(diffs)
    mean_diff = diffs.mean()
    sd_diff = diffs.std(ddof=1) if n > 1 else float('nan')
    se_diff = sd_diff / np.sqrt(n) if n > 1 else float('nan')
    t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else float('nan')
    ci_low, ci_high = mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff
    cohens_d = mean_diff / sd_diff if sd_diff and sd_diff > 0 else float('nan')
    win_rate = float(np.mean(qsvm_accs > classical_accs))
    tie_rate = float(np.mean(qsvm_accs == classical_accs))
    return {
        'n_seeds': n, 'mean_diff': mean_diff, 'sd_diff': sd_diff,
        '95pct_CI': [ci_low, ci_high], 'cohens_d_paired': cohens_d,
        't_stat': t_stat, 't_pvalue': t_p,
        'wilcoxon_stat': w_stat, 'wilcoxon_pvalue': w_p,
        'win_rate': win_rate, 'tie_rate': tie_rate,
        'qsvm_mean': qsvm_accs.mean(), 'qsvm_std': qsvm_accs.std(ddof=1) if n > 1 else float('nan'),
        'classical_mean': classical_accs.mean(), 'classical_std': classical_accs.std(ddof=1) if n > 1 else float('nan'),
    }


if __name__ == '__main__':
    dataset = sys.argv[1]
    n_qubits = int(sys.argv[2])
    seeds = [int(s) for s in sys.argv[3:]]

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', f'paired_{dataset.lower()}.json')
    existing = json.load(open(out_path)) if os.path.exists(out_path) else []
    done_seeds = {r['seed'] for r in existing}

    rows = existing
    for seed in seeds:
        if seed in done_seeds:
            print(f"[{dataset}] seed={seed} already done, skipping", flush=True)
            continue
        r = get_paired(dataset, n_qubits, seed)
        rows.append(r)
        json.dump(rows, open(out_path, 'w'), indent=2)
        print(f"[{dataset}] seed={seed} qsvm={r['qsvm_v3_acc']:.3f} "
              f"classical_svm={r['classical_svm_acc']:.3f} "
              f"diff={r['qsvm_v3_acc']-r['classical_svm_acc']:+.3f}", flush=True)

    rows = sorted(rows, key=lambda r: r['seed'])
    qsvm_accs = [r['qsvm_v3_acc'] for r in rows]
    classical_accs = [r['classical_svm_acc'] for r in rows]
    if len(rows) > 1:
        res = paired_tests(qsvm_accs, classical_accs)
        print(f"\n=== {dataset}, n={res['n_seeds']} seeds ===")
        print(f"QSVM:      {res['qsvm_mean']:.4f} +/- {res['qsvm_std']:.4f}")
        print(f"Classical: {res['classical_mean']:.4f} +/- {res['classical_std']:.4f}")
        print(f"Mean diff: {res['mean_diff']:+.4f}  95% CI [{res['95pct_CI'][0]:+.4f}, {res['95pct_CI'][1]:+.4f}]")
        print(f"Cohen's d (paired): {res['cohens_d_paired']:.3f}")
        print(f"Paired t-test:  t={res['t_stat']:.3f}  p={res['t_pvalue']:.4f}")
        print(f"Wilcoxon:       W={res['wilcoxon_stat']}  p={res['wilcoxon_pvalue']}")
        print(f"Win rate: {res['win_rate']:.1%}  Tie rate: {res['tie_rate']:.1%}")
