#!/usr/bin/env python3
"""
Collect Q1 ablation results and compute ∆zero per dataset.

Q1 asks: how much does performance degrade when all PE/SE is removed?

    ∆zero(D) = Perf_D(fθ(Pbase)) − Perf_D(fθ(∅))

Sign convention: positive ∆zero always means PE helps.
  • ROC-AUC / Accuracy  → ∆zero = C1 − C2       (higher is better)
  • MAE                 → ∆zero = C2_MAE − C1_MAE  (lower MAE is better,
                                                   so C2 > C1 means PE helps)

Results are read from:
    results/<config-name>/agg/test/best.json   (written by agg_runs after training)
with a fallback to manually scanning per-seed stats.json files.

Usage (from GraphGPS root, after running scripts/run_q1_ablation.sh):
    python scripts/collect_q1_results.py
    python scripts/collect_q1_results.py --results-dir /path/to/results
"""

import argparse
import json
import os

import numpy as np


# ── Experiment registry ───────────────────────────────────────────────────────
# metric_agg: how the best val epoch is selected ('argmin' for MAE, 'argmax' otherwise)
EXPERIMENTS = [
    dict(dataset='ZINC (subset)',
         metric='mae',        higher_is_better=False, metric_agg='argmin',
         c1='zinc-GPS+RWSE',  c2='zinc-GPS-noPE'),
    dict(dataset='ogbg-molhiv',
         metric='auc',        higher_is_better=True,  metric_agg='argmax',
         c1='ogbg-molhiv-GPS', c2='ogbg-molhiv-GPS-noPE'),
    dict(dataset='MUTAG',
         metric='accuracy',   higher_is_better=True,  metric_agg='argmax',
         c1='mutag-GPS',      c2='mutag-GPS-noPE'),
    dict(dataset='ENZYMES',
         metric='accuracy',   higher_is_better=True,  metric_agg='argmax',
         c1='enzymes-GPS',    c2='enzymes-GPS-noPE'),
    dict(dataset='NCI1',
         metric='accuracy',   higher_is_better=True,  metric_agg='argmax',
         c1='nci1-GPS',       c2='nci1-GPS-noPE'),
    dict(dataset='CIFAR10',
         metric='accuracy',   higher_is_better=True,  metric_agg='argmax',
         c1='cifar10-GPS',    c2='cifar10-GPS-noPE'),
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_jsonl(path):
    """Read a JSONL file (one JSON dict per line) into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def best_from_agg(results_dir, config_name, metric):
    """Read mean ± std from the aggregated best.json written by agg_runs."""
    path = os.path.join(results_dir, config_name, 'agg', 'test', 'best.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        stats = json.load(f)
    # agg_runs stores the metric under its plain key and std under '<metric>_std'
    metric_key = _find_metric_key(stats, metric)
    if metric_key is None:
        return None, None
    return stats.get(metric_key), stats.get(f'{metric_key}_std')


def best_from_seeds(results_dir, config_name, metric, metric_agg):
    """Manually compute mean ± std across seed directories."""
    exp_dir = os.path.join(results_dir, config_name)
    if not os.path.exists(exp_dir):
        return None, None

    seed_scores = []
    for entry in sorted(os.listdir(exp_dir)):
        if not entry.isdigit():
            continue
        seed_dir = os.path.join(exp_dir, entry)
        val_path  = os.path.join(seed_dir, 'val',  'stats.json')
        test_path = os.path.join(seed_dir, 'test', 'stats.json')
        if not (os.path.exists(val_path) and os.path.exists(test_path)):
            continue

        val_records  = read_jsonl(val_path)
        test_records = read_jsonl(test_path)
        if not val_records or not test_records:
            continue

        metric_key = _find_metric_key(val_records[0], metric)
        if metric_key is None:
            continue

        val_scores = [r.get(metric_key, float('nan')) for r in val_records]
        best_idx   = int(np.argmin(val_scores) if metric_agg == 'argmin'
                         else np.argmax(val_scores))
        best_epoch = val_records[best_idx]['epoch']

        # Find the test record at the same epoch
        test_at_best = next(
            (r for r in test_records if r.get('epoch') == best_epoch),
            test_records[min(best_idx, len(test_records) - 1)]
        )
        score = test_at_best.get(metric_key)
        if score is not None:
            seed_scores.append(float(score))

    if not seed_scores:
        return None, None
    return float(np.mean(seed_scores)), float(np.std(seed_scores))


def _find_metric_key(record, preferred):
    """Return the metric key from a stats dict, falling back to common aliases."""
    if preferred in record:
        return preferred
    for alt in ['mae', 'auc', 'accuracy', 'ap', 'f1']:
        if alt in record:
            return alt
    return None


def load_result(results_dir, config_name, metric, metric_agg):
    """Try agg best.json first; fall back to per-seed scan."""
    val, std = best_from_agg(results_dir, config_name, metric)
    if val is None:
        val, std = best_from_seeds(results_dir, config_name, metric, metric_agg)
    return val, std


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_result(val, std):
    if val is None:
        return '(missing)'
    s = f'{val:.4f}'
    if std is not None and not np.isnan(std):
        s += f' ±{std:.4f}'
    return s


def fmt_delta(delta):
    if delta is None:
        return 'n/a'
    sign = '+' if delta >= 0 else ''
    return f'{sign}{delta:.4f}'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Collect Q1 ablation results (∆zero per dataset)')
    parser.add_argument('--results-dir', default='results',
                        help='Root results directory (default: results)')
    args = parser.parse_args()
    results_dir = args.results_dir

    print()
    print('Q1 — Systematic Zero-Embedding Ablation  (§2.2)')
    print('∆zero(D) = Perf(fθ(Pbase)) − Perf(fθ(∅))  '
          '[positive = PE/SE improves performance]')
    print(f'Results read from: {os.path.abspath(results_dir)}')
    print()

    col_w = [22, 10, 26, 22, 10]
    header = (f"{'Dataset':<{col_w[0]}} {'Metric':<{col_w[1]}} "
              f"{'C1  Pbase (with PE)':<{col_w[2]}} "
              f"{'C2  P=∅ (no PE)':<{col_w[3]}} "
              f"{'∆zero':>{col_w[4]}}")
    sep = '─' * len(header)
    print(header)
    print(sep)

    missing_runs = []
    rows = []

    for exp in EXPERIMENTS:
        ds       = exp['dataset']
        metric   = exp['metric']
        agg      = exp['metric_agg']
        hib      = exp['higher_is_better']

        c1_val, c1_std = load_result(results_dir, exp['c1'], metric, agg)
        c2_val, c2_std = load_result(results_dir, exp['c2'], metric, agg)

        if c1_val is None:
            missing_runs.append(exp['c1'])
        if c2_val is None:
            missing_runs.append(exp['c2'])

        if c1_val is not None and c2_val is not None:
            # Positive ∆zero → PE helps, regardless of metric direction
            delta = (c1_val - c2_val) if hib else (c2_val - c1_val)
        else:
            delta = None

        row = (f"{ds:<{col_w[0]}} {metric:<{col_w[1]}} "
               f"{fmt_result(c1_val, c1_std):<{col_w[2]}} "
               f"{fmt_result(c2_val, c2_std):<{col_w[3]}} "
               f"{fmt_delta(delta):>{col_w[4]}}")
        print(row)
        rows.append((ds, metric, c1_val, c1_std, c2_val, c2_std, delta))

    print(sep)
    print()

    # Summary
    completed = [(r[0], r[6]) for r in rows if r[6] is not None]
    if completed:
        print('Summary:')
        helped   = [(ds, d) for ds, d in completed if d > 0]
        hurt     = [(ds, d) for ds, d in completed if d <= 0]
        print(f'  PE improves performance on {len(helped)}/{len(completed)} datasets: '
              f"{', '.join(ds for ds, _ in helped) or 'none'}")
        if hurt:
            print(f'  PE did not help on: '
                  f"{', '.join(ds for ds, _ in hurt)}")
        deltas = [d for _, d in completed]
        print(f'  Mean |∆zero|: {np.mean(np.abs(deltas)):.4f}')
        print()

    if missing_runs:
        print(f'Missing results for: {missing_runs}')
        print('Run experiments first:  bash scripts/run_q1_ablation.sh')
        print('(set SKIP_SLOW=1 to skip ZINC and molhiv for quick testing)')


if __name__ == '__main__':
    main()
