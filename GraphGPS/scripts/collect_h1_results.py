#!/usr/bin/env python3
"""Collect H1 / Q2 ablation results and compute Δ(p; D) per dataset.

Q2 / H1 asks: does adding a single left-behind SE p (Ricci curvature) on top
of the standard Pbase (LapPE) help, and on which datasets?

    Δ(p; D) = Perf_D(fθ(Pbase ∪ {p})) − Perf_D(fθ(Pbase))

Sign convention: positive Δ always means the SE helps.
  * ROC-AUC / Accuracy → Δ = C3 − C1   (higher is better)
  * MAE                → Δ = C1 − C3   (lower is better)

Results are read from:
    results/<config-name>/agg/test/best.json   (written by agg_runs)
with a fallback to per-seed stats.json scanning, mirroring Q1.

Usage (from GraphGPS root, after running scripts/run_h1_ablation.sh):
    python scripts/collect_h1_results.py
    python scripts/collect_h1_results.py --variant ollivier
    python scripts/collect_h1_results.py --results-dir /path/to/results
"""

import argparse
import json
import os

import numpy as np


# ── Experiment registry ───────────────────────────────────────────────────────
# c1 = baseline (Pbase = LapPE only). c3 = Pbase + Ricci(<variant>).
EXPERIMENTS = [
    dict(dataset='MUTAG',
         metric='accuracy', higher_is_better=True, metric_agg='argmax',
         c1='mutag-GPS',
         c3={'forman':   'mutag-GPS-Ricci-forman',
             'ollivier': 'mutag-GPS-Ricci-ollivier'}),
    dict(dataset='ENZYMES',
         metric='accuracy', higher_is_better=True, metric_agg='argmax',
         c1='enzymes-GPS',
         c3={'forman':   'enzymes-GPS-Ricci-forman',
             'ollivier': 'enzymes-GPS-Ricci-ollivier'}),
    dict(dataset='NCI1',
         metric='accuracy', higher_is_better=True, metric_agg='argmax',
         c1='nci1-GPS',
         c3={'forman':   'nci1-GPS-Ricci-forman',
             'ollivier': 'nci1-GPS-Ricci-ollivier'}),
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_jsonl(path):
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


def _find_metric_key(record, preferred):
    if preferred in record:
        return preferred
    for alt in ['mae', 'auc', 'accuracy', 'ap', 'f1']:
        if alt in record:
            return alt
    return None


def best_from_agg(results_dir, config_name, metric):
    path = os.path.join(results_dir, config_name, 'agg', 'test', 'best.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        stats = json.load(f)
    metric_key = _find_metric_key(stats, metric)
    if metric_key is None:
        return None, None
    return stats.get(metric_key), stats.get(f'{metric_key}_std')


def best_from_seeds(results_dir, config_name, metric, metric_agg):
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
        best_idx = int(np.argmin(val_scores) if metric_agg == 'argmin'
                       else np.argmax(val_scores))
        best_epoch = val_records[best_idx]['epoch']

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


def load_result(results_dir, config_name, metric, metric_agg):
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
        description='Collect H1 / Q2 Ricci ablation results (Δ(p; D))')
    parser.add_argument('--results-dir', default='results',
                        help='Root results directory (default: results)')
    parser.add_argument('--variant', default='forman',
                        choices=['forman', 'ollivier'],
                        help='Ricci variant to evaluate against C1')
    args = parser.parse_args()
    results_dir = args.results_dir
    variant = args.variant

    print()
    print(f'H1 — Marginal value of Ricci ({variant}) curvature  (§2.2 / §2.3)')
    print('Δ(p; D) = Perf(fθ(Pbase ∪ {p})) − Perf(fθ(Pbase))   '
          '[positive = Ricci helps]')
    print(f'Results read from: {os.path.abspath(results_dir)}')
    print()

    col_w = [22, 10, 26, 26, 10]
    header = (f"{'Dataset':<{col_w[0]}} {'Metric':<{col_w[1]}} "
              f"{'C1  Pbase (LapPE)':<{col_w[2]}} "
              f"{f'C3  + Ricci/{variant}':<{col_w[3]}} "
              f"{'Δ(p; D)':>{col_w[4]}}")
    sep = '─' * len(header)
    print(header)
    print(sep)

    missing_runs = []
    rows = []

    for exp in EXPERIMENTS:
        ds     = exp['dataset']
        metric = exp['metric']
        agg    = exp['metric_agg']
        hib    = exp['higher_is_better']
        c3_cfg = exp['c3'][variant]

        c1_val, c1_std = load_result(results_dir, exp['c1'], metric, agg)
        c3_val, c3_std = load_result(results_dir, c3_cfg, metric, agg)

        if c1_val is None:
            missing_runs.append(exp['c1'])
        if c3_val is None:
            missing_runs.append(c3_cfg)

        if c1_val is not None and c3_val is not None:
            delta = (c3_val - c1_val) if hib else (c1_val - c3_val)
        else:
            delta = None

        row = (f"{ds:<{col_w[0]}} {metric:<{col_w[1]}} "
               f"{fmt_result(c1_val, c1_std):<{col_w[2]}} "
               f"{fmt_result(c3_val, c3_std):<{col_w[3]}} "
               f"{fmt_delta(delta):>{col_w[4]}}")
        print(row)
        rows.append((ds, metric, c1_val, c1_std, c3_val, c3_std, delta))

    print(sep)
    print()

    completed = [(r[0], r[6]) for r in rows if r[6] is not None]
    if completed:
        print('Summary:')
        helped = [(ds, d) for ds, d in completed if d > 0]
        hurt   = [(ds, d) for ds, d in completed if d <= 0]
        print(f'  Ricci/{variant} helps on {len(helped)}/{len(completed)} '
              f'datasets: {", ".join(ds for ds, _ in helped) or "none"}')
        if hurt:
            print(f'  Ricci/{variant} did not help on: '
                  f'{", ".join(ds for ds, _ in hurt)}')
        deltas = [d for _, d in completed]
        print(f'  Mean Δ(p; D): {np.mean(deltas):+.4f}')
        print(f'  Mean |Δ(p; D)|: {np.mean(np.abs(deltas)):.4f}')
        print()

    if missing_runs:
        unique_missing = sorted(set(missing_runs))
        print(f'Missing results for: {unique_missing}')
        print('Run experiments first:  bash scripts/run_h1_ablation.sh')
        print('(set RUN_OLLIVIER=1 to also run the Ollivier variant)')


if __name__ == '__main__':
    main()
