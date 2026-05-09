#!/usr/bin/env python3
"""
Generate 5 EDA / results plots for the GraphGPS structural encoding ablation.
Run from GraphGPS root with the project venv active:
    python scripts/plot_results.py

Outputs (saved to results/plots/):
    1. dataset_profile.png      - Structural stats per dataset
    2. ricci_distributions.png  - Edge curvature distributions (Forman + Ollivier)
    3. accuracy_conditions.png  - Accuracy per condition with error bars
    4. delta_heatmap.png        - Δ accuracy heatmap across all conditions
    5. learning_curves.png      - Train/val curves C1 vs C3-Forman per dataset
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import networkx as nx
import seaborn as sns
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = 'results'
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASETS  = ['MUTAG', 'ENZYMES', 'NCI1']
N_SAMPLE  = 150   # graphs sampled per dataset for curvature distributions

PALETTE = {
    'C1':          '#2196F3',
    'C2':          '#FF5722',
    'C3-Forman':   '#4CAF50',
    'C3-Ollivier': '#9C27B0',
}

CONFIG_MAP = {
    'MUTAG':   {'C1': 'mutag-GPS',   'C2': 'mutag-GPS-noPE',
                'C3-Forman': 'mutag-GPS-Ricci-forman',
                'C3-Ollivier': 'mutag-GPS-Ricci-ollivier'},
    'ENZYMES': {'C1': 'enzymes-GPS', 'C2': 'enzymes-GPS-noPE',
                'C3-Forman': 'enzymes-GPS-Ricci-forman',
                'C3-Ollivier': 'enzymes-GPS-Ricci-ollivier'},
    'NCI1':    {'C1': 'nci1-GPS',    'C2': 'nci1-GPS-noPE',
                'C3-Forman': 'nci1-GPS-Ricci-forman',
                'C3-Ollivier': 'nci1-GPS-Ricci-ollivier'},
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_jsonl(path):
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return records


def load_agg_best(config_name, metric='accuracy'):
    path = os.path.join(RESULTS_DIR, config_name, 'agg', 'test', 'best.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        stats = json.load(f)
    return stats.get(metric), stats.get(f'{metric}_std')


def load_seed_curves(config_name, split='val', metric='accuracy'):
    exp_dir = os.path.join(RESULTS_DIR, config_name)
    curves = []
    if not os.path.exists(exp_dir):
        return curves
    for entry in sorted(os.listdir(exp_dir)):
        if not entry.isdigit():
            continue
        records = read_jsonl(os.path.join(exp_dir, entry, split, 'stats.json'))
        if records:
            curves.append([r.get(metric, float('nan')) for r in records])
    return curves


def load_pyg_dataset(name):
    return TUDataset(root='/tmp/TUDataset', name=name)

# ── Plot 1 — Dataset Structural Profile ───────────────────────────────────────

def plot_dataset_profile():
    print('Plot 1: dataset structural profile...')

    stats = {}
    for name in DATASETS:
        dataset = load_pyg_dataset(name)
        n_nodes, n_edges, degrees = [], [], []
        for data in dataset:
            G = to_networkx(data, to_undirected=True)
            n  = G.number_of_nodes()
            e  = G.number_of_edges()
            if n == 0:
                continue
            n_nodes.append(n)
            n_edges.append(e)
            degrees.extend([d for _, d in G.degree()])
        stats[name] = {
            'Num graphs':  len(dataset),
            'Avg nodes':   np.mean(n_nodes),
            'Avg edges':   np.mean(n_edges),
            'Avg degree':  np.mean(degrees),
        }
        print(f'  {name}: {len(dataset)} graphs, '
              f'avg nodes={stats[name]["Avg nodes"]:.1f}')

    metrics = list(next(iter(stats.values())).keys())
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle('Dataset Structural Profile', fontweight='bold', y=1.02)

    for ax, metric in zip(axes, metrics):
        vals = [stats[ds][metric] for ds in DATASETS]
        bars = ax.bar(DATASETS, vals, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(vals) * 1.25)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'dataset_profile.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')

# ── Plot 2 — Ricci Curvature Distributions ────────────────────────────────────

def plot_ricci_distributions():
    print('Plot 2: Ricci curvature distributions...')

    import ot
    from GraphRicciCurvature.FormanRicci import FormanRicci

    def forman_curvatures(G):
        frc = FormanRicci(G)
        frc.compute_ricci_curvature()
        return [d['formanCurvature'] for _, _, d in frc.G.edges(data=True)]

    def ollivier_curvatures(G, alpha=0.5):
        nodes = list(G.nodes())
        idx   = {n: i for i, n in enumerate(nodes)}
        n     = len(nodes)
        spl   = dict(nx.all_pairs_shortest_path_length(G))
        C     = np.array([[float(spl[u].get(v, 0)) for v in nodes] for u in nodes])
        def measure(u):
            m = np.zeros(n); nbrs = list(G.neighbors(u)); deg = len(nbrs)
            m[idx[u]] = 1.0 - alpha
            if deg > 0:
                for nb in nbrs: m[idx[nb]] += alpha / deg
            return m
        kappas = []
        for u, v in G.edges():
            duv = float(spl[u].get(v, 1))
            w1  = ot.emd2(measure(u), measure(v), C) if duv > 0 else 0.0
            kappas.append(1.0 - w1 / duv if duv > 0 else 0.0)
        return kappas

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey='row')
    fig.suptitle('Edge Ricci Curvature Distributions per Dataset',
                 fontweight='bold', y=1.01)

    for col, name in enumerate(DATASETS):
        dataset = load_pyg_dataset(name)
        sample  = list(dataset)[:N_SAMPLE]

        all_forman, all_ollivier = [], []
        for data in sample:
            G = to_networkx(data, to_undirected=True)
            if G.number_of_edges() == 0:
                continue
            try:
                all_forman.extend(forman_curvatures(G))
            except Exception:
                pass
            try:
                all_ollivier.extend(ollivier_curvatures(G))
            except Exception:
                pass

        for row, (vals, label, color) in enumerate([
            (all_forman,   'Forman κ',   '#4CAF50'),
            (all_ollivier, 'Ollivier κ', '#9C27B0'),
        ]):
            ax = axes[row][col]
            if vals:
                ax.hist(vals, bins=40, color=color, alpha=0.75, edgecolor='white')
                ax.axvline(np.mean(vals), color='black', linestyle='--',
                           linewidth=1.2, label=f'mean={np.mean(vals):.2f}')
                ax.legend(fontsize=8)
            ax.set_title(f'{name} — {label}' if row == 0 else '')
            ax.set_xlabel('κ value')
            ax.set_ylabel('Edge count' if col == 0 else '')
            ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'ricci_distributions.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')

# ── Plot 3 — Accuracy per Condition ───────────────────────────────────────────

def plot_accuracy_conditions():
    print('Plot 3: accuracy per condition...')

    conditions = ['C1', 'C2', 'C3-Forman', 'C3-Ollivier']
    means = {c: [] for c in conditions}
    stds  = {c: [] for c in conditions}

    for ds in DATASETS:
        for cond in conditions:
            cfg_name = CONFIG_MAP[ds][cond]
            mu, sigma = load_agg_best(cfg_name, 'accuracy')
            means[cond].append(mu)
            stds[cond].append(sigma if sigma is not None else 0.0)

    x     = np.arange(len(DATASETS))
    width = 0.20
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(10, 5))
    for cond, offset in zip(conditions, offsets):
        vals  = [v if v is not None else 0 for v in means[cond]]
        errs  = stds[cond]
        bars  = ax.bar(x + offset * width, vals, width,
                       label=cond, color=PALETTE[cond],
                       edgecolor='white', linewidth=0.8)
        ax.errorbar(x + offset * width, vals, yerr=errs,
                    fmt='none', color='black', capsize=3, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy per Condition and Dataset', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(title='Condition', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'accuracy_conditions.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')

# ── Plot 4 — Δ Heatmap ────────────────────────────────────────────────────────

def plot_delta_heatmap():
    print('Plot 4: delta heatmap...')

    delta_cols = ['Δzero\n(C1−C2)', 'Δ Forman\n(C3−C1)', 'Δ Ollivier\n(C3−C1)']
    matrix = np.zeros((len(DATASETS), 3))

    for i, ds in enumerate(DATASETS):
        c1_mu, _ = load_agg_best(CONFIG_MAP[ds]['C1'], 'accuracy')
        c2_mu, _ = load_agg_best(CONFIG_MAP[ds]['C2'], 'accuracy')
        c3f_mu, _ = load_agg_best(CONFIG_MAP[ds]['C3-Forman'], 'accuracy')
        c3o_mu, _ = load_agg_best(CONFIG_MAP[ds]['C3-Ollivier'], 'accuracy')

        matrix[i, 0] = (c1_mu - c2_mu)  if (c1_mu and c2_mu)  else float('nan')
        matrix[i, 1] = (c3f_mu - c1_mu) if (c3f_mu and c1_mu) else float('nan')
        matrix[i, 2] = (c3o_mu - c1_mu) if (c3o_mu and c1_mu) else float('nan')

    matrix_pct = matrix * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    vmax = np.nanmax(np.abs(matrix_pct))
    im = ax.imshow(matrix_pct, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label='Δ Accuracy (pp)')

    ax.set_xticks(range(3))
    ax.set_xticklabels(delta_cols, fontsize=10)
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS)
    ax.set_title('Δ Accuracy Heatmap (percentage points)', fontweight='bold')

    for i in range(len(DATASETS)):
        for j in range(3):
            val = matrix_pct[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:+.1f}pp', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=text_color)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'delta_heatmap.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')

# ── Plot 5 — Learning Curves ──────────────────────────────────────────────────

def plot_learning_curves():
    print('Plot 5: learning curves...')

    pairs = [
        ('MUTAG',   'mutag-GPS',   'mutag-GPS-Ricci-forman'),
        ('ENZYMES', 'enzymes-GPS', 'enzymes-GPS-Ricci-forman'),
        ('NCI1',    'nci1-GPS',    'nci1-GPS-Ricci-forman'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Training Curves: C1 (LapPE) vs C3-Forman (LapPE + Ricci)',
                 fontweight='bold', y=1.02)

    for ax, (name, c1_cfg, c3_cfg) in zip(axes, pairs):
        for cfg_name, label, color, style in [
            (c1_cfg, 'C1 val',   PALETTE['C1'],        '-'),
            (c3_cfg, 'C3 val',   PALETTE['C3-Forman'], '-'),
            (c1_cfg, 'C1 train', PALETTE['C1'],        '--'),
            (c3_cfg, 'C3 train', PALETTE['C3-Forman'], '--'),
        ]:
            split = 'val' if 'val' in label else 'train'
            curves = load_seed_curves(cfg_name, split=split, metric='accuracy')
            if not curves:
                continue
            max_len = max(len(c) for c in curves)
            padded  = [c + [float('nan')] * (max_len - len(c)) for c in curves]
            arr     = np.array(padded, dtype=float)
            mean    = np.nanmean(arr, axis=0)
            std     = np.nanstd(arr, axis=0)
            epochs  = np.arange(max_len)

            alpha_fill = 0.15
            ax.plot(epochs, mean, linestyle=style, color=color,
                    linewidth=1.8, label=label)
            if 'val' in label:
                ax.fill_between(epochs, mean - std, mean + std,
                                color=color, alpha=alpha_fill)

        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy' if name == 'MUTAG' else '')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'learning_curves.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Saving plots to: {os.path.abspath(PLOTS_DIR)}\n')
    plot_dataset_profile()
    plot_ricci_distributions()
    plot_accuracy_conditions()
    plot_delta_heatmap()
    plot_learning_curves()
    print('\nAll 5 plots saved.')
