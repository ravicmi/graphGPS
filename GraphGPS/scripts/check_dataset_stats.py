#!/usr/bin/env python3
"""
Verifies that candidate datasets meet the computational feasibility requirements
for the section 2.3 ablation study:
  - Graph-level supervised task
  - Small graphs (fast girth/diameter computation per H2)
  - Structural diversity (bridges present for H1 curvature hypothesis)

Usage (from GraphGPS root):
    python scripts/check_dataset_stats.py [--data-dir DATA_DIR] [--sample N]

Outputs a summary table and per-dataset breakdown with timing estimates for
NetworkX-based diameter, girth, and bridge detection — the same operations
that the FormanSE and GraphStatsSE preprocessing will run during training.
"""

import argparse
import os
import time

import numpy as np
import networkx as nx
import torch
from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset, ZINC
from torch_geometric.utils import to_networkx


# ── helpers ──────────────────────────────────────────────────────────────────

def graph_stats(pyg_data):
    """Return (n_nodes, n_edges, diameter, girth, has_bridge) for one graph."""
    G = to_networkx(pyg_data, to_undirected=True)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    comps = list(nx.connected_components(G))
    largest = G.subgraph(max(comps, key=len)).copy()

    try:
        diam = nx.diameter(largest)
    except Exception:
        diam = 0

    try:
        girth = nx.girth(G)
    except nx.exception.NetworkXError:
        girth = 0  # acyclic (tree/forest)

    has_bridge = any(True for _ in nx.bridges(G))

    return n, m, diam, girth, has_bridge


def analyze(name, dataset, sample_size):
    """Print stats for one dataset."""
    n_total = len(dataset)
    n_sample = min(sample_size, n_total)
    indices = np.random.default_rng(42).choice(n_total, n_sample, replace=False)

    nodes_all = [dataset[int(i)].num_nodes for i in range(n_total)]
    edges_all = [dataset[int(i)].edge_index.shape[1] // 2 for i in range(n_total)]

    node_feat_dim = dataset.num_node_features
    edge_feat_dim = dataset.num_edge_features

    t0 = time.perf_counter()
    ns, ms, diams, girths, bridges = [], [], [], [], []
    for idx in indices:
        n, m, d, g, b = graph_stats(dataset[int(idx)])
        ns.append(n); ms.append(m); diams.append(d)
        girths.append(g); bridges.append(b)
    elapsed = time.perf_counter() - t0

    ms_per_graph = elapsed / n_sample * 1000
    full_est_s = ms_per_graph * n_total / 1000

    # Determine task type from label shape
    y0 = dataset[0].y
    if y0 is not None and y0.numel() == 1:
        if torch.is_floating_point(y0):
            task_str = "regression"
        else:
            n_cls = len(torch.unique(torch.cat([dataset[i].y for i in range(n_total)])))
            task_str = f"{n_cls}-class classification"
    else:
        task_str = "unknown"

    fits = ms_per_graph < 500  # feasible if <500ms per graph

    print(f"\n{'='*60}")
    print(f"  {name}  ({'FITS ✓' if fits else 'SLOW ✗'})")
    print(f"{'='*60}")
    print(f"  Graphs total : {n_total}")
    print(f"  Task         : {task_str}")
    print(f"  Node feats   : {node_feat_dim}  |  Edge feats: {edge_feat_dim}")
    print(f"  Nodes  — min:{min(nodes_all):4d}  avg:{np.mean(nodes_all):6.1f}  "
          f"max:{max(nodes_all):4d}")
    print(f"  Edges  — min:{min(edges_all):4d}  avg:{np.mean(edges_all):6.1f}  "
          f"max:{max(edges_all):4d}")
    print(f"  --- Sampled {n_sample} graphs ---")
    print(f"  Diameter     : avg={np.mean(diams):.1f}  max={max(diams)}")
    acyclic_pct = sum(g == 0 for g in girths) / n_sample * 100
    print(f"  Girth        : avg={np.mean([g for g in girths if g > 0] or [0]):.1f}  "
          f"acyclic={acyclic_pct:.0f}%")
    bridge_pct = np.mean(bridges) * 100
    print(f"  Bridges (H1) : {bridge_pct:.0f}% of sampled graphs have bridge edges")
    print(f"  Timing       : {ms_per_graph:.1f} ms/graph  →  "
          f"~{full_est_s:.0f}s for full dataset")
    print(f"  Feasible     : {'YES (< 500 ms/graph)' if fits else 'NO  (>= 500 ms/graph)'}")

    return fits


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets',
                        help='Root directory for dataset caching')
    parser.add_argument('--sample', type=int, default=200,
                        help='Number of graphs to sample for timing stats')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    results = {}

    # TU datasets
    for ds_name in ['MUTAG', 'ENZYMES', 'NCI1']:
        print(f"\nLoading {ds_name}...")
        ds = TUDataset(root=os.path.join(args.data_dir, ds_name), name=ds_name)
        results[ds_name] = analyze(ds_name, ds, args.sample)

    # GNNBenchmark: CIFAR10 superpixel graphs (vision, non-molecular)
    print("\nLoading CIFAR10 (superpixels)...")
    cifar_splits = [
        GNNBenchmarkDataset(
            root=os.path.join(args.data_dir, 'CIFAR10'), name='CIFAR10', split=s)
        for s in ['train', 'val', 'test']
    ]
    cifar_graphs = [cifar_splits[0][i] for i in range(len(cifar_splits[0]))] + \
                   [cifar_splits[1][i] for i in range(len(cifar_splits[1]))] + \
                   [cifar_splits[2][i] for i in range(len(cifar_splits[2]))]

    class _CIFARDataset:
        def __init__(self, graphs, src):
            self._graphs = graphs
            self._src = src
        def __len__(self): return len(self._graphs)
        def __getitem__(self, i): return self._graphs[i]
        @property
        def num_node_features(self): return self._src[0].num_node_features
        @property
        def num_edge_features(self): return self._src[0].num_edge_features

    results['CIFAR10'] = analyze('CIFAR10', _CIFARDataset(cifar_graphs, cifar_splits), args.sample)

    # ZINC subset
    print("\nLoading ZINC (subset)...")
    zinc_splits = [
        ZINC(root=os.path.join(args.data_dir, 'ZINC'), subset=True, split=s)
        for s in ['train', 'val', 'test']
    ]
    # Concatenate into a single list for stats
    zinc_graphs = [zinc_splits[0][i] for i in range(len(zinc_splits[0]))] + \
                  [zinc_splits[1][i] for i in range(len(zinc_splits[1]))] + \
                  [zinc_splits[2][i] for i in range(len(zinc_splits[2]))]

    class _ListDataset:
        def __init__(self, graphs):
            self._graphs = graphs
        def __len__(self): return len(self._graphs)
        def __getitem__(self, i): return self._graphs[i]
        @property
        def num_node_features(self): return 1
        @property
        def num_edge_features(self): return 1

    results['ZINC (subset)'] = analyze('ZINC (subset)', _ListDataset(zinc_graphs), args.sample)

    # ogbg-molhiv (optional — requires OGB)
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
        print("\nLoading ogbg-molhiv...")
        molhiv = PygGraphPropPredDataset(
            name='ogbg-molhiv',
            root=os.path.join(args.data_dir, 'OGB')
        )
        results['ogbg-molhiv'] = analyze('ogbg-molhiv', molhiv, args.sample)
    except ImportError:
        print("\nogbg-molhiv skipped (OGB not installed)")

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY — Section 2.3 feasibility")
    print(f"{'='*60}")
    print(f"  {'Dataset':<20}  {'Feasible':>10}")
    print(f"  {'-'*20}  {'-'*10}")
    for name, fits in results.items():
        mark = 'FITS ✓' if fits else 'SLOW ✗'
        print(f"  {name:<20}  {mark:>10}")
    print()


if __name__ == '__main__':
    main()
