"""Discrete Ricci curvature precompute (Forman / Ollivier).

Used by the H1 / Q2 ablation. For each PyG graph the transform writes
``data.edge_curvature`` of shape ``[num_edges, 1]`` aligned with
``data.edge_index``. Edges where curvature is mathematically undefined (e.g.
isolated edges with no triangles for some Forman variants, or numerical
failures) are written as ``NaN`` so that ``RicciEdgeEncoder`` can substitute a
learnable zero-embedding (per the project proposal, section 2.1).
"""

import logging

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx


def _normalize(values, mode):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if mode == 'zscore' and finite.sum() > 1:
        mu = arr[finite].mean()
        sigma = arr[finite].std()
        if sigma > 1e-8:
            arr[finite] = (arr[finite] - mu) / sigma
    return arr


def _compute_forman(G):
    """Forman-Ricci curvature on the undirected simple graph ``G``.

    Returns a dict ``{(u, v): kappa}`` keyed by sorted node pair.
    """
    from GraphRicciCurvature.FormanRicci import FormanRicci

    frc = FormanRicci(G)
    frc.compute_ricci_curvature()
    out = {}
    for u, v, attr in frc.G.edges(data=True):
        key = (u, v) if u <= v else (v, u)
        out[key] = float(attr.get('formanCurvature', float('nan')))
    return out


def _compute_ollivier(G, alpha):
    """Ollivier-Ricci curvature using POT — no multiprocessing, fork-safe.

    κ(u,v) = 1 − W₁(μᵤ, μᵥ) / d(u,v)
    Measure: each node places mass (1-alpha) on itself and alpha/deg on each
    neighbour (Lin-Lu-Yau). W₁ solved with POT exact EMD.
    """
    import ot

    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    spl = dict(nx.all_pairs_shortest_path_length(G))

    # Cost matrix: shortest-path distance between every pair of nodes.
    C = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            C[i, j] = float(spl[u].get(v, 0))

    def _measure(u):
        m = np.zeros(n)
        nbrs = list(G.neighbors(u))
        deg = len(nbrs)
        m[node_idx[u]] = 1.0 - alpha
        if deg > 0:
            for nb in nbrs:
                m[node_idx[nb]] += alpha / deg
        return m

    out = {}
    for u, v in G.edges():
        mu, mv = _measure(u), _measure(v)
        duv = float(spl[u].get(v, 1))
        w1 = ot.emd2(mu, mv, C) if duv > 0 else 0.0
        kappa = 1.0 - w1 / duv if duv > 0 else 0.0
        key = (u, v) if u <= v else (v, u)
        out[key] = kappa
    return out


def compute_ricci_curvature(data, cfg):
    """Per-graph transform: attach ``data.edge_curvature``."""
    variant = cfg.ricci.variant.lower()
    edge_index = data.edge_index
    num_edges = edge_index.shape[1]

    if num_edges == 0:
        data.edge_curvature = torch.zeros((0, 1), dtype=torch.float)
        return data

    G = to_networkx(data, to_undirected=True)
    if G.number_of_edges() == 0:
        data.edge_curvature = torch.full((num_edges, 1), float('nan'),
                                          dtype=torch.float)
        return data

    try:
        if variant == 'forman':
            kappa_map = _compute_forman(G)
        elif variant == 'ollivier':
            kappa_map = _compute_ollivier(G, alpha=cfg.ricci.alpha)
        else:
            raise ValueError(f"Unknown cfg.ricci.variant: {variant}")
    except Exception as exc:
        logging.warning(
            f"Ricci ({variant}) failed on graph with {G.number_of_nodes()} "
            f"nodes / {G.number_of_edges()} edges: {exc}. Falling back to NaN."
        )
        data.edge_curvature = torch.full((num_edges, 1), float('nan'),
                                          dtype=torch.float)
        return data

    raw = np.empty(num_edges, dtype=np.float64)
    for i in range(num_edges):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        key = (u, v) if u <= v else (v, u)
        raw[i] = kappa_map.get(key, float('nan'))

    raw = _normalize(raw, cfg.ricci.normalize.lower())
    data.edge_curvature = torch.from_numpy(raw).float().unsqueeze(-1)
    return data
