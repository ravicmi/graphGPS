"""Edge encoder that consumes precomputed Ricci curvature (Forman or Ollivier).

Reads ``batch.edge_curvature`` of shape ``[num_edges, 1]``. Edges with ``NaN``
curvature (mathematically undefined / numerically failed) are mapped to a
learnable zero-embedding ``0_emb`` of size ``emb_dim`` per the project proposal
(section 2.1), so that no false structural prior is injected.
"""

import torch
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('RicciEdge')
class RicciEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(1, emb_dim)
        self.zero_emb = torch.nn.Parameter(torch.zeros(emb_dim))
        torch.nn.init.normal_(self.zero_emb, std=0.02)

    def forward(self, batch):
        if not hasattr(batch, 'edge_curvature') or batch.edge_curvature is None:
            raise RuntimeError(
                "RicciEdgeEncoder requires 'edge_curvature' on the batch. "
                "Make sure cfg.ricci.enable=True so the loader precomputes it."
            )
        kappa = batch.edge_curvature
        if kappa.dim() == 1:
            kappa = kappa.unsqueeze(-1)
        kappa = kappa.float()

        nan_mask = torch.isnan(kappa).any(dim=-1)
        kappa_safe = torch.where(torch.isnan(kappa),
                                 torch.zeros_like(kappa), kappa)
        encoded = self.linear(kappa_safe)
        encoded = torch.where(nan_mask.unsqueeze(-1),
                              self.zero_emb.expand_as(encoded),
                              encoded)

        batch.edge_attr = encoded
        return batch
