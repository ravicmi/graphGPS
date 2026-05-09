from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('ricci_cfg')
def ricci_cfg(cfg):
    """Discrete Ricci curvature edge feature (Q2 / H1)."""
    cfg.ricci = CN()

    cfg.ricci.enable = False

    cfg.ricci.variant = 'forman'

    cfg.ricci.alpha = 0.5

    cfg.ricci.normalize = 'zscore'
