from __future__ import annotations

import torch

_LAST_ROUTER_WEIGHTS: torch.Tensor | None = None


def set_router_weights(weights: torch.Tensor):
    """Store the latest router weights for reward-side telemetry."""
    global _LAST_ROUTER_WEIGHTS
    _LAST_ROUTER_WEIGHTS = weights.detach()


def get_router_weights() -> torch.Tensor | None:
    """Get latest router weights (shape: [num_envs, num_skills])."""
    return _LAST_ROUTER_WEIGHTS


def clear_router_weights():
    global _LAST_ROUTER_WEIGHTS
    _LAST_ROUTER_WEIGHTS = None
