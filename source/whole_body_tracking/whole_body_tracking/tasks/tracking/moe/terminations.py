from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .commands import AttackTargetCommand
from .utils import DEFAULT_EEF_NAMES, compute_min_eef_distance


def hit_target(
    env: ManagerBasedRLEnv,
    command_name: str = "attack_target",
    hit_radius: float = 0.05,
    body_names: list[str] | None = None,
    require_match: bool = False,
) -> torch.Tensor:
    command: AttackTargetCommand = env.command_manager.get_term(command_name)
    eef_names = body_names or DEFAULT_EEF_NAMES
    min_dist = compute_min_eef_distance(env, command.target_pos_b, eef_names)
    hit = min_dist <= hit_radius
    if require_match:
        if not hasattr(env, "_moe_current_skill"):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        match = env._moe_current_skill == command.skill_ids
        hit = hit & match
    return hit
