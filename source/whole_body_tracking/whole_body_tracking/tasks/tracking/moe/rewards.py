from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .commands import AttackTargetCommand
from .utils import DEFAULT_EEF_NAMES, compute_min_eef_distance


def target_hit(
    env: ManagerBasedRLEnv,
    command_name: str = "attack_target",
    hit_radius: float = 0.12,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: AttackTargetCommand = env.command_manager.get_term(command_name)
    eef_names = body_names or DEFAULT_EEF_NAMES
    min_dist = compute_min_eef_distance(env, command.target_pos_b, eef_names)
    hit = (min_dist <= hit_radius).float()
    return hit * command.target_present.squeeze(-1)


def target_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "attack_target",
    body_names: list[str] | None = None,
    clamp: float | None = None,
) -> torch.Tensor:
    command: AttackTargetCommand = env.command_manager.get_term(command_name)
    eef_names = body_names or DEFAULT_EEF_NAMES
    min_dist = compute_min_eef_distance(env, command.target_pos_b, eef_names)

    if not hasattr(env, "_moe_prev_dist"):
        env._moe_prev_dist = min_dist.clone()
    prev = env._moe_prev_dist
    progress = prev - min_dist
    if clamp is not None:
        progress = torch.clamp(progress, -clamp, clamp)
    env._moe_prev_dist = min_dist.detach()
    return progress * command.target_present.squeeze(-1)


def moe_weight_entropy(env: ManagerBasedRLEnv, eps: float = 1.0e-6) -> torch.Tensor:
    if not hasattr(env, "_moe_last_weights"):
        return torch.zeros(env.num_envs, device=env.device)
    w = env._moe_last_weights
    entropy = -(w * (w + eps).log()).sum(dim=-1)
    return entropy


def skill_match(env: ManagerBasedRLEnv, command_name: str = "attack_target") -> torch.Tensor:
    """Reward if selected skill matches the target skill (Stage A supervision)."""
    if not getattr(env, "_moe_lock_skill", False):
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_moe_current_skill"):
        return torch.zeros(env.num_envs, device=env.device)
    command: AttackTargetCommand = env.command_manager.get_term(command_name)
    match = (env._moe_current_skill == command.skill_ids).float()
    return match


def switch_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty when switching skills (Stage B)."""
    if getattr(env, "_moe_lock_skill", False):
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_moe_skill_changed"):
        return torch.zeros(env.num_envs, device=env.device)
    return env._moe_skill_changed
