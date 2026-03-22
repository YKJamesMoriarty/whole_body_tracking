from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .commands import AttackTargetCommand
from isaaclab.envs.mdp.rewards import is_alive
from .utils import DEFAULT_EEF_NAMES, compute_min_eef_distance


def target_hit(
    env: ManagerBasedRLEnv,
    command_name: str = "attack_target",
    hit_radius: float = 0.05,
    body_names: list[str] | None = None,
    require_match: bool = False,
) -> torch.Tensor:
    command: AttackTargetCommand = env.command_manager.get_term(command_name)
    eef_names = body_names or DEFAULT_EEF_NAMES
    min_dist = compute_min_eef_distance(env, command.target_pos_b, eef_names)
    hit = (min_dist <= hit_radius).float()
    if require_match:
        if not hasattr(env, "_moe_current_skill"):
            return torch.zeros(env.num_envs, device=env.device)
        match = (env._moe_current_skill == command.skill_ids).float()
        hit = hit * match
    return hit * command.target_present.squeeze(-1)


def target_hit_stage1(
    env: ManagerBasedRLEnv,
    command_name: str = "attack_target",
    hit_radius: float = 0.05,
    body_names: list[str] | None = None,
    require_match: bool = True,
) -> torch.Tensor:
    if not getattr(env, "_moe_lock_skill", False):
        return torch.zeros(env.num_envs, device=env.device)
    return target_hit(env, command_name, hit_radius, body_names, require_match)


def target_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "attack_target",
    body_names: list[str] | None = None,
    clamp: float | None = None,
    require_match: bool = False,
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
    if require_match:
        if not hasattr(env, "_moe_current_skill"):
            return torch.zeros(env.num_envs, device=env.device)
        match = (env._moe_current_skill == command.skill_ids).float()
        progress = progress * match
    return progress * command.target_present.squeeze(-1)


def target_progress_stage1(
    env: ManagerBasedRLEnv,
    command_name: str = "attack_target",
    body_names: list[str] | None = None,
    clamp: float | None = None,
    require_match: bool = True,
) -> torch.Tensor:
    if not getattr(env, "_moe_lock_skill", False):
        return torch.zeros(env.num_envs, device=env.device)
    return target_progress(env, command_name, body_names, clamp, require_match)


def moe_weight_entropy(env: ManagerBasedRLEnv, eps: float = 1.0e-6) -> torch.Tensor:
    if not hasattr(env, "_moe_last_weights"):
        return torch.zeros(env.num_envs, device=env.device)
    w = env._moe_last_weights
    entropy = -(w * (w + eps).log()).sum(dim=-1)
    return entropy


def alive_when_no_target(env: ManagerBasedRLEnv, command_name: str = "attack_target") -> torch.Tensor:
    """Alive reward only when target is not visible (used for later stages)."""
    command: AttackTargetCommand = env.command_manager.get_term(command_name)
    present = command.target_present.squeeze(-1)
    return is_alive(env) * (1.0 - present)


def skill_match_stage1(env: ManagerBasedRLEnv, command_name: str = "attack_target") -> torch.Tensor:
    """Reward if selected skill matches the target skill (Stage A-1, locked)."""
    if not getattr(env, "_moe_lock_skill", False):
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_moe_current_skill"):
        return torch.zeros(env.num_envs, device=env.device)
    command: AttackTargetCommand = env.command_manager.get_term(command_name)
    match = (env._moe_current_skill == command.skill_ids).float()
    return match


def skill_mismatch_stage1(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-shot penalty applied after an episode ends with the wrong locked skill (Stage A-1)."""
    if not hasattr(env, "_moe_episode_mismatch"):
        return torch.zeros(env.num_envs, device=env.device)
    penalty = env._moe_episode_mismatch
    # consume the penalty so it is applied only once
    env._moe_episode_mismatch = torch.zeros_like(penalty)
    return penalty
