from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply

from whole_body_tracking.learning.router_telemetry import get_router_weights
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """Shared utility for termination terms that still index specific bodies."""
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]

def effector_target_hit(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Pulse reward for first valid hit in each episode."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    hit_mask, _ = command.check_hit()
    command.update_curriculum(hit_mask)
    return hit_mask.float()


def amp_style_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Optional AMP style reward (0 when AMP is disabled)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.compute_amp_style_reward()


def router_diversity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_population_entropy: float = 0.60,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cross-env anti-collapse penalty for the MoE router.

    Semantics (use with a POSITIVE reward weight):
    - Returns 0 when skills are well distributed across environments.
    - Returns a NEGATIVE penalty when one skill dominates the population
      (i.e., most environments assign the majority of weight to the same expert).

    Only the population-level (cross-env) distribution is considered.
    Each individual environment is free to specialise: a single env routing
    90% of weight to the cross-punch expert is fine as long as other envs
    are using different skills.  What we want to prevent is ALL envs routing
    to the same expert regardless of context.

    Implementation:
        mean_weights[k] = average router weight for skill k across all envs
        pop_entropy     = H(mean_weights) / log(num_skills)  in [0, 1]
        penalty         = clamp(pop_entropy - min_population_entropy, max=0)

    penalty == 0  when pop_entropy >= min_population_entropy (healthy diversity)
    penalty < 0   when pop_entropy <  min_population_entropy (collapse happening)
    """
    _ = command_name  # keep manager API compatibility
    weights = get_router_weights()
    if weights is None or weights.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device)

    # Population-level mean skill usage.  Shape: [num_skills]
    mean_weights = torch.mean(weights, dim=0)

    # Normalised entropy of the population distribution.  Scalar in [0, 1].
    num_skills = float(weights.shape[-1])
    norm = torch.log(torch.tensor(num_skills, device=weights.device, dtype=weights.dtype) + eps)
    pop_entropy = -torch.sum(mean_weights * torch.log(mean_weights + eps)) / norm

    # Penalty: 0 when diverse enough, negative proportional to entropy deficit.
    penalty = torch.clamp(pop_entropy - min_population_entropy, max=0.0)

    # Broadcast scalar penalty to every env (same pressure everywhere).
    return penalty.expand(env.num_envs)


def post_hit_return_to_start_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_xy: float = 0.15,
) -> torch.Tensor:
    """
    Reward returning to episode start anchor position after hit.

    - Before hit: reward = 0
    - After hit: reward = exp(-||p_xy - p_start_xy||^2 / std_xy^2)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    current_xy = command.robot_anchor_pos_w[:, :2]
    start_xy = command.episode_start_anchor_pos_w[:, :2]
    error = torch.sum(torch.square(current_xy - start_xy), dim=-1)
    reward = torch.exp(-error / std_xy**2)
    return torch.where(command.has_hit, reward, torch.zeros_like(reward))


def alive_bonus(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Small survival reward to encourage policies that avoid early falls."""
    _ = command_name
    return torch.ones(env.num_envs, device=env.device)


def posture_unstable(
    env: ManagerBasedRLEnv,
    command_name: str,
    tilt_threshold: float = 0.78,
    full_penalty_tilt: float = 0.50,
    penalty_exponent: float = 2.0,
) -> torch.Tensor:
    """
    Penalize torso tilt with a dead-zone to preserve kick-style motion freedom.

    - No penalty when cos(up, world_up) >= tilt_threshold.
    - Smoothly increases penalty as tilt approaches `full_penalty_tilt`.
    - Saturates at -1.0 for very large tilt.
    """
    if full_penalty_tilt >= tilt_threshold:
        raise ValueError("full_penalty_tilt must be smaller than tilt_threshold.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    root_quat_w = command.robot_anchor_quat_w
    local_up = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).repeat(env.num_envs, 1)
    body_up_w = quat_apply(root_quat_w, local_up)
    cos_with_world_up = body_up_w[:, 2]
    normalized = torch.clamp(
        (tilt_threshold - cos_with_world_up) / (tilt_threshold - full_penalty_tilt),
        min=0.0,
        max=1.0,
    )
    return -(normalized**penalty_exponent)


def root_roll_pitch_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float = 1.5,
) -> torch.Tensor:
    """Penalize excessive roll/pitch angular velocity to reduce abrupt loss of balance."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    roll_pitch_rate = torch.norm(command.robot_anchor_ang_vel_w[:, :2], dim=-1)
    excess = torch.clamp(roll_pitch_rate - deadband, min=0.0)
    return -(excess**2)


def robot_falling_penalty_event(
    env: ManagerBasedRLEnv,
    termination_term_name: str = "robot_falling",
) -> torch.Tensor:
    """One-shot fall penalty independent of simulation dt.

    Reward manager multiplies by dt, so divide here to keep a constant event magnitude.
    """
    fallen = env.termination_manager.get_term(termination_term_name).float()
    return fallen / max(env.step_dt, 1e-8)
