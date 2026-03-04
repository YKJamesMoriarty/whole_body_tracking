from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def effector_target_hit(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Pulse reward for first valid hit in each episode."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    hit_mask, _ = command.check_hit()
    command.update_curriculum(hit_mask)
    return hit_mask.float()


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


def effector_target_near(
    env: ManagerBasedRLEnv,
    command_name: str,
    guidance_radius: float = 0.25,
    scale: float = 10.0,
) -> torch.Tensor:
    """
    Dense progress reward before hit:
    reward only when the active effector reaches a new minimum distance to target.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    effector_pos_w = command.robot_body_pos_w[:, command.effector_index]
    current_distance = torch.norm(effector_pos_w - command.target_pos_w, dim=-1)
    in_guidance_sphere = current_distance < guidance_radius
    command.has_entered_guidance_sphere = command.has_entered_guidance_sphere | in_guidance_sphere

    has_progress = current_distance < command.min_distance_to_target
    should_reward = in_guidance_sphere & has_progress & command.task_rewards_enabled

    progress = command.min_distance_to_target - current_distance
    reward = torch.where(should_reward, progress * scale, torch.zeros_like(current_distance))

    command.min_distance_to_target = torch.where(should_reward, current_distance, command.min_distance_to_target)
    return reward


def posture_unstable(
    env: ManagerBasedRLEnv,
    command_name: str,
    tilt_threshold: float = 0.8,
) -> torch.Tensor:
    """
    Penalize torso tilt when it exceeds a threshold.

    `tilt_threshold` is the minimum allowed cosine with world-up:
    - 1.0 means perfectly upright
    - smaller value allows more tilt
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    root_quat_w = command.robot_anchor_quat_w
    local_up = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).repeat(env.num_envs, 1)
    body_up_w = quat_apply(root_quat_w, local_up)
    cos_with_world_up = body_up_w[:, 2]
    penalty = torch.clamp(tilt_threshold - cos_with_world_up, min=0.0)
    return -penalty
