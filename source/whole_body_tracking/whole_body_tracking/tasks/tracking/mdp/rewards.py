from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

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


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


# =============================================================================
# 拳击机器人任务导向奖励 (Stage 1: Mimic 训练辅助)
# =============================================================================


def effector_target_tracking_exp(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    std: float = 0.2,
    effector_body_name: str = "right_wrist_yaw_link"
) -> torch.Tensor:
    """
    攻击肢体靠近目标位置的奖励
    
    含义:
    - 鼓励机器人的攻击肢体（如右手）靠近目标位置
    - Stage 1: 目标位置 = 参考动作中该肢体应该到达的位置
    - Stage 2: 目标位置 = 场景中实际的攻击目标点
    
    奖励公式:
    - reward = exp(-distance² / std²)
    - 距离越小，奖励越接近 1.0
    - 距离越大，奖励越接近 0.0
    
    与 motion_body_pos 奖励的区别:
    - motion_body_pos: 考虑所有 14 个身体部位的平均误差
    - effector_target_tracking: 只关注攻击肢体，更聚焦
    
    数据一致性说明:
    - 观测函数 target_relative_position() 使用 command.body_pos_relative_w
    - 本奖励函数也使用 command.body_pos_relative_w
    - 两者在同一个 step 中计算，使用的是**同一帧**的数据
    
    Stage 2 TODO:
    - 将 target_pos_w 替换为场景中实际的目标点
    
    Args:
        env: 环境实例
        command_name: 动作命令名称
        std: 标准差，控制奖励的敏感度（值越小，对距离越敏感）
        effector_body_name: 攻击肢体名称
    
    Returns:
        Tensor (num_envs,): 每个环境的奖励值 [0, 1]
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 获取攻击肢体的索引
    try:
        effector_index = command.cfg.body_names.index(effector_body_name)
    except ValueError:
        effector_index = -1
    
    # Stage 1: 目标位置 = 参考动作中该肢体的位置（已对齐到机器人当前位置）
    # 这和观测函数 target_relative_position() 使用的是**同一个数据源**
    target_pos_w = command.body_pos_relative_w[:, effector_index]  # (num_envs, 3)
    
    # 机器人实际的攻击肢体位置
    robot_effector_pos_w = command.robot_body_pos_w[:, effector_index]  # (num_envs, 3)
    
    # 计算距离的平方
    error = torch.sum(torch.square(target_pos_w - robot_effector_pos_w), dim=-1)  # (num_envs,)
    
    # 指数奖励：距离越小，奖励越高
    reward = torch.exp(-error / std**2)
    
    return reward
