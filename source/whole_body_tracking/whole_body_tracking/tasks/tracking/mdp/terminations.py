from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


# =============================================================================
# Stage 2: 物理稳定性终止条件 (替代 mimic 相关的终止条件)
# =============================================================================


def robot_falling(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    height_threshold: float = 0.3,
    tilt_threshold: float = 0.7,
) -> torch.Tensor:
    """
    [Stage 2 专用] 机器人摔倒检测
    
    摔倒判定条件 (任一满足即终止):
    1. 高度过低: 骨盆/Root 高度 < height_threshold (默认 0.3m)
    2. 过度倾斜: 躯干 Z 轴与世界 Z 轴夹角过大 (dot < tilt_threshold)
    
    设计目的:
    - 替代 Stage 1 的 anchor_pos, anchor_ori, ee_body_pos 等 mimic 终止条件
    - Stage 2 不再强约束跟踪参考动作，但仍需保证物理稳定性
    - 防止机器人学到会摔倒的策略
    
    Args:
        env: 环境实例
        asset_cfg: 机器人资产配置
        height_threshold: 骨盆最低允许高度 (米)
        tilt_threshold: 躯干倾斜余弦阈值 (1=竖直, 0=水平)
    
    Returns:
        Tensor (num_envs,): True 表示摔倒，需要终止
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取骨盆 (Root body) 的位置
    root_pos_w = asset.data.root_pos_w  # (num_envs, 3)
    root_height = root_pos_w[:, 2]  # Z 轴高度
    
    # 获取躯干朝向 (检查是否倾斜过度)
    root_quat_w = asset.data.root_quat_w  # (num_envs, 4)
    
    # 计算躯干 Z 轴在世界坐标系中的方向
    # 局部 Z 轴 = [0, 0, 1]
    local_up = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).repeat(env.num_envs, 1)
    world_up_body = math_utils.quat_apply(root_quat_w, local_up)  # (num_envs, 3)
    
    # 与世界 Z 轴的点积 (1=完全竖直, 0=水平, -1=倒立)
    dot_with_world_up = world_up_body[:, 2]  # 直接取 Z 分量
    
    # 摔倒条件:
    # 1. 高度过低
    height_too_low = root_height < height_threshold
    # 2. 过度倾斜
    too_tilted = dot_with_world_up < tilt_threshold
    
    return height_too_low | too_tilted
