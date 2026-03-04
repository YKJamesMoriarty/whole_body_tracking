from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_lin_vel_w.view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_ang_vel_w.view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


# =============================================================================
# 拳击机器人任务导向观测 (Stage 2: Task-Oriented RL)
# 这些观测为击打目标点任务提供必要信息
# =============================================================================


def target_relative_position(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    (1) 目标相对位置 (3 维): [x, y, z]
    
    坐标系设计说明:
    ================
    目标位置表示在 **机器人局部坐标系 (Body Frame)** 中:
    - 原点: 机器人 Root (Pelvis/骨盆)
    - X轴: 机器人正前方
    - Y轴: 机器人左侧
    - Z轴: 机器人上方
    
    Stage 2:
    - target = 课程学习随机采样的目标点 (command.target_pos_w)
    - 与 Hit 检测和可视化使用相同的数据源
    
    Args:
        env: 环境实例
        command_name: 动作命令名称
    
    Returns:
        Tensor (num_envs, 3): 目标在机器人局部坐标系中的位置 [x_前后, y_左右, z_上下]
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # Stage 2: real target exists all the time, but observation is visible only briefly
    # at episode start. After that (or after hit), observation is hidden to simulate
    # camera losing target.
    target_pos_w = command.target_pos_w
    
    # 获取机器人 Root (Pelvis) 的世界坐标位置和朝向
    robot_root_pos_w = command.robot_anchor_pos_w   # (num_envs, 3)
    robot_root_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # 将目标位置从世界坐标系转换到机器人局部坐标系
    # 数学公式: p_local = R_robot^(-1) * (p_target_world - p_robot_world)
    target_pos_b_visible, _ = subtract_frame_transforms(
        robot_root_pos_w,
        robot_root_quat_w,
        target_pos_w,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1),
    )

    hidden_obs = command.hidden_target_obs_local.unsqueeze(0).repeat(env.num_envs, 1)
    visible_mask = command.target_is_visible.unsqueeze(1)
    return torch.where(visible_mask, target_pos_b_visible.view(env.num_envs, 3), hidden_obs)


def target_relative_velocity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    (2) 目标相对速度 (3 维): [vx, vy, vz]
    
    含义: 目标点相对于机器人 Root 的速度（在局部坐标系中）
    
    Stage 1: 设为零向量（假设目标是静止的）
    
    Stage 2 TODO: 
    - 如果目标会移动（例如对手在闪避），需要计算实际的相对速度
    - 速度也需要转换到机器人局部坐标系
    
    Returns:
        Tensor (num_envs, 3): 目标速度 [vx, vy, vz]，Stage 1 全为零
    """
    # Stage 1: 目标静止，相对速度为零
    return torch.zeros(env.num_envs, 3, device=env.device)


def strikes_left(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    (3) 累积 Hit 次数 (1 维)
    
    含义: 当前 episode 内成功 Hit 目标的累积次数
    
    Stage 1: 常数 1.0 (模型训练时使用固定值)
    
    Stage 2: 返回累积 Hit 次数，让 Critic 区分不同阶段:
    - cumulative_hit_count = 0: 还没 Hit 过，进攻阶段
    - cumulative_hit_count = 1: 已 Hit 1 次，可能在冷静期或第二次进攻
    - cumulative_hit_count = 2: 已 Hit 2 次，以此类推
    
    设计目的:
    - 这是客观的物理事实，不是人工 flag
    - Critic 可以学到: hit_count=0 且手靠近目标 → V(s) 高
    - Critic 可以学到: hit_count=1 且手靠近目标 → V(s) 取决于冷静期状态
    
    注意:
    - 名称保持 "strikes_left" 以兼容 Stage 1 模型加载
    - 值的含义从 "剩余次数" 变为 "累积次数"，但不影响模型结构
    
    Returns:
        Tensor (num_envs, 1): 累积 Hit 次数
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.cumulative_hit_count.view(env.num_envs, 1)


def time_left(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    (4) 剩余时间 (1 维)
    
    含义: 当前攻击窗口的归一化剩余时间
    
    Stage 1: 设为常数 1.0（满时间窗口）
    
    Stage 2 TODO:
    - 根据攻击窗口计算实际剩余时间
    - 例如：攻击窗口2秒，已过0.5秒，则返回 1.5/2.0 = 0.75
    
    Returns:
        Tensor (num_envs, 1): 归一化的剩余时间
    """
    return torch.ones(env.num_envs, 1, device=env.device)


def active_effector_one_hot(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    (5) 活跃攻击肢体 (4 维) - One-Hot 编码
    
    含义: 指示当前正在使用哪个肢体进行攻击
    格式: [左手, 右手, 左脚, 右脚]
    
    示例:
    - 右手出拳: [0, 1, 0, 0]
    - 左脚踢击: [0, 0, 1, 0]
    
    Stage 1: 硬编码为 "右手" = [0, 1, 0, 0]
    
    Stage 2 TODO:
    - 根据当前技能/动作类型动态设置
    - 从动作文件名或技能命令中解析
    - 例如: "hook_left" -> 左手, "roundhouse_right" -> 右脚
    
    Returns:
        Tensor (num_envs, 4): One-Hot 编码的活跃肢体
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.active_effector_one_hot


def skill_type_one_hot(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    (6) 技能类型 (16 维) - One-Hot 编码
    
    含义: 指示当前正在执行的技能/招式类型
    
    为什么是16维?
    - 预留足够的维度给未来可能的所有技能
    - 避免后续修改网络架构
    - 支持拳法、腿法、组合技等多种类型
    
    Stage 1: 硬编码为 "直拳Jab" (索引0)
    
    Stage 2 TODO:
    - 根据当前技能命令或动作类型动态设置
    - 从动作文件名解析 (例如: "cross_right_normal" -> Cross)
    
    Returns:
        Tensor (num_envs, 16): One-Hot 编码的技能类型
    """
    # Keep this channel but zero it out to avoid meaningless fixed features.
    return torch.zeros(env.num_envs, 16, device=env.device)
