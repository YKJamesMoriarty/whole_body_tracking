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

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


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
# 拳击机器人任务导向观测 (Stage 1: Mimic 模仿训练)
# 这些观测是为 Stage 2: Task-Oriented RL (击打目标点) 做准备
# 在 Stage 1 中，我们使用"假的但有关联的数据"，让网络学习有意义的表征
# =============================================================================


def target_relative_position(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    (1) 目标相对位置 (3 维): [x, y, z]，在机器人局部坐标系 (Body Frame) 中

    mobility 分支 (stance 类技能):
    - 没有实际攻击目标，固定返回局部坐标 (0, 0, -10)
    - 让网络学到"这个位置 = 当前无目标"的语义
    - z=-10 表示目标在地面深处，不会与实际身体部位混淆

    Returns:
        Tensor (num_envs, 3): 固定值 [0, 0, -10]，表示无目标
    """
    # stance 技能: 无攻击目标，固定返回"地下"位置作为"无目标"信号
    result = torch.zeros(env.num_envs, 3, device=env.device)
    result[:, 2] = -10.0  # z 轴方向向下 10m，表示"无目标"
    return result


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
    (3) 剩余攻击次数 (1 维)

    mobility 分支 (stance 类技能):
    - 无攻击任务，固定返回 0.0 表示"无剩余攻击次数"

    Stage 2 TODO (攻击技能分支):
    - 跟踪实际剩余攻击次数并归一化

    Returns:
        Tensor (num_envs, 1): 0.0 表示无攻击次数
    """
    return torch.zeros(env.num_envs, 1, device=env.device)


def time_left(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    (4) 剩余时间 (1 维)

    mobility 分支 (stance 类技能):
    - 无进攻时间窗口，固定返回 0.0 表示"不在攻击窗口内"

    Stage 2 TODO (攻击技能分支):
    - 根据攻击窗口计算实际归一化剩余时间

    Returns:
        Tensor (num_envs, 1): 0.0 表示不在攻击窗口内
    """
    return torch.zeros(env.num_envs, 1, device=env.device)


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
    # mobility 分支 (stance 类技能): 无攻击肢体，返回全零 [0, 0, 0, 0]
    # 全零语义: "当前无活跃攻击肢体"
    # Stage 2 TODO: 根据动作文件名或技能命令动态确定活跃肢体
    return torch.zeros(env.num_envs, 4, device=env.device)


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
    # 技能索引对照表 (16 维):
    # ===== 拳法 (0-5) =====
    # 0: r-Cross (右直拳)
    # 1: r-swing (右摆拳)
    # 2: roundhouse_right_normal_low (右-低位鞭腿)
    # 3: roundhouse_right_fast_high (右-高位鞭腿)
    # 4: frontkick_right_normal_body (右脚前蹬)

    # 5: stance_orthodox_idle (摆架子/右势))
    # ===== 腿法 (6-11) =====
    # 6: LowKick (低扫腿)
    # 7: MidKick (中段踢)
    # 8: HighKick (高踢)
    # 9: FrontKick (前踢)
    # 10: SideKick (侧踢)
    # 11: RoundhouseKick (回旋踢)
    # ===== 组合/特殊 (12-15) =====
    # 12: Combo1 (组合1)
    # 13: Combo2 (组合2)
    # 14: 预留
    # 15: 预留
    
    one_hot = torch.zeros(env.num_envs, 16, device=env.device)
    # 右直拳0，右摆拳1，右低位鞭腿2，右高位鞭腿3，右脚前蹬4
    one_hot[:, 5] = 1.0  # 例如 one_hot[:, 0] = 1.0 为 直拳 (Stage 1)
    
    # Stage 2 TODO: 从以下来源解析技能类型:
    #   - 动作文件名 (例如: "cross_right_normal" -> Cross)
    #   - 外部技能命令
    #   - Registry 元数据
    
    return one_hot
