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


def target_relative_position(env: ManagerBasedEnv, command_name: str, effector_body_name: str = "right_wrist_yaw_link") -> torch.Tensor:
    """
    (1) 目标相对位置 (3 维): [x, y, z]
    
    坐标系设计说明:
    ================
    目标位置表示在 **机器人局部坐标系 (Body Frame)** 中:
    - 原点: 机器人 Root (Pelvis/骨盆)
    - X轴: 机器人正前方
    - Y轴: 机器人左侧
    - Z轴: 机器人上方
    
    为什么不用世界坐标系?
    - 世界坐标系不随机器人移动旋转，不利于策略迁移
    - 局部坐标系让网络学到"目标在我前方1米"这样的相对概念
    
    为什么不相对于攻击肢体?
    - 不同肢体(左手/右手/左脚/右脚)攻击时，参考系会变化
    - 相对于 Root 统一且稳定，无论用哪个肢体攻击都一样
    
    Stage 1 技巧:
    - 我们把当前参考动作中"攻击肢体"的目标位置作为 dummy target
    - 这样网络学到：当肢体移动到 target 位置时，是正确的模仿状态
    
    Stage 2 TODO:
    - 替换为场景中实际的目标点位置 (例如：对手的头部/身体位置)
    
    Args:
        env: 环境实例
        command_name: 动作命令名称
        effector_body_name: 攻击肢体名称，用于 Stage 1 生成 dummy target
                           可选: "right_wrist_yaw_link", "left_wrist_yaw_link",
                                 "right_ankle_roll_link", "left_ankle_roll_link"
    
    Returns:
        Tensor (num_envs, 3): 目标在机器人局部坐标系中的位置 [x_前后, y_左右, z_上下]
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 获取攻击肢体在 body_names 列表中的索引
    # 注意: body_names 定义在 flat_env_cfg.py 中
    try:
        effector_index = command.cfg.body_names.index(effector_body_name)
    except ValueError:
        # 如果找不到指定的肢体名称，默认使用最后一个 body (通常是手腕)
        effector_index = -1
    
    # Stage 1: 使用参考动作中该肢体的世界坐标位置作为 "目标"
    # 这是当前参考帧中该肢体应该到达的位置
    target_pos_w = command.body_pos_w[:, effector_index]  # (num_envs, 3) 世界坐标
    
    # 获取机器人 Root (Pelvis) 的世界坐标位置和朝向
    robot_root_pos_w = command.robot_anchor_pos_w   # (num_envs, 3)
    robot_root_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # 将目标位置从世界坐标系转换到机器人局部坐标系
    # 数学公式: p_local = R_robot^(-1) * (p_target_world - p_robot_world)
    # subtract_frame_transforms: 计算 target 相对于 robot_root 的局部坐标
    target_pos_b, _ = subtract_frame_transforms(
        robot_root_pos_w,
        robot_root_quat_w,
        target_pos_w,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1),  # 目标朝向无所谓，用单位四元数
    )
    
    return target_pos_b.view(env.num_envs, 3)


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
    
    含义: 归一化的剩余攻击次数，用于让策略知道还能打几拳/几脚
    
    Stage 1: 设为常数 1.0（满攻击次数）
    
    Stage 2 TODO:
    - 跟踪实际剩余攻击次数并归一化
    - 例如：总共3次攻击机会，已用1次，则返回 2/3 ≈ 0.67
    
    Returns:
        Tensor (num_envs, 1): 归一化的剩余攻击次数
    """
    return torch.ones(env.num_envs, 1, device=env.device)


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
    # Stage 1: 假设所有攻击都是右手出拳
    # 格式: [左手, 右手, 左脚, 右脚]
    one_hot = torch.zeros(env.num_envs, 4, device=env.device)
    one_hot[:, 1] = 1.0  # 右手激活
    
    # Stage 2 TODO: 根据以下信息动态确定活跃肢体:
    #   - 动作文件名 (例如: "cross" -> 右手, "hook_left" -> 左手)
    #   - 技能命令输入
    #   - 当前动作的阶段
    
    return one_hot


def skill_type_one_hot(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """
    (6) 技能类型 (8 维) - One-Hot 编码
    
    含义: 指示当前正在执行的技能/招式类型
    格式: [直拳Jab, 交叉拳Cross, 摆拳Hook, 上勾拳Uppercut, 
           低扫腿LowKick, 中段踢MidKick, 预留1, 预留2]
    
    为什么是8维?
    - 当前只定义了4-6种技能
    - 预留额外维度给未来可能的技能 (前踢FrontKick, 侧踢SideKick 等)
    - 避免后续修改网络架构
    
    Stage 1: 硬编码为 "直拳Jab" (索引0) = [1, 0, 0, 0, 0, 0, 0, 0]
    
    Stage 2 TODO:
    - 根据当前技能命令或动作类型动态设置
    - 从动作文件名解析 (例如: "cross_right_normal" -> Cross)
    
    Returns:
        Tensor (num_envs, 8): One-Hot 编码的技能类型
    """
    # 技能索引对照表:
    # 0: Jab (直拳)
    # 1: Cross (交叉拳)
    # 2: Hook (摆拳)
    # 3: Uppercut (上勾拳)
    # 4: LowKick (低扫腿)
    # 5: MidKick (中段踢)
    # 6: 预留 (例如 FrontKick 前踢)
    # 7: 预留 (例如 SideKick 侧踢)
    
    one_hot = torch.zeros(env.num_envs, 8, device=env.device)
    one_hot[:, 0] = 1.0  # 默认: 直拳 (Stage 1)
    
    # Stage 2 TODO: 从以下来源解析技能类型:
    #   - 动作文件名 (例如: "cross_right_normal" -> Cross)
    #   - 外部技能命令
    #   - Registry 元数据
    
    return one_hot
