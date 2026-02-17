from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_apply

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


# =============================================================================
# Stage 2: Task-Oriented 奖励函数
# =============================================================================


def effector_target_hit(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """
    [核心任务奖励] 有效击中目标的脉冲奖励
    
    Hit 条件 (必须同时满足):
    1. 距离达标: effector 到 target 的距离 < hit_distance_threshold (0.06m)
    2. 速度达标: effector 的速度 > hit_speed_threshold (0.5 m/s)
       注意: 使用绝对速度模长，不用投影速度，因为摆拳的切向速度很大
    3. 冷却达标: current_time - last_hit_time > hit_cooldown (0.5s)
    
    设计目的:
    - 这是整个 Stage 2 训练的**核心目标**
    - 速度门控区分"有效打击"和"轻轻触碰"
    - 冷却机制适配"5连击"长序列：迫使机器人击中后收手蓄力
    
    Returns:
        Tensor (num_envs,): 1.0 表示有效 Hit，0.0 表示未 Hit
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 调用 check_hit 方法进行检测
    # 该方法会同时更新 last_hit_time
    hit_mask, _, _ = command.check_hit()
    
    # 更新课程学习状态
    command.update_curriculum(hit_mask)
    
    return hit_mask.float()


def effector_target_hit_velocity_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_threshold: float = 0.5,
) -> torch.Tensor:
    """
    [速度奖励] 在 Hit 时刻根据速度给予额外奖励
    
    逻辑:
    - 只在 Hit 成功时给予奖励
    - 奖励值 = (speed - threshold) / threshold，归一化后的超额速度
    - 速度越快，奖励越高 (但 clamp 防止过大)
    
    设计目的:
    - 鼓励机器人打得更有力
    - 权重设置为 1.0 左右，避免为了速度牺牲稳定性
    
    Returns:
        Tensor (num_envs,): 速度奖励，只在 Hit 时非零
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 获取攻击肢体的速度
    effector_vel_w = command.robot_body_lin_vel_w[:, command.effector_index]
    speeds = torch.norm(effector_vel_w, dim=-1)
    
    # 获取攻击肢体的位置
    effector_pos_w = command.robot_body_pos_w[:, command.effector_index]
    distances = torch.norm(effector_pos_w - command.target_pos_w, dim=-1)
    
    # 检查是否在 Hit 范围内且速度达标
    dist_ok = distances < command.hit_distance_threshold
    speed_ok = speeds > speed_threshold
    
    # 计算超额速度奖励 (归一化)
    extra_speed = (speeds - speed_threshold) / speed_threshold
    extra_speed = torch.clamp(extra_speed, min=0.0, max=3.0)  # 最多 3 倍奖励
    
    # 只在 Hit 条件满足时给予奖励
    reward = torch.where(dist_ok & speed_ok, extra_speed, torch.zeros_like(extra_speed))
    
    return reward


def effector_target_near(
    env: ManagerBasedRLEnv,
    command_name: str,
    guidance_radius: float = 0.25,
) -> torch.Tensor:
    """
    [引导奖励] 引导攻击肢体靠近目标
    
    公式: r = exp(-distance² / (2 * σ²))
    其中 σ = guidance_radius，对应引导大球半径
    
    设计目的:
    - 解决稀疏 Hit 奖励难以探索的问题 (Reward Shaping)
    - 提供平滑梯度，引导肢体进入"引导球"范围
    - 一旦进入范围，机器人更容易触发 Hit 奖励
    
    数学分析:
    - d=0 时，r=1.0 (最大)
    - d=σ 时，r≈0.61
    - d=2σ 时，r≈0.14
    
    Returns:
        Tensor (num_envs,): [0, 1] 范围的奖励
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 获取攻击肢体位置
    effector_pos_w = command.robot_body_pos_w[:, command.effector_index]
    
    # 计算到目标的距离
    distance = torch.norm(effector_pos_w - command.target_pos_w, dim=-1)
    
    # 高斯形式的奖励
    sigma = guidance_radius
    reward = torch.exp(-distance**2 / (2 * sigma**2))
    
    return reward


def effector_face_target(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """
    [战术奖励] 鼓励机器人躯干朝向目标
    
    逻辑:
    1. 计算躯干前向向量 (从 root 到 torso 的方向或 torso 的朝向)
    2. 计算机器人到目标的方向向量
    3. 计算两者的点积 (cos θ)
    4. 映射到 [0, 1]: r = 0.5 * (dot + 1.0)
    
    设计目的:
    - 战术姿态约束：正对目标有利于发力、防守、保持平衡
    - 防止机器人扭曲身体去够背后的目标
    - 使动作更拟人
    
    数学分析:
    - 完全正对时，dot=1, r=1.0
    - 垂直时，dot=0, r=0.5
    - 完全背对时，dot=-1, r=0.0
    
    Returns:
        Tensor (num_envs,): [0, 1] 范围的奖励
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 获取机器人 Root (anchor) 的位置和朝向
    root_pos_w = command.robot_anchor_pos_w  # (num_envs, 3)
    root_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # 计算前向向量 (在 Isaac Lab 中，局部 X 轴是前向)
    # 将局部前向向量 [1, 0, 0] 转换到世界坐标系
    local_forward = torch.tensor([[1.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
    world_forward = quat_apply(root_quat_w, local_forward)  # (num_envs, 3)
    
    # 计算机器人到目标的方向向量 (只考虑水平方向)
    dir_to_target = command.target_pos_w - root_pos_w  # (num_envs, 3)
    dir_to_target[:, 2] = 0  # 忽略高度差
    dir_to_target = dir_to_target / (torch.norm(dir_to_target, dim=-1, keepdim=True) + 1e-6)
    
    # 前向向量也只考虑水平方向
    world_forward[:, 2] = 0
    world_forward = world_forward / (torch.norm(world_forward, dim=-1, keepdim=True) + 1e-6)
    
    # 计算点积 (cos θ)
    dot = torch.sum(world_forward * dir_to_target, dim=-1)  # (num_envs,)
    
    # 映射到 [0, 1]
    reward = 0.5 * (dot + 1.0)
    
    return reward


def pen_touch_lazy(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """
    [反蹭分惩罚] 惩罚手在目标附近但速度很慢的行为
    
    逻辑:
    IF (distance < hit_threshold) AND (speed < hit_speed_threshold)
    THEN penalty = -1.0
    ELSE penalty = 0.0
    
    设计目的:
    - 防止 RL 发现"手放在球里虽然没 Hit 但有靠近奖励"的漏洞
    - 强制机器人：击中后必须迅速离开
    - 赖在目标上不动会持续扣分
    
    Returns:
        Tensor (num_envs,): -1.0 表示惩罚，0.0 表示无惩罚
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 获取攻击肢体位置和速度
    effector_pos_w = command.robot_body_pos_w[:, command.effector_index]
    effector_vel_w = command.robot_body_lin_vel_w[:, command.effector_index]
    
    # 计算距离和速度
    distance = torch.norm(effector_pos_w - command.target_pos_w, dim=-1)
    speed = torch.norm(effector_vel_w, dim=-1)
    
    # 检查是否"蹭分"
    in_range = distance < command.hit_distance_threshold
    too_slow = speed < command.hit_speed_threshold
    
    # 在范围内但速度慢 = 惩罚
    penalty = torch.where(in_range & too_slow, 
                          torch.ones_like(distance) * -1.0,
                          torch.zeros_like(distance))
    
    return penalty


def posture_unstable(
    env: ManagerBasedRLEnv,
    command_name: str,
    tilt_threshold: float = 0.5,  # 约 30 度
) -> torch.Tensor:
    """
    [姿态惩罚] 惩罚身体过度倾斜
    
    逻辑:
    - 计算躯干的倾斜程度 (相对于竖直方向的角度)
    - 如果超过阈值，给予惩罚
    - 惩罚与超出量成比例
    
    设计目的:
    - 物理稳定性：防止机器人过度倾斜导致跌倒
    - 防止为追求极端攻击距离而牺牲平衡
    
    数学:
    - 计算躯干 Z 轴与世界 Z 轴的夹角
    - tilt = 1 - dot(torso_up, world_up) ∈ [0, 2]
    - tilt > threshold 时开始惩罚
    
    Returns:
        Tensor (num_envs,): 惩罚值，正常姿态为 0
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 获取躯干朝向
    root_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # 计算躯干的上向量 (局部 Z 轴)
    local_up = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).repeat(env.num_envs, 1)
    world_up_body = quat_apply(root_quat_w, local_up)  # (num_envs, 3)
    
    # 世界 Z 轴
    world_up = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).repeat(env.num_envs, 1)
    
    # 计算点积 (1 = 完全竖直，0 = 水平，-1 = 倒立)
    dot = torch.sum(world_up_body * world_up, dim=-1)  # (num_envs,)
    
    # 倾斜程度 = 1 - dot ∈ [0, 2]
    tilt = 1.0 - dot
    
    # 超过阈值的部分给予惩罚
    penalty = torch.where(tilt > tilt_threshold,
                          (tilt - tilt_threshold),
                          torch.zeros_like(tilt))
    
    return -penalty  # 返回负值作为惩罚
