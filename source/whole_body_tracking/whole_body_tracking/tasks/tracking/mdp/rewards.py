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
    
    Hit 条件 (简化版 - 只看距离):
    - effector 到 target 的距离 < hit_distance_threshold (0.06m)
    - 任务奖励处于生效状态 (不在 Hit 后的 1s 等待期)
    
    Hit 后机制:
    - 不立即重采样，启动 1s 延迟计时器
    - 这 1s 内任务奖励失效 (激励跟随参考动作收手，获得 Mimic 奖励)
    - 1s 后自动重采样目标位置
    
    Returns:
        Tensor (num_envs,): 1.0 表示有效 Hit，0.0 表示未 Hit
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 更新 Hit 后延迟重采样计时器 (必须在 check_hit 之前调用)
    command.update_hit_resample_timer()
    
    # 调用 check_hit 方法进行检测
    hit_mask, _ = command.check_hit()
    
    # 更新课程学习状态
    command.update_curriculum(hit_mask)
    
    return hit_mask.float()


def effector_target_near(
    env: ManagerBasedRLEnv,
    command_name: str,
    guidance_radius: float = 0.25,
    scale: float = 10.0,
) -> torch.Tensor:
    """
    [进展奖励] 只有当 effector 比历史最近距离更近时才给奖励
    
    核心设计:
    - 维护 min_distance_to_target: 当前 Hit 周期内的历史最近距离
    - 只有当 current_distance < min_distance_to_target 时才给奖励
    - 奖励 = (min_distance - current_distance) * scale
    - 更新 min_distance = current_distance
    
    行为分析:
    - 手停在原地: 没有奖励 (距离没变近)
    - 手绕圈: 没有奖励 (距离没变近)
    - 手向目标移动: 有奖励 (奖励与接近量成正比)
    - 手 Hit 到目标: 之后不可能更近，Near 奖励自然归零
    
    这完美解决了蹭分问题！
    
    触发条件:
    1. effector 在引导大球范围内 (distance < guidance_radius)
    2. current_distance < min_distance_to_target
    3. 任务奖励处于生效状态
    
    重置条件:
    - 目标小球被重采样时 (Hit 后 1s 或 episode 重置)
    - min_distance_to_target 被重置为 10.0 (大值)
    
    Args:
        guidance_radius: 引导大球半径 (米)
        scale: 奖励放大系数 (每接近 1cm 奖励 = 0.01 * scale)
    
    Returns:
        Tensor (num_envs,): 进展奖励，无进展时为 0
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 如果任务奖励未生效 (Hit 后等待期)，返回 0
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # 获取攻击肢体位置
    effector_pos_w = command.robot_body_pos_w[:, command.effector_index]
    
    # 计算到目标的距离
    current_distance = torch.norm(effector_pos_w - command.target_pos_w, dim=-1)
    
    # 检查是否在引导球范围内
    in_guidance_sphere = current_distance < guidance_radius
    
    # 更新引导球进入状态 (用于监控)
    command.has_entered_guidance_sphere = command.has_entered_guidance_sphere | in_guidance_sphere
    
    # 检查是否有进展 (比历史最近距离更近)
    has_progress = current_distance < command.min_distance_to_target
    
    # 综合条件: 在引导球内 + 有进展 + 任务奖励生效
    should_reward = in_guidance_sphere & has_progress & command.task_rewards_enabled
    
    # 计算进展奖励
    progress = command.min_distance_to_target - current_distance  # 接近了多少
    reward = torch.where(
        should_reward,
        progress * scale,
        torch.zeros_like(current_distance)
    )
    
    # 更新 min_distance (只在任务奖励生效时更新)
    command.min_distance_to_target = torch.where(
        should_reward,
        current_distance,
        command.min_distance_to_target
    )
    
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
    
    注意:
    - 在 Hit 后的 1s 等待期内不计算此奖励 (task_rewards_enabled = False)
    
    Returns:
        Tensor (num_envs,): [0, 1] 范围的奖励
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 如果任务奖励未生效 (Hit 后等待期)，返回 0
    reward = torch.zeros(env.num_envs, device=env.device)
    
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
    face_reward = 0.5 * (dot + 1.0)
    
    # 只在任务奖励生效时给予奖励
    reward = torch.where(
        command.task_rewards_enabled,
        face_reward,
        torch.zeros_like(face_reward)
    )
    
    return reward


# pen_touch_lazy 已删除: 进展奖励机制已解决蹭分问题，不需要额外惩罚


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


# =============================================================================
# Stage 2: 收手阶段奖励/惩罚 (Hit 后冷静期内)
# 
# 设计原则:
# - 只在 Hit 后的冷静期 (task_rewards_enabled = False) 内生效
# - 冷静期前 0.2s 是宽限期，不触发这些奖励/惩罚
# - 鼓励机器人收手，获得 Mimic 奖励
# =============================================================================


def pen_linger_in_hit_sphere(
    env: ManagerBasedRLEnv,
    command_name: str,
    grace_period: float = 0.2,
) -> torch.Tensor:
    """
    [收手惩罚-小球] Hit 后冷静期内，手停留在目标小球内的惩罚
    
    触发条件:
    1. 处于冷静期 (task_rewards_enabled = False)
    2. 冷静期已过宽限期 (hit_resample_timer < hit_resample_delay - grace_period)
    3. effector 在目标小球内 (distance < hit_distance_threshold)
    
    机制:
    - Hit 后 0~0.2s: 宽限期，允许拳头穿越目标，不惩罚
    - Hit 后 0.2~1.0s: 如果手还在目标小球内，给予常量惩罚
    
    设计目的:
    - 强制机器人收手，不能把手放在目标上
    - 与 Mimic 奖励配合，引导跟随参考动作
    
    Returns:
        Tensor (num_envs,): 惩罚值 (-1.0 或 0.0)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    # 计算已过宽限期的条件
    # hit_resample_timer 从 1.0 递减到 0
    # 过宽限期 = timer < (delay - grace_period) = timer < 0.8
    past_grace_period = command.hit_resample_timer < (command.hit_resample_delay - grace_period)
    
    # 在冷静期内 (task_rewards_enabled = False)
    in_cooldown = ~command.task_rewards_enabled
    
    # 获取距离
    effector_pos_w = command.robot_body_pos_w[:, command.effector_index]
    distance = torch.norm(effector_pos_w - command.target_pos_w, dim=-1)
    
    # 在目标小球内
    in_hit_sphere = distance < command.hit_distance_threshold
    
    # 综合条件: 冷静期 + 过宽限期 + 在小球内
    should_penalize = in_cooldown & past_grace_period & in_hit_sphere
    
    # 常量惩罚
    penalty = torch.where(should_penalize,
                          torch.ones_like(distance) * -1.0,
                          torch.zeros_like(distance))
    
    return penalty


def rew_retract_from_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    guidance_radius: float = 0.25,
    grace_period: float = 0.2,
    scale: float = 10.0,
) -> torch.Tensor:
    """
    [收手奖励-大球] Hit 后冷静期内，手离开目标的进展奖励 (与 Near 对称设计)
    
    触发条件:
    1. 处于冷静期 (task_rewards_enabled = False)
    2. 冷静期已过宽限期 (0.2s 后)
    3. effector 在引导大球内 (distance < guidance_radius)
    4. 当前距离 > 历史最远距离
    
    机制:
    - 进展奖励: 只有距离比历史最远更远时才给奖励
    - 线性奖励: reward = (current_distance - max_distance) * scale
    - 与进攻阶段的 Near 奖励完全对称
    
    奖励量分析:
    - 引导球半径 = 0.25m, Hit 时距离 ≈ 0
    - 手从 0m 收到引导球边缘 0.25m
    - 理论最大累积奖励 = 0.25 * scale = 2.5 (与 Near 相同)
    
    设计目的:
    - 激励机器人收手，远离目标
    - 与 Mimic 奖励配合，引导跟随参考动作
    - 与 Near 奖励对称，形成"进攻-收手"完整周期
    
    Returns:
        Tensor (num_envs,): 进展奖励，无进展时为 0
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # 计算已过宽限期的条件
    past_grace_period = command.hit_resample_timer < (command.hit_resample_delay - grace_period)
    
    # 在冷静期内
    in_cooldown = ~command.task_rewards_enabled
    
    # 获取距离
    effector_pos_w = command.robot_body_pos_w[:, command.effector_index]
    current_distance = torch.norm(effector_pos_w - command.target_pos_w, dim=-1)
    
    # 在引导大球内
    in_guidance_sphere = current_distance < guidance_radius
    
    # 检查是否有进展 (比历史最远更远)
    has_progress = current_distance > command.max_distance_from_target
    
    # 综合条件
    should_reward = in_cooldown & past_grace_period & in_guidance_sphere & has_progress
    
    # 计算进展奖励 (线性)
    progress = current_distance - command.max_distance_from_target
    reward = torch.where(
        should_reward,
        progress * scale,
        torch.zeros_like(current_distance)
    )
    
    # 更新历史最远距离 (只在冷静期内更新)
    command.max_distance_from_target = torch.where(
        should_reward,
        current_distance,
        command.max_distance_from_target
    )
    
    return reward

def effector_velocity_towards_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    guidance_radius: float = 0.25,
) -> torch.Tensor:
    """
    鼓励末端执行器速度方向朝向目标点
    只在引导球范围内且奖励生效时给奖励，否则为0
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    reward = torch.zeros(env.num_envs, device=env.device)

    # 末端位置与速度
    effector_pos_w = command.robot_body_pos_w[:, command.effector_index]
    effector_vel_w = command.robot_body_lin_vel_w[:, command.effector_index]

    # 目标方向
    dir_to_target = command.target_pos_w - effector_pos_w
    dir_to_target = dir_to_target / (torch.norm(dir_to_target, dim=-1, keepdim=True) + 1e-6)
    vel_norm = effector_vel_w / (torch.norm(effector_vel_w, dim=-1, keepdim=True) + 1e-6)

    # 计算点积（cosθ），归一化到[0,1]
    dot = torch.sum(vel_norm * dir_to_target, dim=-1)
    vel_reward = 0.5 * (dot + 1.0)

    # 只在引导球范围内且奖励生效时给奖励
    current_distance = torch.norm(effector_pos_w - command.target_pos_w, dim=-1)
    in_guidance_sphere = current_distance < guidance_radius
    should_reward = in_guidance_sphere & command.task_rewards_enabled

    reward = torch.where(should_reward, vel_reward, torch.zeros_like(vel_reward))
    return reward
# =================================================
# 对末端执行器的跟踪增强奖励
def mimic_right_hand_position_exp(env: ManagerBasedRLEnv, command_name: str, std: float = 0.1) -> torch.Tensor:
    """
    右手末端 mimic 位置奖励，鼓励机器人右手靠近参考动作的右手位置。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 使用实际 body_names 中的右臂/右手 link 名称
    right_hand_link_names = [
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ]
    missing = [name for name in right_hand_link_names if name not in command.cfg.body_names]
    assert not missing, f"Missing right hand link names in body_names: {missing}"
    idxs = [command.cfg.body_names.index(name) for name in right_hand_link_names]
    assert len(idxs) > 0, "No right hand link indexes found!"
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, idxs] - command.robot_body_pos_w[:, idxs]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    # 检查 reward 是否有 NaN/inf
    assert not torch.isnan(reward).any(), "mimic_right_hand_position_exp reward contains NaN!"
    assert not torch.isinf(reward).any(), "mimic_right_hand_position_exp reward contains Inf!"
    return reward

def mimic_right_hand_orientation_exp(env: ManagerBasedRLEnv, command_name: str, std: float = 0.2) -> torch.Tensor:
    """
    右手末端 mimic 姿态奖励，鼓励机器人右手朝向与参考动作一致。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    right_hand_link_names = [
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ]
    missing = [name for name in right_hand_link_names if name not in command.cfg.body_names]
    assert not missing, f"Missing right hand link names in body_names: {missing}"
    idxs = [command.cfg.body_names.index(name) for name in right_hand_link_names]
    assert len(idxs) > 0, "No right hand link indexes found!"
    error = quat_error_magnitude(command.body_quat_relative_w[:, idxs], command.robot_body_quat_w[:, idxs]) ** 2
    reward = torch.exp(-error.mean(-1) / std**2)
    assert not torch.isnan(reward).any(), "mimic_right_hand_orientation_exp reward contains NaN!"
    assert not torch.isinf(reward).any(), "mimic_right_hand_orientation_exp reward contains Inf!"
    return reward