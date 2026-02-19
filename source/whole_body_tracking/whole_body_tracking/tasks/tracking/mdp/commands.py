from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
import isaaclab.sim as sim_utils

# Debug draw 用于绘制线框
try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    HAS_DEBUG_DRAW = True
except ImportError:
    HAS_DEBUG_DRAW = False

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# 目标小球采样区域配置 (相对于机器人 Root/Pelvis 的局部坐标系)
# 基于 Unitree G1 机器人的尺寸设计，适合拳击攻击动作
# =============================================================================
DEFAULT_TARGET_SAMPLING_RANGE = {
    "x": (0.6, 0.65),   # 前方 60-65cm，有效攻击距离
    "y": (-0.4, 0.4),    # 左右 ±40cm，覆盖左右出拳
    "z": (0.25, 0.5),    # 高度 25-50cm（胸部到下巴高度）
}


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        
        # =====================================================================
        # Stage 2: 替换无意义的 sampling metrics
        # 
        # 原 Stage 1 metrics (已移除):
        # - sampling_entropy: 采样分布熵值 (Stage 2 始终为 0，无意义)
        # - sampling_top1_prob: 最高概率 (Stage 2 始终为 1.0，无意义)  
        # - sampling_top1_bin: 最可能的 bin (Stage 2 始终为 0，无意义)
        # 
        # Stage 2 新增 metrics:
        # - resample_count: 记录目标重采样次数 (用于监控)
        # - right_hand_to_target_dist: 右手到目标距离 (核心监控指标)
        # - effector_speed: 攻击肢体速度 (监控出拳速度)
        # =====================================================================
        self.metrics["resample_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["right_hand_to_target_dist"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["effector_speed"] = torch.zeros(self.num_envs, device=self.device)

        # =====================================================================
        # Stage 2: 目标小球 (Target Sphere) 初始化
        # 用于 Task-Oriented RL 的击打目标
        # =====================================================================
        
        # 目标小球的世界坐标位置 (num_envs, 3)
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # =====================================================================
        # Stage 2: 累积 Hit 计数器
        # 
        # 设计目的:
        # - 记录每个环境在当前 episode 内成功 Hit 的次数
        # - 用于更新 strikes_left 观测，让 Critic 区分不同阶段
        # - Episode 重置时清零，冷静期后重采样时不清零
        # =====================================================================
        self.cumulative_hit_count = torch.zeros(self.num_envs, device=self.device)
        
        # 获取参考动作第一帧的 Root (Pelvis) 位置和朝向
        # 这是采样区域的基准位置，在整个 episode 中保持不变
        self._reference_root_pos_w = self.motion.body_pos_w[0, 0].clone()  # (3,) 第一帧的 pelvis 位置
        self._reference_root_quat_w = self.motion.body_quat_w[0, 0].clone()  # (4,) 第一帧的 pelvis 朝向
        
        # 采样范围配置 (相对于机器人 Root 的局部坐标)
        self._target_sampling_range = self.cfg.target_sampling_range
        
        # 引导大球半径 (固定值)
        self._guidance_sphere_radius = self.cfg.guidance_sphere_radius
        
        # =====================================================================
        # Stage 2: Hit 检测状态 (每个 env 独立维护)
        # =====================================================================
        
        # 当前仿真时间 (每个 env 独立)
        # 由 _update_command 更新
        self.current_time = torch.zeros(self.num_envs, device=self.device)
        
        # Hit 距离阈值 (米) - 只用距离判断，不用速度
        self.hit_distance_threshold = self.cfg.hit_distance_threshold
        
        # 攻击肢体索引 (右手手腕)
        self.effector_body_name = self.cfg.effector_body_name
        self.effector_index = self.cfg.body_names.index(self.effector_body_name)
        
        # =====================================================================
        # Stage 2: Hit 后延迟重采样机制
        # 
        # 设计目的:
        # - Hit 后目标位置保持 1 秒不变
        # - 这 1 秒内任务奖励 (Hit/Near/Face/Speed) 全部失效
        # - 鼓励机器人在奖励消失后跟随参考动作收手 (拿 Mimic 奖励)
        # - 1 秒后重采样目标位置，任务奖励重新生效
        # =====================================================================
        
        # Hit 后等待重采样的剩余时间 (秒)
        # > 0 表示正在等待重采样 (任务奖励失效中)
        # <= 0 表示正常状态 (任务奖励生效)
        self.hit_resample_timer = torch.zeros(self.num_envs, device=self.device)
        
        # Hit 后到重采样的延迟时间 (秒)
        self.hit_resample_delay = self.cfg.hit_resample_delay
        
        # 任务奖励是否生效 (Hit/Near/Face/Speed)
        # False 表示在 Hit 后的等待期，任务奖励不计算
        self.task_rewards_enabled = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 单步时间 (用于累积)
        self.step_dt = self._env.step_dt
        
        # =====================================================================
        # Stage 2: 进展奖励状态 (Near 奖励)
        # 
        # 核心设计:
        # - 只有当 current_distance < min_historical_distance 时才给奖励
        # - 手停在原地: 没有奖励 (距离没变近)
        # - 手绕圈: 没有奖励 (距离没变近)
        # - 手向目标移动: 有奖励
        # - 手 Hit 到目标: 距离=0，之后不可能更近，Near 奖励自然归零
        # 
        # 这完美解决了蹭分问题！
        # =====================================================================
        
        # 当前 Hit 周期内，手到目标的历史最近距离
        # 只有比这个距离更近时才给 Near 奖励
        # Hit 后重采样时重置为一个大值
        self.min_distance_to_target = torch.full((self.num_envs,), 10.0, device=self.device)
        
        # 记录每个环境是否已经进入过引导大球范围
        # 用于判断是否开始计算进展奖励
        self.has_entered_guidance_sphere = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # =====================================================================
        # Stage 2: 收手阶段状态 (Hit 后冷静期内的奖励/惩罚)
        # 
        # 核心设计:
        # - Hit 后 0~0.2s: 宽限期，不给惩罚/奖励 (允许拳头穿越目标)
        # - Hit 后 0.2~1.0s: 激励收手
        #   - 惩罚: 手在目标小球内 → 常量惩罚
        #   - 奖励: 手离开目标越远越好 → 进展奖励 (与 Near 对称)
        # =====================================================================
        
        # 冷静期内的宽限时间 (秒)
        # Hit 后 0~grace_period 内不触发收手奖励/惩罚
        self.retract_grace_period = 0.2
        
        # 收手阶段的历史最远距离 (用于进展奖励)
        # 只有比这个距离更远时才给收手奖励
        # Hit 时重置为 0 (因为 Hit 时距离接近 0)
        self.max_distance_from_target = torch.zeros(self.num_envs, device=self.device)
        
        # =====================================================================
        # Stage 2: 课程学习状态
        # =====================================================================
        
        # 当前课程难度等级 (0.0 ~ 1.0)
        # 0.0 = 最简单 (小范围采样，接近中心点)
        # 1.0 = 最难 (全范围采样)
        self.curriculum_level = torch.tensor(0.0, device=self.device)
        
        # 课程升级所需的 Hit 成功率阈值
        self.curriculum_hit_rate_threshold = self.cfg.curriculum_hit_rate_threshold
        
        # 课程升级的步长
        self.curriculum_level_step = self.cfg.curriculum_level_step
        
        # 滑动窗口大小 (用于计算最近 N step 的 Hit 率)
        self.curriculum_window_size = self.cfg.curriculum_window_size
        
        # 滑动窗口: 记录最近 window_size 个 step 的 Hit 数量
        # 每个位置存储该 step 所有 env 的 Hit 总数
        self._hit_history = torch.zeros(self.curriculum_window_size, dtype=torch.long, device=self.device)
        self._hit_history_idx = 0  # 当前写入位置 (循环写入)
        self._hit_history_filled = False  # 窗口是否已填满
        
        # 初始采样范围 (Level 0) 和最终采样范围 (Level 1)
        self._init_sampling_range = self.cfg.init_sampling_range
        self._final_sampling_range = self.cfg.target_sampling_range
        
        # Hit 相关的 metrics
        self.metrics["hit_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_hit_count"] = torch.tensor(0.0, device=self.device)
        self.metrics["curriculum_level"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_hit_rate"] = torch.zeros(self.num_envs, device=self.device)
        
        # 初始化目标位置（会在 reset 时被重新采样）
        self._init_target_positions()

    def _init_target_positions(self):
        """初始化所有环境的目标位置"""
        # 在采样范围内随机采样目标位置
        self._resample_target_positions(torch.arange(self.num_envs, device=self.device))
    
    def _get_current_sampling_range(self) -> dict:
        """
        根据当前课程等级计算采样范围
        
        使用线性插值：
        range = init_range + curriculum_level * (final_range - init_range)
        
        Returns:
            dict: 当前课程等级对应的采样范围 {"x": (min, max), "y": ..., "z": ...}
        """
        level = self.curriculum_level.item()
        current_range = {}
        
        for axis in ["x", "y", "z"]:
            init_min, init_max = self._init_sampling_range[axis]
            final_min, final_max = self._final_sampling_range[axis]
            
            # 线性插值
            current_min = init_min + level * (final_min - init_min)
            current_max = init_max + level * (final_max - init_max)
            current_range[axis] = (current_min, current_max)
        
        return current_range
    
    def _resample_target_positions(self, env_ids: torch.Tensor):
        """
        为指定环境重新采样目标位置
        
        采样逻辑:
        1. 根据当前课程等级计算采样范围
        2. 在局部坐标系 (相对于参考动作第一帧的 Root) 中采样
        3. 将局部坐标转换到世界坐标系
        4. 加上各环境的 origin 偏移
        
        Args:
            env_ids: 需要重新采样的环境索引
        """
        if len(env_ids) == 0:
            return
        
        num_samples = len(env_ids)
        
        # 根据课程等级获取当前采样范围
        current_range = self._get_current_sampling_range()
        x_range = current_range["x"]
        y_range = current_range["y"]
        z_range = current_range["z"]
        
        # 更新内部状态以供可视化使用
        self._target_sampling_range = current_range
        
        # 随机采样 (num_samples, 3)
        local_target_pos = torch.zeros(num_samples, 3, device=self.device)
        local_target_pos[:, 0] = sample_uniform(x_range[0], x_range[1], (num_samples,), device=self.device)
        local_target_pos[:, 1] = sample_uniform(y_range[0], y_range[1], (num_samples,), device=self.device)
        local_target_pos[:, 2] = sample_uniform(z_range[0], z_range[1], (num_samples,), device=self.device)
        
        # 将局部坐标转换到世界坐标系
        # 使用参考动作第一帧的 Root 朝向进行旋转
        # world_pos = root_pos + quat_apply(root_quat, local_pos)
        world_target_pos = self._reference_root_pos_w + quat_apply(
            self._reference_root_quat_w.unsqueeze(0).repeat(num_samples, 1),
            local_target_pos
        )
        
        # 加上各环境的 origin 偏移
        world_target_pos = world_target_pos + self._env.scene.env_origins[env_ids]
        
        # 更新目标位置
        self.target_pos_w[env_ids] = world_target_pos
        
        # 重置被采样环境的任务奖励状态
        self.task_rewards_enabled[env_ids] = True
        self.hit_resample_timer[env_ids] = 0.0
        
        # 重置进展奖励状态
        # min_distance 重置为大值，让新目标可以触发进展奖励
        self.min_distance_to_target[env_ids] = 10.0
        
        # 重置引导球进入状态
        self.has_entered_guidance_sphere[env_ids] = False
        
        # 重置收手阶段状态 (重采样后进入进攻阶段，不需要收手)
        self.max_distance_from_target[env_ids] = 0.0
    
    def check_hit(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        检查是否发生有效 Hit
        
        新设计 (简化版):
        ==================
        Hit 条件: 只看距离，不看速度和冷却
        - effector 到 target 的距离 < hit_distance_threshold
        - 任务奖励处于生效状态 (不在 Hit 后的等待期)
        
        Hit 后机制:
        - 不立即重采样，而是启动 1 秒延迟计时器
        - 这 1 秒内任务奖励失效 (Hit/Near/Face/Speed)
        - 机器人在此期间跟随参考动作收手，获得 Mimic 奖励
        - 1 秒后自动重采样目标位置，任务奖励重新生效
        
        Returns:
            hit_mask: (num_envs,) bool, 哪些环境发生了有效 Hit
            distances: (num_envs,) float, effector 到 target 的距离
        """
        # 获取攻击肢体的位置
        effector_pos_w = self.robot_body_pos_w[:, self.effector_index]  # (num_envs, 3)
        
        # 计算距离
        distances = torch.norm(effector_pos_w - self.target_pos_w, dim=-1)  # (num_envs,)
        
        # Hit 条件: 距离达标 且 任务奖励生效中
        dist_ok = distances < self.hit_distance_threshold
        hit_mask = dist_ok & self.task_rewards_enabled
        
        # =====================================================================
        # Hit 成功后: 启动延迟重采样计时器，禁用任务奖励，初始化收手状态
        # =====================================================================
        hit_env_ids = torch.where(hit_mask)[0]
        if len(hit_env_ids) > 0:
            # 启动延迟计时器
            self.hit_resample_timer[hit_env_ids] = self.hit_resample_delay
            # 禁用任务奖励
            self.task_rewards_enabled[hit_env_ids] = False
            # 初始化收手阶段的历史最远距离 (从当前距离开始，因为 Hit 时距离接近 0)
            self.max_distance_from_target[hit_env_ids] = distances[hit_env_ids]
            
            # =========================================================================
            # Stage 2: Hit 成功后的观测更新
            # =========================================================================
            
            # (1) 递增累积 Hit 计数器
            # - 让 strikes_left 观测值变化，Critic 知道"我已经 Hit 过了"
            # - Episode 内累积，重置时清零
            self.cumulative_hit_count[hit_env_ids] += 1.0
            
            # (2) 将 target_pos_w 设置到地下深处 [0, 0, -10]
            # - 表示"当前没有攻击目标"
            # - 在机器人局部坐标系中，Z=-10 表示地下 10 米
            # - Critic 学到: target_pos 在地下 = 冷静期，没有 Hit 奖励
            # - 注意: 需要加上 env_origin 偏移
            underground_pos = torch.zeros(len(hit_env_ids), 3, device=self.device)
            underground_pos[:, 2] = -10.0  # Z = -10 (地下)
            self.target_pos_w[hit_env_ids] = underground_pos + self._env.scene.env_origins[hit_env_ids]
        
        return hit_mask, distances
    
    def update_hit_resample_timer(self):
        """
        更新 Hit 后的延迟重采样计时器
        
        每个 step 调用一次:
        - 减少计时器
        - 计时器归零时触发重采样并重新启用任务奖励
        """
        # 找出正在等待重采样的环境
        waiting_mask = self.hit_resample_timer > 0
        
        # 减少计时器
        self.hit_resample_timer = torch.where(
            waiting_mask,
            self.hit_resample_timer - self.step_dt,
            self.hit_resample_timer
        )
        
        # 检查哪些环境的计时器刚刚归零 (需要重采样)
        timer_just_expired = (self.hit_resample_timer <= 0) & waiting_mask
        resample_env_ids = torch.where(timer_just_expired)[0]
        
        if len(resample_env_ids) > 0:
            self._resample_target_positions(resample_env_ids)
    
    def update_curriculum(self, hit_mask: torch.Tensor):
        """
        更新课程学习状态 (使用滑动窗口统计)
        
        滑动窗口机制:
        ==============
        - 维护一个大小为 window_size (500) 的循环数组
        - 每个 step 记录所有 env 的 Hit 总数到当前位置
        - 位置指针循环前进，新数据自动覆盖最旧的数据
        - 始终统计最近 500 step 的 Hit 情况
        
        Hit 率计算:
        ===========
        hit_rate = total_hits / (window_size × num_envs)
        
        例如: window_size=500, num_envs=4
        - 总机会数 = 500 × 4 = 2000
        - 如果 500 step 内共 Hit 40 次
        - hit_rate = 40 / 2000 = 2%
        
        升级条件:
        ========
        1. 滑动窗口已填满 (至少运行了 500 step)
        2. Hit 成功率 >= threshold (默认 2%)
        3. 当前未满级
        
        Args:
            hit_mask: (num_envs,) bool, 本次 step 哪些环境发生了 Hit
        """
        # =====================================================================
        # 滑动窗口统计: 记录最近 window_size 个 step 的 Hit 情况
        # =====================================================================
        
        # 将本次 step 的 Hit 数量写入滑动窗口
        hit_count_this_step = hit_mask.sum().long()
        self._hit_history[self._hit_history_idx] = hit_count_this_step
        self._hit_history_idx = (self._hit_history_idx + 1) % self.curriculum_window_size
        
        # 检查窗口是否已填满 (第一轮循环完成后)
        if self._hit_history_idx == 0:
            self._hit_history_filled = True
        
        # 计算滑动窗口内的 Hit 成功率
        # Hit 率 = 窗口内总 Hit 次数 / (窗口大小 * num_envs)
        if self._hit_history_filled:
            # 窗口已满，使用完整窗口
            total_hits = self._hit_history.sum().item()
            total_opportunities = self.curriculum_window_size * self.num_envs
        else:
            # 窗口未满，只使用已填充部分
            total_hits = self._hit_history[:self._hit_history_idx].sum().item()
            total_opportunities = self._hit_history_idx * self.num_envs
        
        if total_opportunities > 0:
            current_hit_rate = total_hits / total_opportunities
        else:
            current_hit_rate = 0.0
        
        # 更新 metrics
        self.metrics["curriculum_level"][:] = self.curriculum_level.item()
        self.metrics["curriculum_hit_rate"][:] = current_hit_rate
        self.metrics["hit_count"][:] = hit_mask.float()
        # 更新 max_hit_count 指标
        self.metrics["max_hit_count"] = self.cumulative_hit_count.max().float()
        
        # =====================================================================
        # 课程升级判断
        # 条件: 滑动窗口已满 且 Hit 率 >= 阈值 且 未到满级
        # =====================================================================
        if (self._hit_history_filled and 
            current_hit_rate >= self.curriculum_hit_rate_threshold and
            self.curriculum_level < 1.0):
            
            # 升级!
            old_level = self.curriculum_level.item()
            self.curriculum_level = torch.clamp(
                self.curriculum_level + self.curriculum_level_step, 
                max=1.0
            )
            new_level = self.curriculum_level.item()
            
            # 重置滑动窗口 (新难度从零开始统计)
            self._hit_history.zero_()
            self._hit_history_idx = 0
            self._hit_history_filled = False
            
            # 打印升级信息
            print(f"\n{'='*60}")
            print(f">>> CURRICULUM LEVEL UP! {old_level:.2f} -> {new_level:.2f}")
            print(f">>> Hit Rate (last {self.curriculum_window_size} steps): {current_hit_rate:.2%}")
            new_range = self._get_current_sampling_range()
            print(f">>> New Sampling Range:")
            print(f">>>   X: [{new_range['x'][0]:.3f}, {new_range['x'][1]:.3f}]")
            print(f">>>   Y: [{new_range['y'][0]:.3f}, {new_range['y'][1]:.3f}]")
            print(f">>>   Z: [{new_range['z'][0]:.3f}, {new_range['z'][1]:.3f}]")
            print(f"{'='*60}\n")

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        
        # =====================================================================
        # Stage 2: 手部速度监控 (方法2 - 记录到 wandb)
        # 用于分析训练过程中手部速度的变化趋势
        # =====================================================================
        
        # 获取右手 (right_wrist_yaw_link) 的速度
        right_wrist_idx = self.cfg.body_names.index("right_wrist_yaw_link")
        right_hand_lin_vel_w = self.robot_body_lin_vel_w[:, right_wrist_idx]  # (num_envs, 3)
        right_hand_speed = torch.norm(right_hand_lin_vel_w, dim=-1)  # (num_envs,) 标量速度 m/s
        
        # 获取左手 (left_wrist_yaw_link) 的速度
        left_wrist_idx = self.cfg.body_names.index("left_wrist_yaw_link")
        left_hand_lin_vel_w = self.robot_body_lin_vel_w[:, left_wrist_idx]  # (num_envs, 3)
        left_hand_speed = torch.norm(left_hand_lin_vel_w, dim=-1)  # (num_envs,) 标量速度 m/s
        
        # 记录到 metrics (会自动上传到 wandb)
        self.metrics["right_hand_speed"] = right_hand_speed  # 右手速度 (m/s)
        self.metrics["left_hand_speed"] = left_hand_speed    # 左手速度 (m/s)
        self.metrics["max_hand_speed"] = torch.max(right_hand_speed, left_hand_speed)  # 双手最大速度
        
        # 计算手部到目标的距离 (用于分析 hit 奖励的触发条件)
        right_hand_pos_w = self.robot_body_pos_w[:, right_wrist_idx]  # (num_envs, 3)
        dist_to_target = torch.norm(right_hand_pos_w - self.target_pos_w, dim=-1)  # (num_envs,)
        self.metrics["right_hand_to_target_dist"] = dist_to_target  # 右手到目标的距离 (m)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """
        Stage 2 修改: 强制从动作第一帧开始训练
        
        原始 BeyondMimic 设计:
        - 将动作数据分成多个片段 (bin)
        - 统计每个片段的失败率
        - 失败率高的片段采样概率更高 (难点强化训练)
        - 随机选择一个片段的起始点
        
        Stage 2 修改原因:
        - 分片训练会导致机器人从动作中间开始 (比如收拳阶段)
        - 此时攻击目标不合理，因为收拳阶段不应该引导去打目标
        - cross 数据是原地出拳，第一帧是标准站姿，最适合作为起点
        
        修改方案:
        - 保留分片统计机制 (用于监控，指标仍然记录)
        - 但每次 reset 都从 time_steps = 0 开始
        - 这样每个 episode 都从标准站姿开始，目标采样位置稳定
        """
        # 保留原有的失败统计逻辑 (用于监控)
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # =====================================================================
        # Stage 2 核心修改: 强制从第一帧开始
        # 
        # 原 Stage 1 逻辑: 根据各 bin 的失败率自适应采样起始帧
        # Stage 2 逻辑: 固定从 frame 0 开始，机器人始终以准备姿态作为参考
        # 
        # 注意: sampling_entropy, sampling_top1_prob, sampling_top1_bin 这三个
        # metrics 在 Stage 2 中不再有意义 (始终是 0, 1.0, 0)，已移除。
        # 替代为 Stage 2 特有的 metrics (hit_count, curriculum_level 等)
        # =====================================================================
        self.time_steps[env_ids] = 0
        
        # Stage 2 专用 metrics: 记录采样事件
        # 由于固定从 frame 0 开始，这里只记录发生了重采样
        self.metrics["resample_count"][:] = self.metrics.get("resample_count", torch.zeros(self.num_envs, device=self.device)) + len(env_ids) / self.num_envs

    def _resample_command(self, env_ids: Sequence[int]):
        """Episode 重置时调用，重新采样动作和目标位置"""
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        
        # Stage 2: 重新采样目标位置
        self._resample_target_positions(torch.tensor(env_ids, device=self.device))
        
        # =====================================================================
        # Stage 2: Episode 重置时清零累积 Hit 计数器
        # 
        # 注意区分两种重采样:
        # - _resample_command(): Episode 重置，需要清零累积计数
        # - _resample_target_positions(): 冷静期后重采样，不清零累积计数
        # =====================================================================
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self.cumulative_hit_count[env_ids_tensor] = 0.0

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        
        # Stage 2: 更新当前仿真时间 (每个 env 独立)
        # dt = decimation * sim.dt
        dt = self._env.cfg.decimation * self._env.cfg.sim.dt
        self.current_time += dt
        
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()
        
        # =====================================================================
        # Stage 2: 更新监控 metrics
        # 每个 step 记录关键指标，用于 WandB 监控训练进度
        # =====================================================================
        
        # 计算右手到目标的距离
        effector_pos_w = self.robot_body_pos_w[:, self.effector_index]  # (num_envs, 3)
        effector_vel_w = self.robot_body_lin_vel_w[:, self.effector_index]  # (num_envs, 3)
        
        distances = torch.norm(effector_pos_w - self.target_pos_w, dim=-1)  # (num_envs,)
        speeds = torch.norm(effector_vel_w, dim=-1)  # (num_envs,)
        
        self.metrics["right_hand_to_target_dist"][:] = distances
        self.metrics["effector_speed"][:] = speeds

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )
                
                # =========================================================
                # Stage 2: 目标小球 Marker 初始化
                # =========================================================
                self.target_sphere_visualizer = VisualizationMarkers(
                    self.cfg.target_sphere_cfg.replace(prim_path="/Visuals/Command/target_sphere")
                )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)
            
            # Stage 2: 显示目标小球
            self.target_sphere_visualizer.set_visibility(True)
            
            # 获取 debug draw 接口
            if HAS_DEBUG_DRAW:
                self._debug_draw = omni_debug_draw.acquire_debug_draw_interface()

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)
            
            # Stage 2: 隐藏目标小球
            if hasattr(self, "target_sphere_visualizer"):
                self.target_sphere_visualizer.set_visibility(False)
            
            # 清除 debug lines
            if HAS_DEBUG_DRAW and hasattr(self, "_debug_draw"):
                self._debug_draw.clear_lines()

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        
        # =====================================================================
        # Stage 2: 手部速度实时打印 (方法1 - 用于 play 时观察)
        # 帮助设计奖励函数：了解正常出拳时手部能达到的速度范围
        # 设置完奖励函数后可以注释掉这段代码
        # =====================================================================
        right_wrist_idx = self.cfg.body_names.index("right_wrist_yaw_link")
        left_wrist_idx = self.cfg.body_names.index("left_wrist_yaw_link")
        
        # 获取手部速度 (只取第一个环境的数据用于显示)
        right_hand_vel = self.robot_body_lin_vel_w[0, right_wrist_idx]  # (3,)
        left_hand_vel = self.robot_body_lin_vel_w[0, left_wrist_idx]    # (3,)
        right_speed = torch.norm(right_hand_vel).item()  # 标量 m/s
        left_speed = torch.norm(left_hand_vel).item()    # 标量 m/s
        
        # 获取右手到目标的距离
        right_hand_pos = self.robot_body_pos_w[0, right_wrist_idx]  # (3,)
        dist_to_target = torch.norm(right_hand_pos - self.target_pos_w[0]).item()  # m
        
        # 将手部速度写入日志文件 (避免被 Isaac Lab 警告淹没)
        # 使用方法: 在另一个终端运行 tail -f logs/hand_speed.log 实时查看
        max_speed = max(right_speed, left_speed)
        if max_speed > 0.5:  # 速度超过 0.5 m/s 时才记录
            import pathlib
            log_dir = pathlib.Path(__file__).resolve().parents[6] / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "hand_speed.log"
            with open(log_file, "a") as f:
                f.write(f"R: {right_speed:.2f} m/s | L: {left_speed:.2f} m/s | Dist: {dist_to_target:.3f} m\n")

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])
        
        # =====================================================================
        # Stage 2: 绘制目标小球、引导大球线框、采样区域边界
        # =====================================================================
        
        # 1. 绘制目标小球 (红色实心球)
        self.target_sphere_visualizer.visualize(self.target_pos_w)
        
        # 2. 绘制引导大球线框和采样区域边界 (使用 debug lines)
        if HAS_DEBUG_DRAW and hasattr(self, "_debug_draw"):
            self._debug_draw.clear_lines()
            self._draw_guidance_spheres()
            self._draw_sampling_regions()
    
    def _draw_guidance_spheres(self):
        """
        绘制引导大球的线框 (浅绿色)
        
        为每个环境的目标小球绘制 3 个正交圆环 (XY, YZ, XZ 平面)
        来模拟一个线框球体
        """
        radius = self._guidance_sphere_radius
        num_segments = 32  # 每个圆环的线段数
        color = [0.5, 1.0, 0.5, 0.8]  # 浅绿色 RGBA
        thickness = 2.0
        
        source_positions = []
        target_positions = []
        colors = []
        thicknesses = []
        
        for env_idx in range(self.num_envs):
            center = self.target_pos_w[env_idx].cpu().numpy()
            
            # 绘制 XY 平面圆环 (水平)
            self._add_circle_lines(
                center, radius, "xy", num_segments,
                source_positions, target_positions, colors, thicknesses,
                color, thickness
            )
            
            # 绘制 YZ 平面圆环 (前后垂直)
            self._add_circle_lines(
                center, radius, "yz", num_segments,
                source_positions, target_positions, colors, thicknesses,
                color, thickness
            )
            
            # 绘制 XZ 平面圆环 (左右垂直)
            self._add_circle_lines(
                center, radius, "xz", num_segments,
                source_positions, target_positions, colors, thicknesses,
                color, thickness
            )
        
        if source_positions:
            self._debug_draw.draw_lines(source_positions, target_positions, colors, thicknesses)
    
    def _draw_sampling_regions(self):
        """
        绘制采样区域边界 (蓝色长方体线框)
        
        为每个环境绘制一个长方体的 12 条边
        
        注意: 始终绘制**最终采样范围** (课程等级 100% 时的范围)
        而不是当前课程等级的采样范围，因为:
        1. 初始课程等级为 0 时，采样范围非常小 (±1mm)，会被目标小球遮盖
        2. 用户需要看到完整的最终目标区域，了解训练目标
        """
        color = [0.3, 0.5, 1.0, 0.8]  # 蓝色 RGBA
        thickness = 2.0
        
        # 使用最终采样范围 (而不是当前课程等级的范围)
        x_range = self._final_sampling_range["x"]
        y_range = self._final_sampling_range["y"]
        z_range = self._final_sampling_range["z"]
        
        source_positions = []
        target_positions = []
        colors = []
        thicknesses = []
        
        for env_idx in range(self.num_envs):
            # 获取该环境的基准位置 (参考动作第一帧 Root 位置 + env origin)
            base_pos = self._reference_root_pos_w.cpu().numpy() + self._env.scene.env_origins[env_idx].cpu().numpy()
            base_quat = self._reference_root_quat_w.cpu().numpy()
            
            # 计算长方体的 8 个顶点 (在局部坐标系)
            corners_local = [
                [x_range[0], y_range[0], z_range[0]],  # 0: 左下后
                [x_range[1], y_range[0], z_range[0]],  # 1: 右下后
                [x_range[1], y_range[1], z_range[0]],  # 2: 右上后
                [x_range[0], y_range[1], z_range[0]],  # 3: 左上后
                [x_range[0], y_range[0], z_range[1]],  # 4: 左下前
                [x_range[1], y_range[0], z_range[1]],  # 5: 右下前
                [x_range[1], y_range[1], z_range[1]],  # 6: 右上前
                [x_range[0], y_range[1], z_range[1]],  # 7: 左上前
            ]
            
            # 将局部坐标转换到世界坐标
            corners_world = []
            for corner in corners_local:
                # 使用四元数旋转
                corner_tensor = torch.tensor(corner, dtype=torch.float32, device=self.device)
                rotated = quat_apply(
                    self._reference_root_quat_w.unsqueeze(0),
                    corner_tensor.unsqueeze(0)
                ).squeeze(0).cpu().numpy()
                world_corner = base_pos + rotated
                corners_world.append(world_corner.tolist())
            
            # 长方体的 12 条边
            edges = [
                # 底面 4 条边
                (0, 1), (1, 2), (2, 3), (3, 0),
                # 顶面 4 条边
                (4, 5), (5, 6), (6, 7), (7, 4),
                # 连接上下的 4 条边
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            
            for start_idx, end_idx in edges:
                source_positions.append(corners_world[start_idx])
                target_positions.append(corners_world[end_idx])
                colors.append(color)
                thicknesses.append(thickness)
        
        if source_positions:
            self._debug_draw.draw_lines(source_positions, target_positions, colors, thicknesses)
    
    def _add_circle_lines(
        self, center, radius, plane, num_segments,
        source_positions, target_positions, colors, thicknesses,
        color, thickness
    ):
        """
        在指定平面上添加圆环的线段
        
        Args:
            center: 圆心位置 (3,)
            radius: 圆的半径
            plane: "xy", "yz", 或 "xz"
            num_segments: 线段数量
        """
        for i in range(num_segments):
            angle1 = 2 * math.pi * i / num_segments
            angle2 = 2 * math.pi * (i + 1) / num_segments
            
            if plane == "xy":
                p1 = [center[0] + radius * math.cos(angle1),
                      center[1] + radius * math.sin(angle1),
                      center[2]]
                p2 = [center[0] + radius * math.cos(angle2),
                      center[1] + radius * math.sin(angle2),
                      center[2]]
            elif plane == "yz":
                p1 = [center[0],
                      center[1] + radius * math.cos(angle1),
                      center[2] + radius * math.sin(angle1)]
                p2 = [center[0],
                      center[1] + radius * math.cos(angle2),
                      center[2] + radius * math.sin(angle2)]
            elif plane == "xz":
                p1 = [center[0] + radius * math.cos(angle1),
                      center[1],
                      center[2] + radius * math.sin(angle1)]
                p2 = [center[0] + radius * math.cos(angle2),
                      center[1],
                      center[2] + radius * math.sin(angle2)]
            else:
                continue
            
            source_positions.append(p1)
            target_positions.append(p2)
            colors.append(color)
            thicknesses.append(thickness)


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # =========================================================================
    # Stage 2: 目标小球配置 (Task-Oriented RL)
    # =========================================================================
    
    # 最终采样范围 (Level 1.0 时的范围)
    # 相对于机器人 Root/Pelvis 的局部坐标系
    target_sampling_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: DEFAULT_TARGET_SAMPLING_RANGE.copy()
    )
    
    # =========================================================================
    # 初始采样范围 (Level 0.0 时的范围，课程学习起点)
    # 
    # 设计原则:
    # - 必须在 DEFAULT_TARGET_SAMPLING_RANGE 的范围内
    # - 设置为最终范围的中心点附近的极小区域
    # - 便于初期学习，降低探索难度
    # 
    # DEFAULT_TARGET_SAMPLING_RANGE 中心点:
    # - X: (0.6 + 0.65) / 2 = 0.625
    # - Y: (-0.4 + 0.4) / 2 = 0.0
    # - Z: (0.25 + 0.5) / 2 = 0.375
    # =========================================================================
    init_sampling_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (0.624, 0.626),   # 前方 62.4-62.6cm (中心点 ±1mm)
            "y": (-0.05, 0.05),   # 左右 ±5cm (中心点附近)
            "z": (0.374, 0.376),  # 高度 37.4-37.6cm (中心点 ±1mm)
        }
    )
    
    # 引导大球半径 (固定值，用于可视化奖励生效范围)
    guidance_sphere_radius: float = 0.25
    
    # =========================================================================
    # Stage 2: Hit 检测配置
    # =========================================================================
    
    # Hit 距离阈值 (米) - 对应目标小球半径
    # 简化设计: 只用距离判断，不用速度
    hit_distance_threshold: float = 0.12
    
    # Hit 后的延迟重采样时间 (秒)
    # 设计目的:
    # - Hit 后目标位置保持 1 秒不变
    # - 这 1 秒内任务奖励失效，只有 Mimic 奖励
    # - 鼓励机器人跟随参考动作收手
    # - 1 秒后重采样目标位置，任务奖励重新生效
    hit_resample_delay: float = 1.0
    
    # 攻击肢体名称
    effector_body_name: str = "right_wrist_yaw_link"
    
    # =========================================================================
    # Stage 2: 课程学习配置
    # 
    # 设计原则:
    # - 采样范围从中心点逐步扩展到全范围
    # - 设置 4 个等级 (0.0, 0.25, 0.5, 0.75, 1.0) 以简化课程
    # - 使用滑动窗口统计最近 500 step 的 Hit 率
    # - Hit 率 >= 75% 才能升级，确保当前难度已掌握
    # =========================================================================
    
    # 每次课程升级的步长 (0.0 ~ 1.0)
    # 设置为 0.25 意味着有 4 个等级: Level 0, 0.25, 0.5, 0.75, 1.0
    curriculum_level_step: float = 0.25
    
    # 课程升级所需的滑动窗口大小 (Step 数)
    # 使用最近 500 step 的 Hit 率来判断是否升级
    curriculum_window_size: int = 500
    
    # 课程升级所需的 Hit 成功率阈值 (0.0 ~ 1.0)
    # 
    # =========================================================================
    # 阈值计算分析 (考虑 Hit 后目标重采样):
    # =========================================================================
    # 
    # Step 时长 = decimation × dt = 4 × 0.005 = 0.02 秒
    # 
    # 一次完整攻击周期 (Hit 后目标移动到新位置):
    #   - 出拳攻击: ~0.3s
    #   - 收回手臂: ~0.3s  
    #   - 瞄准新位置: ~0.2s
    #   - 总计: ~0.8-1.0s = 40-50 steps
    # 
    # 理论最大 Hit 率 = 1 hit / 50 steps = 2%
    # 
    # 设置为 0.005 (0.5%) 表示:
    #   - 达到理论上限的 25%
    #   - 500 steps × 128 envs = 64000 次机会
    #   - 需要 320 次 Hit (每 env 平均 2.5 次/500steps)
    #   - 每 env 约 4s 完成一次有效攻击，合理
    # 
    # 注意: 由于 Hit 后目标重采样，阈值比旧版 (0.02) 大幅降低
    # =========================================================================
    curriculum_hit_rate_threshold: float = 0.005
    
    # 目标小球 Marker 配置 (红色实心球)
    target_sphere_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/target_sphere",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.06,  # 半径 6cm，与 hit_distance_threshold 一致
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.2, 0.2),  # 红色
                    opacity=0.9,
                ),
            ),
        },
    )
