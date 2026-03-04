from __future__ import annotations

import math
import numpy as np
import os
import pathlib
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
# [已注释] 目标小球采样区域配置 (课程学习使用，已取消)
# 改为固定目标点，在 MotionCommandCfg.fixed_target_local_pos 中配置
# =============================================================================
# DEFAULT_TARGET_SAMPLING_RANGE = {
#     "x": (0.6, 0.65),   # 前方 60-65cm，有效攻击距离
#     "y": (-0.4, 0.4),    # 左右 ±40cm，覆盖左右出拳
#     "z": (0.25, 0.5),    # 高度 25-50cm（胸部到下巴高度）
# }


def _normalize_motion_name(motion_file: str) -> str:
    """
    Resolve skill name from a motion npz path.

    Examples:
      artifacts/trim_cross_right_normal_body_2_150:v2/motion.npz -> cross_right_normal_body_2_150
      iros_motion/npz/trim_hook_left_normal_body2_150.npz -> hook_left_normal_body2_150
    """
    motion_path = pathlib.Path(motion_file)
    if motion_path.name == "motion.npz":
        raw_name = motion_path.parent.name
    else:
        raw_name = motion_path.stem
    raw_name = raw_name.split(":", 1)[0]
    if raw_name.startswith("trim_"):
        raw_name = raw_name[len("trim_") :]
    return raw_name


def _resolve_effector_group(
    motion_name: str,
    left_hand_skills: Sequence[str],
    right_hand_skills: Sequence[str],
    left_foot_skills: Sequence[str],
    right_foot_skills: Sequence[str],
) -> str:
    """Map motion name to effector group: left_hand/right_hand/left_foot/right_foot."""
    if motion_name in left_hand_skills:
        return "left_hand"
    if motion_name in right_hand_skills:
        return "right_hand"
    if motion_name in left_foot_skills:
        return "left_foot"
    if motion_name in right_foot_skills:
        return "right_foot"
    return "right_hand"


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
        # - effector_to_target_dist: 当前攻击肢体到目标距离 (核心监控指标)
        # - effector_speed: 攻击肢体速度 (监控出拳速度)
        # =====================================================================
        self.metrics["resample_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["effector_to_target_dist"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["effector_speed"] = torch.zeros(self.num_envs, device=self.device)

        # Target position in world frame.
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Episode state for task phase split.
        self.cumulative_hit_count = torch.zeros(self.num_envs, device=self.device)
        self.has_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_rewards_enabled = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Camera-visibility simulation:
        # Agent sees target only for a short random window at episode start.
        self.target_visible_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_visible_time_left = torch.zeros(self.num_envs, device=self.device)
        self.hidden_target_obs_local = torch.tensor(self.cfg.hidden_target_obs_local, device=self.device, dtype=torch.float32)

        # Reference root from the first motion frame (used by fixed target sampling).
        self._reference_root_pos_w = self.motion.body_pos_w[0, 0].clone()
        self._reference_root_quat_w = self.motion.body_quat_w[0, 0].clone()
        self._guidance_sphere_radius = self.cfg.guidance_sphere_radius
        self.hit_distance_threshold = self.cfg.hit_distance_threshold
        self.step_dt = self._env.step_dt
        self.current_time = torch.zeros(self.num_envs, device=self.device)

        # Unified effector setup: infer skill from motion name, then map to active effector.
        self.motion_name = _normalize_motion_name(self.cfg.motion_file)
        self.effector_group = _resolve_effector_group(
            self.motion_name,
            self.cfg.left_hand_skill_names,
            self.cfg.right_hand_skill_names,
            self.cfg.left_foot_skill_names,
            self.cfg.right_foot_skill_names,
        )
        effector_name_map = {
            "left_hand": "left_wrist_yaw_link",
            "right_hand": "right_wrist_yaw_link",
            "left_foot": "left_ankle_roll_link",
            "right_foot": "right_ankle_roll_link",
        }
        one_hot_map = {
            "left_hand": [1.0, 0.0, 0.0, 0.0],
            "right_hand": [0.0, 1.0, 0.0, 0.0],
            "left_foot": [0.0, 0.0, 1.0, 0.0],
            "right_foot": [0.0, 0.0, 0.0, 1.0],
        }
        self.effector_body_name = effector_name_map[self.effector_group]
        self.effector_index = self.cfg.body_names.index(self.effector_body_name)
        self.active_effector_one_hot = torch.tensor(
            one_hot_map[self.effector_group], dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)

        # Hit can only be triggered by hands/feet end effectors.
        self.hit_effector_indices = torch.tensor(
            [self.cfg.body_names.index(name) for name in self.cfg.hit_effector_body_names],
            dtype=torch.long,
            device=self.device,
        )

        # Post-hit retract reward uses distance to episode start anchor.
        self.episode_start_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Pre-hit near reward state.
        self.min_distance_to_target = torch.full((self.num_envs,), 10.0, device=self.device)
        self.has_entered_guidance_sphere = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_distance_from_target = torch.zeros(self.num_envs, device=self.device)

        # =====================================================================
        # [已注释] 课程学习状态 (已取消，改为固定目标点训练)
        # =====================================================================
        # self.curriculum_level = torch.tensor(0.0, device=self.device)
        # self.curriculum_hit_rate_threshold = self.cfg.curriculum_hit_rate_threshold
        # self.curriculum_level_step = self.cfg.curriculum_level_step
        # self.curriculum_window_size = self.cfg.curriculum_window_size
        # self._hit_history = torch.zeros(self.curriculum_window_size, dtype=torch.long, device=self.device)
        # self._hit_history_idx = 0
        # self._hit_history_filled = False
        # self._init_sampling_range = self.cfg.init_sampling_range
        # self._final_sampling_range = self.cfg.target_sampling_range

        # Hit 相关的 metrics (保留用于监控)
        self.metrics["hit_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_hit_count"] = torch.zeros(self.num_envs, device=self.device)
        # [已注释] 课程学习 metrics
        # self.metrics["curriculum_level"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["curriculum_hit_rate"] = torch.zeros(self.num_envs, device=self.device)
        
        # 初始化目标位置（会在 reset 时被重新采样）
        self._init_target_positions()

    def _init_target_positions(self):
        """初始化所有环境的目标位置"""
        # 在采样范围内随机采样目标位置
        self._resample_target_positions(torch.arange(self.num_envs, device=self.device))
    
    # [已注释] _get_current_sampling_range: 课程学习采样范围计算，已取消
    # def _get_current_sampling_range(self) -> dict:
    #     level = self.curriculum_level.item()
    #     current_range = {}
    #     for axis in ["x", "y", "z"]:
    #         init_min, init_max = self._init_sampling_range[axis]
    #         final_min, final_max = self._final_sampling_range[axis]
    #         current_min = init_min + level * (final_min - init_min)
    #         current_max = init_max + level * (final_max - init_max)
    #         current_range[axis] = (current_min, current_max)
    #     return current_range
    
    def _resample_target_positions(self, env_ids: torch.Tensor):
        """
        Sample target around a fixed local center with per-axis randomization.

        This updates the *real* target in world coordinates. The observation-side visibility
        is handled separately by target_visible_mask/target_visible_time_left.
        """
        if len(env_ids) == 0:
            return

        num_samples = len(env_ids)
        fx, fy, fz = self.cfg.fixed_target_local_pos
        local_target_pos = torch.tensor([[fx, fy, fz]], dtype=torch.float32, device=self.device).repeat(num_samples, 1)

        ranges = self.cfg.target_randomization_local_range
        rx = sample_uniform(ranges["x"][0], ranges["x"][1], (num_samples,), device=self.device)
        ry = sample_uniform(ranges["y"][0], ranges["y"][1], (num_samples,), device=self.device)
        rz = sample_uniform(ranges["z"][0], ranges["z"][1], (num_samples,), device=self.device)
        local_target_pos[:, 0] += rx
        local_target_pos[:, 1] += ry
        local_target_pos[:, 2] += rz

        world_target_pos = self._reference_root_pos_w + quat_apply(
            self._reference_root_quat_w.unsqueeze(0).repeat(num_samples, 1),
            local_target_pos,
        )
        world_target_pos = world_target_pos + self._env.scene.env_origins[env_ids]
        self.target_pos_w[env_ids] = world_target_pos

        # Reset episode-level task state.
        self.task_rewards_enabled[env_ids] = True
        self.has_hit[env_ids] = False
        self.min_distance_to_target[env_ids] = 10.0
        self.has_entered_guidance_sphere[env_ids] = False
        self.max_distance_from_target[env_ids] = 0.0

        # Sample the short "target visible" window (camera sees target briefly).
        t_min, t_max = self.cfg.target_visible_time_range_s
        self.target_visible_time_left[env_ids] = sample_uniform(t_min, t_max, (num_samples,), device=self.device)
        self.target_visible_mask[env_ids] = True

    def _delayed_spawn_target(self, env_ids: torch.Tensor):
        """
        Backward-compatible wrapper.

        Old stage-2 code used delayed spawn; current design spawns target immediately
        and controls only *observation visibility* using a short time window.
        """
        self._resample_target_positions(env_ids)

    def check_hit(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        检查是否发生有效 Hit

        规则:
        - 仅四个攻击末端 (双手/双脚) 可触发 hit
        - 距离阈值: hit_distance_threshold
        - 每个 episode 只记录首次 hit

        Hit 后:
        - 目标物理位置不变
        - 任务奖励开关关闭 (near/hit 停止)
        - 目标观测立刻隐藏，模拟相机看不到目标
        
        Returns:
            hit_mask: (num_envs,) bool, 哪些环境发生了有效 Hit
            distances: (num_envs,) float, 最近攻击末端到 target 的距离
        """
        # Hit can only be triggered by hand/foot end effectors.
        effector_positions = self.robot_body_pos_w[:, self.hit_effector_indices]  # (num_envs, K, 3)
        target_positions = self.target_pos_w[:, None, :]  # (num_envs, 1, 3)
        distances_all = torch.norm(effector_positions - target_positions, dim=-1)  # (num_envs, K)
        min_distances, _ = torch.min(distances_all, dim=-1)  # (num_envs,)

        # Hit only once per episode.
        dist_ok = min_distances < self.hit_distance_threshold
        hit_mask = dist_ok & self.task_rewards_enabled & (~self.has_hit)

        hit_env_ids = torch.where(hit_mask)[0]
        if len(hit_env_ids) > 0:
            self.has_hit[hit_env_ids] = True
            self.task_rewards_enabled[hit_env_ids] = False
            self.cumulative_hit_count[hit_env_ids] += 1.0
            self.max_distance_from_target[hit_env_ids] = min_distances[hit_env_ids]

            # After hit: target still exists physically, but observation is hidden to the agent.
            self.target_visible_mask[hit_env_ids] = False
            self.target_visible_time_left[hit_env_ids] = 0.0

        return hit_mask, min_distances
    
    def update_hit_resample_timer(self):
        """
        Update short target-visibility window (camera observation simulation).

        Real target stays fixed for the whole episode. Only the observation switches to
        hidden marker after visibility window expires or after hit.
        """
        visible_and_not_hit = self.target_visible_mask & (~self.has_hit)
        if torch.any(visible_and_not_hit):
            self.target_visible_time_left[visible_and_not_hit] -= self.step_dt
            expired = visible_and_not_hit & (self.target_visible_time_left <= 0.0)
            self.target_visible_mask[expired] = False

    def update_curriculum(self, hit_mask: torch.Tensor):
        """
        [已简化] 课程学习逻辑已取消，仅更新 hit 监控指标

        原课程学习机制 (已注释):
        - 滑动窗口统计 Hit 率
        - Hit 率达阈值后自动扩大采样范围

        Args:
            hit_mask: (num_envs,) bool, 本次 step 哪些环境发生了 Hit
        """
        # 更新 hit metrics (用于 WandB 监控)
        self.metrics["hit_count"][:] = hit_mask.float()
        self.metrics["max_hit_count"][:] = self.cumulative_hit_count.max().float()

        # =====================================================================
        # [已注释] 课程学习: 滑动窗口统计 + 升级逻辑
        # =====================================================================
        # hit_count_this_step = hit_mask.sum().long()
        # self._hit_history[self._hit_history_idx] = hit_count_this_step
        # self._hit_history_idx = (self._hit_history_idx + 1) % self.curriculum_window_size
        # if self._hit_history_idx == 0:
        #     self._hit_history_filled = True
        # if self._hit_history_filled:
        #     total_hits = self._hit_history.sum().item()
        #     total_opportunities = self.curriculum_window_size * self.num_envs
        # else:
        #     total_hits = self._hit_history[:self._hit_history_idx].sum().item()
        #     total_opportunities = self._hit_history_idx * self.num_envs
        # current_hit_rate = total_hits / total_opportunities if total_opportunities > 0 else 0.0
        # self.metrics["curriculum_level"][:] = self.curriculum_level.item()
        # self.metrics["curriculum_hit_rate"][:] = current_hit_rate
        # if (self._hit_history_filled and
        #         current_hit_rate >= self.curriculum_hit_rate_threshold and
        #         self.curriculum_level < 1.0):
        #     old_level = self.curriculum_level.item()
        #     self.curriculum_level = torch.clamp(
        #         self.curriculum_level + self.curriculum_level_step, max=1.0
        #     )
        #     self._hit_history.zero_()
        #     self._hit_history_idx = 0
        #     self._hit_history_filled = False
        #     print(f">>> CURRICULUM LEVEL UP! {old_level:.2f} -> {self.curriculum_level.item():.2f}")

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

    @property
    def target_is_visible(self) -> torch.Tensor:
        return self.target_visible_mask & (~self.has_hit)

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
        
        # Stage-2 monitoring metrics (active effector driven).
        effector_lin_vel_w = self.robot_body_lin_vel_w[:, self.effector_index]
        effector_pos_w = self.robot_body_pos_w[:, self.effector_index]
        self.metrics["effector_speed"] = torch.norm(effector_lin_vel_w, dim=-1)
        self.metrics["effector_to_target_dist"] = torch.norm(effector_pos_w - self.target_pos_w, dim=-1)

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
        self.metrics["resample_count"][env_ids] += 1.0

    def _resample_command(self, env_ids: Sequence[int]):
        """Episode 重置时调用，重新采样动作和目标位置"""
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._resample_target_positions(env_ids_tensor)

        self.cumulative_hit_count[env_ids_tensor] = 0.0
        self.current_time[env_ids_tensor] = 0.0

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
        self.episode_start_anchor_pos_w[env_ids_tensor] = root_pos[env_ids_tensor]

    def _update_command(self):
        self.time_steps += 1
        
        # Stage 2: 更新当前仿真时间 (每个 env 独立)
        # dt = decimation * sim.dt
        dt = self._env.cfg.decimation * self._env.cfg.sim.dt
        self.current_time += dt
        self.update_hit_resample_timer()

        # Keep last reference frame until environment reset.
        self.time_steps = torch.clamp(self.time_steps, max=self.motion.time_step_total - 1)

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
        
        self.metrics["effector_to_target_dist"][:] = distances
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
        
        # 2. 绘制引导大球线框 (使用 debug lines)
        if HAS_DEBUG_DRAW and hasattr(self, "_debug_draw"):
            self._debug_draw.clear_lines()
            self._draw_guidance_spheres()
            # [已注释] 采样区域边界可视化 (课程学习已取消)
            # self._draw_sampling_regions()
    
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
    # 固定目标点配置 (已取消课程学习，改为固定目标训练)
    # 坐标系: 相对于参考动作第一帧 Root (Pelvis) 的局部坐标系
    #   x > 0 : 机器人前方 (m)
    #   y     : 左右偏移 (m, 正=左, 负=右)
    #   z > 0 : 骨盆以上高度 (m)
    # =========================================================================
    # 默认: 前方 62.5cm, 居中, 骨盆以上 20 cm
    # 可根据参考动作中拳头实际到达的高度调整 z 值(z=0.2正好是cross参考动作中拳头高度）
    fixed_target_local_pos: tuple = (0.625, 0.0, 0.20)

    # [已注释] 课程学习采样范围配置
    # target_sampling_range: dict[str, tuple[float, float]] = field(
    #     default_factory=lambda: {
    #         "x": (0.6, 0.65), "y": (-0.4, 0.4), "z": (0.25, 0.5),
    #     }
    # )
    # init_sampling_range: dict[str, tuple[float, float]] = field(
    #     default_factory=lambda: {
    #         "x": (0.624, 0.626), "y": (-0.05, 0.05), "z": (0.374, 0.376),
    #     }
    # )

    # 引导大球半径 (固定值，用于可视化奖励生效范围)
    guidance_sphere_radius: float = 0.4

    # Target randomization around fixed local center for sim2real robustness.
    target_randomization_local_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.2, 0.2),
        }
    )

    # Agent can see target only for a short random window at episode start.
    target_visible_time_range_s: tuple[float, float] = (0.3, 0.8)
    hidden_target_obs_local: tuple[float, float, float] = (0.0, 0.0, -10.0)

    # Hit is valid only for these end-effectors.
    hit_effector_body_names: tuple[str, ...] = (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )

    # Active-effector mapping by motion name (single source of truth).
    left_hand_skill_names: tuple[str, ...] = ("hook_left_normal_body2_150",)
    right_hand_skill_names: tuple[str, ...] = (
        "cross_right_normal_body_2_150",
        "swing_right_normal_head2_150",
    )
    left_foot_skill_names: tuple[str, ...] = ("roundhouse_left_normal_mid_no_bag_1_150",)
    right_foot_skill_names: tuple[str, ...] = (
        "frontkick_right_fast_body_no_bag_2_150",
        "roundhouse_right_normal_mid_no_bag_2_150",
    )

    # Hit distance threshold (target sphere radius).
    hit_distance_threshold: float = 0.06

    # [已注释] 课程学习配置
    # curriculum_level_step: float = 0.25
    # curriculum_window_size: int = 500
    # curriculum_hit_rate_threshold: float = 0.005
    
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
