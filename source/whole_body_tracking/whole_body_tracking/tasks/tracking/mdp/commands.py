from __future__ import annotations

import math
import numpy as np
import os
import pathlib
import torch
import xml.etree.ElementTree as ET
from collections import deque
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
    quat_from_angle_axis,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
import isaaclab.sim as sim_utils
from whole_body_tracking.learning.expert_command_telemetry import clear_expert_commands, set_expert_commands
from whole_body_tracking.learning.router_telemetry import get_router_weights
from whole_body_tracking.tasks.tracking.stage4.amp_discriminator import AmpDiscriminator

# Debug draw 用于绘制线框
try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    HAS_DEBUG_DRAW = True
except ImportError:
    HAS_DEBUG_DRAW = False

try:
    from pxr import Gf, UsdGeom
    # Isaac Sim / USD 版本差异：部分版本没有 UsdGeom.Text。
    HAS_PXR_TEXT = hasattr(UsdGeom, "Text")
except ImportError:
    HAS_PXR_TEXT = False

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


# ---------------------------------------------------------------------------
# G1 + MimicKit AMP disc_obs compatibility constants
# ---------------------------------------------------------------------------
# Correct joint rotation axes for G1 (from MJCF), keyed by joint name.
# Used as fallback when no URDF is available (e.g. USD-based robot assets).
_G1_JOINT_AXES: dict[str, tuple[float, float, float]] = {
    "left_hip_pitch_joint":       (0.0, 1.0, 0.0),
    "left_hip_roll_joint":        (1.0, 0.0, 0.0),
    "left_hip_yaw_joint":         (0.0, 0.0, 1.0),
    "left_knee_joint":            (0.0, 1.0, 0.0),
    "left_ankle_pitch_joint":     (0.0, 1.0, 0.0),
    "left_ankle_roll_joint":      (1.0, 0.0, 0.0),
    "right_hip_pitch_joint":      (0.0, 1.0, 0.0),
    "right_hip_roll_joint":       (1.0, 0.0, 0.0),
    "right_hip_yaw_joint":        (0.0, 0.0, 1.0),
    "right_knee_joint":           (0.0, 1.0, 0.0),
    "right_ankle_pitch_joint":    (0.0, 1.0, 0.0),
    "right_ankle_roll_joint":     (1.0, 0.0, 0.0),
    "waist_yaw_joint":            (0.0, 0.0, 1.0),
    "waist_roll_joint":           (1.0, 0.0, 0.0),
    "waist_pitch_joint":          (0.0, 1.0, 0.0),
    "left_shoulder_pitch_joint":  (0.0, 1.0, 0.0),
    "left_shoulder_roll_joint":   (1.0, 0.0, 0.0),
    "left_shoulder_yaw_joint":    (0.0, 0.0, 1.0),
    "left_elbow_joint":           (0.0, 1.0, 0.0),
    "left_wrist_roll_joint":      (1.0, 0.0, 0.0),
    "left_wrist_pitch_joint":     (0.0, 1.0, 0.0),
    "left_wrist_yaw_joint":       (0.0, 0.0, 1.0),
    "right_shoulder_pitch_joint": (0.0, 1.0, 0.0),
    "right_shoulder_roll_joint":  (1.0, 0.0, 0.0),
    "right_shoulder_yaw_joint":   (0.0, 0.0, 1.0),
    "right_elbow_joint":          (0.0, 1.0, 0.0),
    "right_wrist_roll_joint":     (1.0, 0.0, 0.0),
    "right_wrist_pitch_joint":    (0.0, 1.0, 0.0),
    "right_wrist_yaw_joint":      (0.0, 0.0, 1.0),
}
# MimicKit G1 MJCF has a FIXED (rigid) 'head_link' body at body traversal
# position 16 — between torso_link (waist_pitch_joint) and
# left_shoulder_pitch_link.  It contributes an identity quaternion as the
# 16th entry (0-indexed: position 15) in dof_to_rot output.
# Its local position relative to torso_link parent is (0.015, 0, 0.43) m.
_G1_MIMICKIT_HEAD_JOINT_INSERT_POS: int = 15   # insert identity here (0-indexed)
_G1_MIMICKIT_HEAD_OFFSET: tuple[float, float, float] = (0.015, 0.0, 0.43)  # torso-local


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

    @classmethod
    def from_robot_static(cls, robot: Articulation, body_indexes: Sequence[int], device: str = "cpu") -> "MotionLoader":
        """Create a single-frame static command source from robot defaults.

        This is used by Stage4 when we do not want to depend on a motion npz file.
        """
        obj = cls.__new__(cls)
        obj.fps = 1.0

        # body_indexes may be a python list or a tensor from caller.
        body_indexes_t = torch.as_tensor(body_indexes, dtype=torch.long, device=device)

        default_joint_pos = robot.data.default_joint_pos[0].to(device=device, dtype=torch.float32)
        default_joint_vel = robot.data.default_joint_vel[0].to(device=device, dtype=torch.float32)

        # Use current initialized body states as static reference frame.
        body_pos = robot.data.body_pos_w[0, body_indexes_t].to(device=device, dtype=torch.float32)
        body_quat = robot.data.body_quat_w[0, body_indexes_t].to(device=device, dtype=torch.float32)
        body_lin_vel = robot.data.body_lin_vel_w[0, body_indexes_t].to(device=device, dtype=torch.float32)
        body_ang_vel = robot.data.body_ang_vel_w[0, body_indexes_t].to(device=device, dtype=torch.float32)

        obj.joint_pos = default_joint_pos.unsqueeze(0)
        obj.joint_vel = default_joint_vel.unsqueeze(0)
        obj._body_pos_w = body_pos.unsqueeze(0)
        obj._body_quat_w = body_quat.unsqueeze(0)
        obj._body_lin_vel_w = body_lin_vel.unsqueeze(0)
        obj._body_ang_vel_w = body_ang_vel.unsqueeze(0)
        # Static source has already been sliced by body_indexes_t above.
        # Keep identity indexing here to avoid second indexing out-of-bounds.
        obj._body_indexes = torch.arange(obj._body_pos_w.shape[1], dtype=torch.long, device=device)
        obj.time_step_total = 1
        return obj

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
        # NOTE:
        # CommandTerm.__init__ may call set_debug_vis(...) immediately, which
        # triggers _set_debug_vis_impl before this constructor body continues.
        # Pre-initialize countdown fields to avoid attribute access races.
        self.show_target_visibility_countdown = bool(cfg.show_target_visibility_countdown)
        self.countdown_env_index = int(cfg.countdown_env_index)
        self.countdown_height_offset = float(cfg.countdown_height_offset)

        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        found_body_indices = list(self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0])
        if len(found_body_indices) != len(self.cfg.body_names):
            raise ValueError(
                "find_bodies 返回数量与 cfg.body_names 不一致: "
                f"{len(found_body_indices)} vs {len(self.cfg.body_names)}"
            )
        num_robot_bodies = len(self.robot.body_names)
        invalid_id_pairs = [
            (i, idx) for i, idx in enumerate(found_body_indices) if (idx < 0) or (idx >= num_robot_bodies)
        ]
        if invalid_id_pairs:
            invalid_desc = ", ".join(
                f"{self.cfg.body_names[i]}->{idx}" for i, idx in invalid_id_pairs
            )
            raise ValueError(
                "cfg.body_names 中存在无效 body 索引（与机器人 USD 不匹配）: "
                + invalid_desc
            )
        self.body_indexes = torch.tensor(found_body_indices, dtype=torch.long, device=self.device)

        # Stage4-MoE runtime no longer uses a shared reference motion source.
        # The base command is static robot-default, while each frozen expert gets
        # its own skill-specific command from `stage4_expert_motion_files`.
        self.motion = MotionLoader.from_robot_static(self.robot, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._expert_motions: list[MotionLoader] = []
        if len(self.cfg.stage4_expert_motion_files) > 0:
            self._expert_motions = [
                MotionLoader(path, self.body_indexes, device=self.device) for path in self.cfg.stage4_expert_motion_files
            ]
        if len(self._expert_motions) == 0:
            clear_expert_commands()
        else:
            self._update_expert_command_telemetry()
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

        self.enable_tracking_error_metrics = bool(self.cfg.enable_tracking_error_metrics)
        if self.enable_tracking_error_metrics:
            self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        
        # Stage4 通用任务状态 metrics。
        self.metrics["has_hit"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_visible"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["task_rewards_enabled"] = torch.zeros(self.num_envs, device=self.device)
        self.enable_router_weight_metrics = bool(self.cfg.enable_router_weight_metrics)
        self._router_metric_names: list[str] = []
        self._router_metric_keys: list[str] = []
        self._router_weight_episode_sum: torch.Tensor | None = None
        self._router_weight_episode_steps = torch.zeros(self.num_envs, device=self.device)
        self._router_entropy_eps = 1.0e-8
        if self.enable_router_weight_metrics and len(self.cfg.router_metric_names) > 0:
            self._initialize_router_metrics(
                num_skills=len(self.cfg.router_metric_names),
                metric_names=list(self.cfg.router_metric_names),
            )

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
        self.hit_distance_threshold = float(self.cfg.hit_distance_threshold)
        self.hit_radius_curriculum_enabled = bool(self.cfg.hit_radius_curriculum_enabled)
        self.current_hit_distance_threshold = float(self.hit_distance_threshold)
        self.hit_radius_end = float(self.cfg.hit_radius_end)
        self.hit_radius_shrink_factor = float(self.cfg.hit_radius_shrink_factor)
        self.hit_curriculum_success_threshold = float(self.cfg.hit_curriculum_success_threshold)
        self.hit_curriculum_window = max(int(self.cfg.hit_curriculum_window), 1)
        self._hit_outcome_window: deque[float] = deque(maxlen=self.hit_curriculum_window)
        self._hit_success_rate_window = 0.0
        self.target_sphere_follow_hit_radius = bool(self.cfg.target_sphere_follow_hit_radius)
        self.guidance_sphere_follow_hit_radius = bool(self.cfg.guidance_sphere_follow_hit_radius)
        self._target_marker_base_radius = self._resolve_target_marker_base_radius()
        if self.hit_radius_curriculum_enabled:
            self.current_hit_distance_threshold = float(self.cfg.hit_radius_start)
            self.hit_distance_threshold = self.current_hit_distance_threshold
        self.step_dt = self._env.step_dt
        self.current_time = torch.zeros(self.num_envs, device=self.device)

        # AMP style reward runtime (optional).
        self.amp_reward_enabled = bool(self.cfg.amp_reward_enabled)
        self.amp_obs_history_steps = max(int(self.cfg.amp_obs_history_steps), 1)
        self.amp_allow_obs_dim_mismatch = bool(self.cfg.amp_allow_obs_dim_mismatch)
        self.amp_disc_obs_mode = str(self.cfg.amp_disc_obs_mode).lower()
        if self.amp_disc_obs_mode not in {"legacy_simple", "mimickit_like"}:
            raise ValueError(
                f"未知 amp_disc_obs_mode='{self.cfg.amp_disc_obs_mode}'，"
                "仅支持 {'legacy_simple', 'mimickit_like'}。"
            )
        self.amp_disc_global_obs = bool(self.cfg.amp_disc_global_obs)
        self.amp_disc_root_height_obs = bool(self.cfg.amp_disc_root_height_obs)
        self.amp_discriminator: AmpDiscriminator | None = None
        self._amp_obs_history: torch.Tensor | None = None
        self._amp_warned_dim_mismatch = False
        self._amp_warned_history_init_error = False
        self._amp_raw_slices: dict[str, tuple[int, int]] = {}
        self._amp_num_joints = int(self.robot_joint_pos.shape[-1])
        self._amp_key_body_indices = torch.zeros(0, dtype=torch.long, device=self.device)
        self._amp_joint_axes = torch.zeros(0, 3, dtype=torch.float32, device=self.device)
        self._amp_mimickit_g1_compat: bool = bool(getattr(self.cfg, "amp_disc_mimickit_g1_compat", True))
        self._amp_head_link_insert_pos: int = -1
        if self.amp_disc_obs_mode == "mimickit_like":
            self._amp_key_body_indices = self._resolve_amp_key_body_indices()
            self._amp_joint_axes = self._resolve_amp_joint_axes()
            if self._amp_mimickit_g1_compat:
                self._amp_head_link_insert_pos = self._compute_head_link_insert_pos()
        if self.amp_reward_enabled:
            if len(self.cfg.amp_disc_bundle_path) == 0:
                raise ValueError("amp_reward_enabled=True 但未提供 amp_disc_bundle_path。")
            self.amp_discriminator = AmpDiscriminator.from_bundle(
                self.cfg.amp_disc_bundle_path,
                activation=self.cfg.amp_disc_activation,
            ).to(self.device)
            # AMP diagnostics (for debugging "amp_style_reward stuck at 0").
            self.metrics["amp_obs_valid"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["amp_disc_logit"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["amp_disc_prob"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["amp_reward_raw"] = torch.zeros(self.num_envs, device=self.device)

        # Unified effector setup: infer skill from motion name, then map to active effector.
        self.motion_name = _normalize_motion_name(self.cfg.motion_file) if len(self.cfg.motion_file) > 0 else "stage4_moe"
        self.effector_group = _resolve_effector_group(
            self.motion_name,
            self.cfg.left_hand_skill_names,
            self.cfg.right_hand_skill_names,
            self.cfg.left_foot_skill_names,
            self.cfg.right_foot_skill_names,
        )

        # Keep compatibility with single-skill expert pretraining:
        # use a fixed active-effector one-hot for all modes/skills.
        self.active_effector_one_hot = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)

        # Hit can only be triggered by hands/feet end effectors.
        self.hit_effector_indices = torch.tensor(
            [self.cfg.body_names.index(name) for name in self.cfg.hit_effector_body_names],
            dtype=torch.long,
            device=self.device,
        )

        # Post-hit retract reward uses distance to episode start anchor.
        self.episode_start_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Hit / curriculum metrics.
        self.metrics["hit_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_hit_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hit_radius"] = torch.full(
            (self.num_envs,), float(self.current_hit_distance_threshold), device=self.device
        )
        self.metrics["hit_success_rate_window"] = torch.zeros(self.num_envs, device=self.device)
        
        # 初始化目标位置（会在 reset 时被重新采样）
        self._init_target_positions()
        # Lazy AMP history initialization:
        # initialize at reset/update when runtime robot buffers are stable.

    def _init_target_positions(self):
        """初始化所有环境的目标位置"""
        # 在采样范围内随机采样目标位置
        self._resample_target_positions(torch.arange(self.num_envs, device=self.device))

    def _resolve_target_marker_base_radius(self) -> float:
        """Read marker prototype radius, used as scale baseline for dynamic hit-radius visualization."""
        sphere_cfg = self.cfg.target_sphere_cfg.markers.get("sphere", None)
        radius = float(getattr(sphere_cfg, "radius", 0.06))
        return max(radius, 1.0e-4)

    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name).strip("_")

    def _initialize_router_metrics(self, num_skills: int, metric_names: list[str] | None = None):
        if num_skills <= 0:
            return
        if self._router_weight_episode_sum is not None and self._router_weight_episode_sum.shape[1] == num_skills:
            return

        if metric_names is None or len(metric_names) != num_skills:
            metric_names = [f"expert_{idx}" for idx in range(num_skills)]

        self._router_metric_names = [self._sanitize_metric_name(name) for name in metric_names]
        self._router_metric_keys = []
        for idx, metric_name in enumerate(self._router_metric_names):
            key = f"router_weight_{metric_name}"
            if key in self.metrics:
                key = f"router_weight_expert_{idx}"
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)
            self._router_metric_keys.append(key)

        self.metrics["router_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self._router_weight_episode_sum = torch.zeros(self.num_envs, num_skills, device=self.device)
        self._router_weight_episode_steps.zero_()

    def _update_router_weight_metrics(self):
        if not self.enable_router_weight_metrics:
            return

        weights = get_router_weights()
        if weights is None:
            return
        if weights.ndim != 2 or weights.shape[0] != self.num_envs:
            return
        # Telemetry comes from policy side and may live on another CUDA device.
        # Align to env device to avoid cross-device ops in metrics accumulation.
        if weights.device != self.device:
            weights = weights.to(device=self.device)

        if self._router_weight_episode_sum is None or self._router_weight_episode_sum.shape[1] != weights.shape[1]:
            configured_names = list(self.cfg.router_metric_names)
            names = configured_names if len(configured_names) == weights.shape[1] else None
            self._initialize_router_metrics(weights.shape[1], names)
        if self._router_weight_episode_sum is None:
            return

        self._router_weight_episode_sum += weights
        self._router_weight_episode_steps += 1.0
        denom = self._router_weight_episode_steps.unsqueeze(-1).clamp_min(1.0)
        running_avg = self._router_weight_episode_sum / denom

        for idx, key in enumerate(self._router_metric_keys):
            self.metrics[key][:] = running_avg[:, idx]

        entropy = -torch.sum(weights * torch.log(weights + self._router_entropy_eps), dim=-1)
        entropy_norm = max(math.log(float(weights.shape[1]) + self._router_entropy_eps), self._router_entropy_eps)
        self.metrics["router_entropy"][:] = entropy / entropy_norm

    def _update_expert_command_telemetry(self):
        """Update per-skill command bank for MoE frozen experts.

        Output shape: [num_envs, num_skills, command_dim]
        where command_dim = joint_pos_dim + joint_vel_dim.
        """
        if len(self._expert_motions) == 0:
            clear_expert_commands()
            return

        per_skill_commands: list[torch.Tensor] = []
        for motion in self._expert_motions:
            step_idx = torch.clamp(self.time_steps, max=motion.time_step_total - 1)
            command = torch.cat([motion.joint_pos[step_idx], motion.joint_vel[step_idx]], dim=-1)
            per_skill_commands.append(command)

        expert_command_bank = torch.stack(per_skill_commands, dim=1)
        set_expert_commands(expert_command_bank)
    
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
        - 距离阈值: current_hit_distance_threshold (可由课程学习动态收缩)
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
        dist_ok = min_distances < self.current_hit_distance_threshold
        hit_mask = dist_ok & self.task_rewards_enabled & (~self.has_hit)

        hit_env_ids = torch.where(hit_mask)[0]
        if len(hit_env_ids) > 0:
            self.has_hit[hit_env_ids] = True
            self.task_rewards_enabled[hit_env_ids] = False
            self.cumulative_hit_count[hit_env_ids] += 1.0

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
        """Update online hit metrics. Radius curriculum is updated on episode reset."""
        self.metrics["hit_count"][:] = hit_mask.float()
        self.metrics["max_hit_count"][:] = self.cumulative_hit_count.max().float()
        self.metrics["hit_radius"][:] = float(self.current_hit_distance_threshold)
        self.metrics["hit_success_rate_window"][:] = float(self._hit_success_rate_window)

    def _record_episode_outcomes_and_update_curriculum(self, env_ids: torch.Tensor):
        """Update hit-radius curriculum from completed episodes."""
        if (not self.hit_radius_curriculum_enabled) or (len(env_ids) == 0):
            return

        outcomes = self.has_hit[env_ids].float().tolist()
        if len(outcomes) == 0:
            return
        self._hit_outcome_window.extend(outcomes)

        history_size = len(self._hit_outcome_window)
        if history_size == 0:
            self._hit_success_rate_window = 0.0
            return

        self._hit_success_rate_window = float(sum(self._hit_outcome_window) / history_size)
        if history_size < self.hit_curriculum_window:
            return

        if self._hit_success_rate_window >= self.hit_curriculum_success_threshold:
            new_radius = max(
                self.hit_radius_end,
                float(self.current_hit_distance_threshold * self.hit_radius_shrink_factor),
            )
            if new_radius < self.current_hit_distance_threshold - 1e-6:
                old_radius = self.current_hit_distance_threshold
                self.current_hit_distance_threshold = new_radius
                self.hit_distance_threshold = new_radius
                self._hit_outcome_window.clear()
                self._hit_success_rate_window = 0.0
                print(
                    f"[HitCurriculum] radius shrink: {old_radius:.4f} -> {new_radius:.4f} "
                    f"(threshold={self.hit_curriculum_success_threshold:.2f})"
                )

    def _resolve_amp_key_body_indices(self) -> torch.Tensor:
        """Resolve key-body indices for MimicKit-like AMP observations."""
        if len(self.cfg.amp_disc_key_body_names) == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)

        # IMPORTANT:
        # Use cfg.body_names indexing (same basis as robot_body_pos_w) to avoid
        # mismatches with low-level robot.data.body_pos_w indexing across versions.
        body_name_to_index = {name: idx for idx, name in enumerate(self.cfg.body_names)}
        resolved_indices: list[int] = []
        missing: list[str] = []
        for name in self.cfg.amp_disc_key_body_names:
            if name in body_name_to_index:
                resolved_indices.append(body_name_to_index[name])
            else:
                missing.append(name)
        if missing:
            print(
                "[AMPReward] 以下 key body 在当前机器人中不存在，已忽略: "
                + ", ".join(missing)
            )
        # Keep on CPU during initialization to avoid masking earlier async CUDA errors.
        return torch.tensor(resolved_indices, dtype=torch.long, device="cpu")

    @staticmethod
    def _parse_urdf_joint_axes(urdf_path: str | pathlib.Path) -> dict[str, list[float]]:
        """Parse revolute/continuous/prismatic joint axes from URDF."""
        urdf_path = pathlib.Path(urdf_path).expanduser().resolve()
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF 不存在: {urdf_path}")

        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        axes: dict[str, list[float]] = {}
        for joint_elem in root.findall("joint"):
            joint_name = joint_elem.attrib.get("name", "")
            joint_type = joint_elem.attrib.get("type", "")
            if len(joint_name) == 0:
                continue
            if joint_type in {"fixed", "floating", "planar"}:
                continue

            axis_elem = joint_elem.find("axis")
            if axis_elem is None:
                axes[joint_name] = [1.0, 0.0, 0.0]
                continue

            axis_str = axis_elem.attrib.get("xyz", "1 0 0").split()
            if len(axis_str) != 3:
                axes[joint_name] = [1.0, 0.0, 0.0]
            else:
                axes[joint_name] = [float(axis_str[0]), float(axis_str[1]), float(axis_str[2])]
        return axes

    def _compute_head_link_insert_pos(self) -> int:
        """Return the position (0-indexed) to insert the synthetic head_link key body.

        MimicKit's G1 MJCF has a FIXED head_link body that IsaacLab's G1 lacks.
        This method determines where in the *resolved* key_pos array the head_link
        entry should be inserted so the order matches MimicKit's disc_obs layout.
        Returns -1 when head_link is already present (no insertion needed) or is
        not configured as a key body at all.
        """
        cfg_names = list(self.cfg.amp_disc_key_body_names)
        if "head_link" not in cfg_names:
            return -1
        body_name_to_index = {name: idx for idx, name in enumerate(self.cfg.body_names)}
        if "head_link" in body_name_to_index:
            return -1  # already resolved — no synthetic insertion needed
        head_cfg_pos = cfg_names.index("head_link")
        # Count how many configured key body names *before* head_link are found
        # in the robot's body list (i.e. were resolved and stored in key_pos).
        insert_pos = sum(1 for name in cfg_names[:head_cfg_pos] if name in body_name_to_index)
        return insert_pos

    def _try_g1_joint_axes(self, default_axes: torch.Tensor) -> torch.Tensor:
        """Return G1 hardcoded joint axes if the robot joints match, else default."""
        joint_names = list(self.robot.joint_names)[: self._amp_num_joints]
        matched = sum(1 for n in joint_names if n in _G1_JOINT_AXES)
        if matched < max(1, self._amp_num_joints // 2):
            print("[AMPReward] Joint names don't match G1 — using default x-axis for all.")
            return default_axes
        axes = default_axes.clone()
        for idx, jname in enumerate(joint_names):
            if jname in _G1_JOINT_AXES:
                axes[idx] = torch.tensor(_G1_JOINT_AXES[jname], dtype=torch.float32, device=self.device)
        print(f"[AMPReward] Applied G1 hardcoded joint axes ({matched}/{self._amp_num_joints} matched).")
        return axes

    def _resolve_amp_joint_axes(self) -> torch.Tensor:
        """Resolve one axis per joint in robot.joint_names order."""
        num_joints = self._amp_num_joints
        if num_joints <= 0:
            return torch.zeros(0, 3, dtype=torch.float32, device=self.device)

        default_axes = torch.zeros(num_joints, 3, dtype=torch.float32, device=self.device)
        default_axes[:, 0] = 1.0

        urdf_path = self.cfg.amp_disc_joint_axis_urdf_path
        if len(urdf_path) == 0:
            spawn_cfg = getattr(self.robot.cfg, "spawn", None)
            urdf_path = getattr(spawn_cfg, "asset_path", "") if spawn_cfg is not None else ""
        if len(urdf_path) == 0:
            return self._try_g1_joint_axes(default_axes)

        try:
            axis_map = self._parse_urdf_joint_axes(urdf_path)
        except Exception as exc:
            print(f"[AMPReward] 解析 URDF joint axis 失败，尝试 G1 硬编码轴: {exc}")
            return self._try_g1_joint_axes(default_axes)

        joint_names = list(self.robot.joint_names)
        if len(joint_names) < num_joints:
            joint_names.extend([f"joint_{i}" for i in range(len(joint_names), num_joints)])
        elif len(joint_names) > num_joints:
            joint_names = joint_names[:num_joints]

        axes = default_axes.clone()
        for idx, joint_name in enumerate(joint_names):
            if joint_name in axis_map:
                axes[idx] = torch.tensor(axis_map[joint_name], dtype=torch.float32, device=self.device)

        axis_norm = torch.linalg.norm(axes, dim=-1, keepdim=True).clamp_min(1e-6)
        axes = axes / axis_norm
        return axes

    def _quat_to_tan_norm(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        """Quaternion -> 6D tangent/normal embedding (MimicKit-compatible layout)."""
        ref_tan = torch.zeros((*quat_wxyz.shape[:-1], 3), dtype=quat_wxyz.dtype, device=quat_wxyz.device)
        ref_tan[..., 0] = 1.0
        tan = quat_apply(quat_wxyz, ref_tan)

        ref_norm = torch.zeros((*quat_wxyz.shape[:-1], 3), dtype=quat_wxyz.dtype, device=quat_wxyz.device)
        ref_norm[..., 2] = 1.0
        norm = quat_apply(quat_wxyz, ref_norm)
        return torch.cat([tan, norm], dim=-1)

    def _compute_amp_step_features(self) -> torch.Tensor:
        """Build one-step AMP features from current robot state."""
        root_pos = self.robot_anchor_pos_w
        root_quat = self.robot_anchor_quat_w
        root_lin_vel = self.robot_anchor_lin_vel_w
        root_ang_vel = self.robot_anchor_ang_vel_w
        joint_pos = self.robot_joint_pos
        joint_vel = self.robot_joint_vel

        if self.amp_disc_obs_mode == "mimickit_like":
            key_body_pos_flat = torch.zeros(self.num_envs, 0, device=self.device, dtype=root_pos.dtype)
            if self._amp_key_body_indices.numel() > 0:
                # Index into the stable, cfg.body_names-aligned body tensor.
                key_body_indices = self._amp_key_body_indices.to(device=self.device, dtype=torch.long)
                key_body_pos_w = self.robot_body_pos_w[:, key_body_indices]
                key_body_pos_flat = key_body_pos_w.reshape(self.num_envs, -1)

            # Save raw slices for later history-to-disc_obs conversion.
            offset = 0
            self._amp_raw_slices = {}
            for name, size in (
                ("root_pos", 3),
                ("root_quat", 4),
                ("root_lin_vel", 3),
                ("root_ang_vel", 3),
                ("joint_pos", self._amp_num_joints),
                ("joint_vel", self._amp_num_joints),
                ("key_pos", key_body_pos_flat.shape[-1]),
            ):
                self._amp_raw_slices[name] = (offset, offset + size)
                offset += size

            return torch.cat(
                [
                    root_pos,
                    root_quat,
                    root_lin_vel,
                    root_ang_vel,
                    joint_pos,
                    joint_vel,
                    key_body_pos_flat,
                ],
                dim=-1,
            )

        # Legacy simplified AMP features.
        body_pos_rel = self.robot_body_pos_w - self.robot_anchor_pos_w.unsqueeze(1)
        body_quat = self.robot_body_quat_w
        return torch.cat(
            [
                root_pos,
                root_quat,
                root_lin_vel,
                root_ang_vel,
                joint_pos,
                joint_vel,
                body_pos_rel.reshape(self.num_envs, -1),
                body_quat.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )

    def _initialize_amp_obs_history(self):
        if not self.amp_reward_enabled:
            return
        try:
            features = self._compute_amp_step_features()
        except Exception as exc:
            # Robot buffers may be unavailable during very early initialization.
            self._amp_obs_history = None
            if not self._amp_warned_history_init_error:
                print(f"[AMPReward] 初始化 AMP 历史观测失败，将在后续 step 重试: {exc}")
                self._amp_warned_history_init_error = True
            return
        feat_dim = features.shape[-1]
        self._amp_obs_history = torch.zeros(
            self.num_envs,
            self.amp_obs_history_steps,
            feat_dim,
            device=self.device,
            dtype=features.dtype,
        )
        self._amp_obs_history[:] = features.unsqueeze(1)

    def _update_amp_obs_history(self, env_ids: torch.Tensor | None = None):
        if not self.amp_reward_enabled:
            return
        if self._amp_obs_history is None:
            self._initialize_amp_obs_history()
        if self._amp_obs_history is None:
            return
        features = self._compute_amp_step_features()
        if env_ids is None:
            self._amp_obs_history = torch.roll(self._amp_obs_history, shifts=-1, dims=1)
            self._amp_obs_history[:, -1, :] = features
            return

        if len(env_ids) == 0:
            return
        self._amp_obs_history[env_ids] = torch.roll(self._amp_obs_history[env_ids], shifts=-1, dims=1)
        self._amp_obs_history[env_ids, -1, :] = features[env_ids]

    def _reset_amp_obs_history(self, env_ids: torch.Tensor):
        if not self.amp_reward_enabled:
            return
        if self._amp_obs_history is None:
            self._initialize_amp_obs_history()
        if self._amp_obs_history is None:
            return
        if len(env_ids) == 0:
            return
        features = self._compute_amp_step_features()
        self._amp_obs_history[env_ids] = features[env_ids].unsqueeze(1).repeat(1, self.amp_obs_history_steps, 1)

    def _build_amp_disc_obs_mimickit_like(self) -> torch.Tensor:
        """Build MimicKit-style disc_obs from history of raw step features."""
        if self._amp_obs_history is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        if len(self._amp_raw_slices) == 0:
            _ = self._compute_amp_step_features()
            if len(self._amp_raw_slices) == 0:
                return torch.zeros(self.num_envs, 1, device=self.device)

        hist = self._amp_obs_history
        num_envs, num_steps, _ = hist.shape

        def _slice(name: str) -> torch.Tensor:
            start, end = self._amp_raw_slices[name]
            return hist[:, :, start:end]

        root_pos = _slice("root_pos")  # [N, T, 3]
        root_quat = _slice("root_quat")  # [N, T, 4]
        root_vel = _slice("root_lin_vel")  # [N, T, 3]
        root_ang_vel = _slice("root_ang_vel")  # [N, T, 3]
        joint_pos = _slice("joint_pos")  # [N, T, J]
        joint_vel = _slice("joint_vel")  # [N, T, J]

        key_pos_flat = _slice("key_pos")  # [N, T, 3K] or empty
        if key_pos_flat.shape[-1] > 0:
            key_pos = key_pos_flat.reshape(num_envs, num_steps, -1, 3)  # [N, T, K, 3]
        else:
            key_pos = torch.zeros(num_envs, num_steps, 0, 3, device=self.device, dtype=hist.dtype)

        # Match MimicKit's reference frame choice: use latest-step root pose.
        ref_root_pos = root_pos[:, -1, :]
        ref_root_quat = root_quat[:, -1, :]

        # Position-like terms.
        root_pos_obs = root_pos - ref_root_pos.unsqueeze(1)
        if key_pos.shape[2] > 0:
            key_pos_obs = key_pos - root_pos.unsqueeze(2)
        else:
            key_pos_obs = key_pos

        root_rot_obs_quat = root_quat
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel

        # heading_inv_t is set only when global_obs=False; used later for head_link too.
        heading_inv_t: torch.Tensor | None = None
        if not self.amp_disc_global_obs:
            heading_inv = quat_inv(yaw_quat(ref_root_quat))  # [N, 4]
            heading_inv_t = heading_inv.unsqueeze(1).repeat(1, num_steps, 1)  # [N, T, 4]

            root_pos_obs = quat_apply(heading_inv_t, root_pos_obs)
            root_rot_obs_quat = quat_mul(heading_inv_t, root_quat)
            root_vel_obs = quat_apply(heading_inv_t, root_vel)
            root_ang_vel_obs = quat_apply(heading_inv_t, root_ang_vel)

            if key_pos_obs.shape[2] > 0:
                heading_inv_key = heading_inv_t.unsqueeze(2).repeat(1, 1, key_pos_obs.shape[2], 1)
                key_pos_obs = quat_apply(heading_inv_key, key_pos_obs)

        # G1-compat: insert synthetic head_link key body at the correct position.
        # head_link is a FIXED body in MimicKit's G1 MJCF, parented to torso_link
        # with local offset _G1_MIMICKIT_HEAD_OFFSET.  root_quat in the history IS
        # the torso_link world quaternion (anchor_body_name == "torso_link").
        if self._amp_head_link_insert_pos >= 0:
            head_offset = torch.tensor(
                list(_G1_MIMICKIT_HEAD_OFFSET), dtype=hist.dtype, device=self.device
            )  # [3]
            # head pos relative to torso_link in world frame = R_torso @ head_offset
            head_pos = quat_apply(
                root_quat.reshape(-1, 4),
                head_offset.view(1, 3).expand(num_envs * num_steps, 3),
            ).reshape(num_envs, num_steps, 1, 3)  # [N, T, 1, 3]
            if heading_inv_t is not None:
                # Transform to local heading frame same as other key bodies.
                head_pos = quat_apply(
                    heading_inv_t.reshape(-1, 4),
                    head_pos.reshape(-1, 3),
                ).reshape(num_envs, num_steps, 1, 3)
            ins = self._amp_head_link_insert_pos
            key_pos_obs = torch.cat(
                [key_pos_obs[:, :, :ins, :], head_pos, key_pos_obs[:, :, ins:, :]],
                dim=2,
            )  # [N, T, K+1, 3]

        if self.amp_disc_root_height_obs:
            root_pos_obs = root_pos_obs.clone()
            root_pos_obs[..., 2] = root_pos[..., 2]
        else:
            root_pos_obs = root_pos_obs[..., :2]

        # Convert root/joint rotations to tan-norm 6D representation.
        root_rot_obs = self._quat_to_tan_norm(root_rot_obs_quat.reshape(-1, 4)).reshape(num_envs, num_steps, -1)

        num_joints = joint_pos.shape[-1]
        if num_joints > 0:
            if self._amp_joint_axes.shape[0] != num_joints:
                axis = torch.zeros(num_joints, 3, dtype=hist.dtype, device=self.device)
                axis[:, 0] = 1.0
            else:
                axis = self._amp_joint_axes.to(dtype=hist.dtype)
            axis_expand = axis.unsqueeze(0).unsqueeze(0).repeat(num_envs, num_steps, 1, 1)
            joint_quat = quat_from_angle_axis(joint_pos.reshape(-1), axis_expand.reshape(-1, 3))
            joint_quat = joint_quat.reshape(num_envs, num_steps, num_joints, 4)
            joint_rot_obs = self._quat_to_tan_norm(joint_quat.reshape(-1, 4)).reshape(num_envs, num_steps, -1)
        else:
            joint_rot_obs = torch.zeros(num_envs, num_steps, 0, dtype=hist.dtype, device=self.device)

        # G1-compat: insert identity tan-norm for MimicKit's FIXED head_link body.
        # In MimicKit's G1 MJCF, head_link sits at body traversal position 16
        # (0-indexed joint position 15, between waist_pitch and left_shoulder_pitch),
        # contributing an identity quaternion to joint_rot_obs.
        if self._amp_mimickit_g1_compat and num_joints == _G1_MIMICKIT_HEAD_JOINT_INSERT_POS + 14:
            # 29 actual DOF joints → insert at position 15 → result is 30 × 6D = 180D.
            identity_tan = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=hist.dtype, device=self.device
            ).view(1, 1, 6).expand(num_envs, num_steps, 6)
            ins = _G1_MIMICKIT_HEAD_JOINT_INSERT_POS * 6  # = 90
            joint_rot_obs = torch.cat(
                [joint_rot_obs[:, :, :ins], identity_tan, joint_rot_obs[:, :, ins:]],
                dim=-1,
            )  # [N, T, 180]

        key_pos_obs_flat = key_pos_obs.reshape(num_envs, num_steps, -1)

        pos_obs = torch.cat([root_pos_obs, root_rot_obs, joint_rot_obs, key_pos_obs_flat], dim=-1)
        vel_obs = torch.cat([root_vel_obs, root_ang_vel_obs, joint_vel], dim=-1)
        disc_obs = torch.cat([pos_obs, vel_obs], dim=-1).reshape(num_envs, -1)
        return disc_obs

    def _align_amp_disc_obs_dim(self, disc_obs: torch.Tensor) -> torch.Tensor:
        if self.amp_discriminator is None:
            return disc_obs

        expected_dim = self.amp_discriminator.meta.obs_dim
        current_dim = disc_obs.shape[-1]
        if current_dim == expected_dim:
            return disc_obs

        if not self.amp_allow_obs_dim_mismatch:
            raise RuntimeError(
                f"AMP disc_obs 维度不匹配: current={current_dim}, expected={expected_dim}。"
                "请对齐 IsaacLab 与 MimicKit 的 disc_obs 定义，或开启 amp_allow_obs_dim_mismatch。"
            )

        if not self._amp_warned_dim_mismatch:
            print(
                "[AMPReward] disc_obs dimension mismatch, applying pad/trim fallback: "
                f"{current_dim} -> {expected_dim}"
            )
            self._amp_warned_dim_mismatch = True

        if current_dim > expected_dim:
            return disc_obs[:, :expected_dim]
        # Pad with the discriminator's norm_mean for the missing dims so that
        # after internal normalization  (x - mean) / std  the padded positions
        # contribute exactly 0 (neutral) rather than  -mean/std  (extreme bias).
        pad_mean = self.amp_discriminator.norm_mean[current_dim:].to(
            device=self.device, dtype=disc_obs.dtype
        )  # shape: [expected_dim - current_dim]
        pad = pad_mean.unsqueeze(0).expand(self.num_envs, -1)
        return torch.cat([disc_obs, pad], dim=-1)

    def get_amp_disc_obs(self) -> torch.Tensor:
        if not self.amp_reward_enabled:
            return torch.zeros(self.num_envs, 1, device=self.device)
        if self._amp_obs_history is None:
            self._initialize_amp_obs_history()
        if self._amp_obs_history is None:
            if self.amp_discriminator is not None:
                return torch.zeros(self.num_envs, self.amp_discriminator.meta.obs_dim, device=self.device)
            return torch.zeros(self.num_envs, 1, device=self.device)

        if self.amp_disc_obs_mode == "mimickit_like":
            disc_obs = self._build_amp_disc_obs_mimickit_like()
        else:
            disc_obs = self._amp_obs_history.reshape(self.num_envs, -1)
        return self._align_amp_disc_obs_dim(disc_obs)

    def compute_amp_style_reward(self) -> torch.Tensor:
        if (not self.amp_reward_enabled) or (self.amp_discriminator is None):
            return torch.zeros(self.num_envs, device=self.device)
        disc_obs = self.get_amp_disc_obs()
        logits = self.amp_discriminator.forward(disc_obs, normalize=True)
        prob = torch.sigmoid(logits)
        raw_reward = -torch.log(torch.clamp(1.0 - prob, min=1e-4)) * float(self.cfg.amp_disc_reward_scale)

        if "amp_obs_valid" in self.metrics:
            self.metrics["amp_obs_valid"][:] = 0.0 if (self._amp_obs_history is None) else 1.0
            self.metrics["amp_disc_logit"][:] = logits
            self.metrics["amp_disc_prob"][:] = prob
            self.metrics["amp_reward_raw"][:] = raw_reward

        return raw_reward

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def _motion_safe_idx(self) -> torch.Tensor:
        """Index into self.motion clamped to its actual length.

        In Stage4, self.motion is a 1-frame static default (time_step_total=1)
        while self.time_steps advances to track expert motion progress.
        Clamping here prevents out-of-bounds access on the static motion
        while allowing _update_expert_command_telemetry to use the real
        time_steps for per-expert motion indexing.
        """
        return self.time_steps.clamp(max=self.motion.time_step_total - 1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self._motion_safe_idx]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self._motion_safe_idx]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._motion_safe_idx] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._motion_safe_idx]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._motion_safe_idx]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._motion_safe_idx]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._motion_safe_idx, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._motion_safe_idx, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._motion_safe_idx, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._motion_safe_idx, self.motion_anchor_body_index]

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
        if self.enable_tracking_error_metrics:
            self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
            self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
            self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
            self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

            self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
                dim=-1
            )
            self.metrics["error_body_rot"] = quat_error_magnitude(
                self.body_quat_relative_w, self.robot_body_quat_w
            ).mean(dim=-1)

            self.metrics["error_body_lin_vel"] = torch.norm(
                self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
            ).mean(dim=-1)
            self.metrics["error_body_ang_vel"] = torch.norm(
                self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
            ).mean(dim=-1)

            self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
            self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        
        self.metrics["has_hit"] = self.has_hit.float()
        self.metrics["target_visible"] = self.target_is_visible.float()
        self.metrics["task_rewards_enabled"] = self.task_rewards_enabled.float()
        self._update_router_weight_metrics()

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
        
    def _resample_command(self, env_ids: Sequence[int]):
        """Episode 重置时调用，重新采样动作和目标位置"""
        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._record_episode_outcomes_and_update_curriculum(env_ids_tensor)
        self._adaptive_sampling(env_ids)
        self._update_expert_command_telemetry()
        self._resample_target_positions(env_ids_tensor)

        self.cumulative_hit_count[env_ids_tensor] = 0.0
        self.current_time[env_ids_tensor] = 0.0
        self._router_weight_episode_steps[env_ids_tensor] = 0.0
        if self._router_weight_episode_sum is not None:
            self._router_weight_episode_sum[env_ids_tensor] = 0.0

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
        self._reset_amp_obs_history(env_ids_tensor)

    def _update_command(self):
        self.time_steps += 1
        
        # Stage 2: 更新当前仿真时间 (每个 env 独立)
        # dt = decimation * sim.dt
        dt = self._env.cfg.decimation * self._env.cfg.sim.dt
        self.current_time += dt
        self.update_hit_resample_timer()

        # Keep last reference frame until environment reset.
        # Stage4: when per-expert motions are loaded, clamp by the longest expert
        # motion so that time_steps advances through all frames (instead of being
        # stuck at 0 because from_robot_static() has time_step_total=1).
        if self._expert_motions:
            _max_steps = max(m.time_step_total for m in self._expert_motions) - 1
        else:
            _max_steps = self.motion.time_step_total - 1
        self.time_steps = torch.clamp(self.time_steps, max=_max_steps)

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

        self._update_expert_command_telemetry()
        self._update_amp_obs_history()

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
            self._set_countdown_text_visibility(True)

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
            self._set_countdown_text_visibility(False)

    def _ensure_countdown_text_prim(self):
        if (not bool(getattr(self, "show_target_visibility_countdown", False))) or (not HAS_PXR_TEXT):
            return
        if hasattr(self, "_countdown_text_geom"):
            return

        stage = sim_utils.get_current_stage()
        text_geom = UsdGeom.Text.Define(stage, "/Visuals/Command/target_visible_countdown")
        text_geom.CreateTextAttr().Set("")
        text_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 1.0, 0.2)])

        xformable = UsdGeom.Xformable(text_geom.GetPrim())
        self._countdown_text_translate_op = xformable.AddTranslateOp()
        self._countdown_text_scale_op = xformable.AddScaleOp()
        self._countdown_text_scale_op.Set(Gf.Vec3f(0.15, 0.15, 0.15))
        self._countdown_text_geom = text_geom

    def _set_countdown_text_visibility(self, visible: bool):
        if (not bool(getattr(self, "show_target_visibility_countdown", False))) or (not HAS_PXR_TEXT):
            return
        self._ensure_countdown_text_prim()
        if not hasattr(self, "_countdown_text_geom"):
            return
        imageable = UsdGeom.Imageable(self._countdown_text_geom.GetPrim())
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _update_countdown_text(self):
        if (not bool(getattr(self, "show_target_visibility_countdown", False))) or (not HAS_PXR_TEXT):
            return
        self._ensure_countdown_text_prim()
        if not hasattr(self, "_countdown_text_geom"):
            return

        env_idx = int(max(0, min(self.num_envs - 1, self.countdown_env_index)))
        time_left = max(float(self.target_visible_time_left[env_idx].item()), 0.0)
        countdown_text = f"Target visible: {time_left:.2f}s"
        self._countdown_text_geom.GetTextAttr().Set(countdown_text)

        target_pos = self.target_pos_w[env_idx]
        self._countdown_text_translate_op.Set(
            Gf.Vec3d(
                float(target_pos[0].item()),
                float(target_pos[1].item()),
                float(target_pos[2].item() + self.countdown_height_offset),
            )
        )

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
        target_scales = None
        if self.target_sphere_follow_hit_radius:
            radius_scale = max(float(self.current_hit_distance_threshold / self._target_marker_base_radius), 1.0e-4)
            target_scales = torch.full((self.num_envs, 3), radius_scale, device=self.device)
        self.target_sphere_visualizer.visualize(self.target_pos_w, scales=target_scales)
        
        # 2. 绘制引导大球线框 (使用 debug lines)
        if HAS_DEBUG_DRAW and hasattr(self, "_debug_draw"):
            self._debug_draw.clear_lines()
            self._draw_guidance_spheres()
            # [已注释] 采样区域边界可视化 (课程学习已取消)
            # self._draw_sampling_regions()
        self._update_countdown_text()
    
    def _draw_guidance_spheres(self):
        """
        绘制引导大球的线框 (浅绿色)
        
        为每个环境的目标小球绘制 3 个正交圆环 (XY, YZ, XZ 平面)
        来模拟一个线框球体
        """
        if self.guidance_sphere_follow_hit_radius:
            radius = float(self.current_hit_distance_threshold)
        else:
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

    # Kept for motion name parsing / backward compatibility.
    motion_file: str = ""
    # Stage4 MoE: per-skill motion references (ordered one-by-one with frozen experts).
    # These are not used for environment command rollout itself; they are used to
    # build expert-specific command inputs for frozen skill policies.
    stage4_expert_motion_files: tuple[str, ...] = ()
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    # Stage4 can disable these to keep logs clean.
    enable_tracking_error_metrics: bool = True

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
    # If True, draw the green wire sphere using current hit radius (curriculum-aware).
    guidance_sphere_follow_hit_radius: bool = False

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
    # Debug text for target visibility countdown.
    show_target_visibility_countdown: bool = True
    countdown_env_index: int = 0
    countdown_height_offset: float = 0.35

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
    # If curriculum is enabled, this value is used as fallback and will be
    # replaced by hit_radius_start at runtime.
    hit_distance_threshold: float = 0.06

    # Hit radius curriculum for stage4:
    # start with easy large radius, then shrink as windowed hit rate improves.
    hit_radius_curriculum_enabled: bool = False
    hit_radius_start: float = 0.30
    hit_radius_end: float = 0.06
    hit_curriculum_window: int = 2000
    hit_curriculum_success_threshold: float = 0.60
    hit_radius_shrink_factor: float = 0.98
    # If True, red target sphere marker radius follows current hit radius.
    target_sphere_follow_hit_radius: bool = False

    # AMP style reward runtime config.
    amp_reward_enabled: bool = False
    amp_disc_bundle_path: str = ""
    amp_disc_activation: str = "elu"
    amp_disc_reward_scale: float = 2.0
    amp_obs_history_steps: int = 10
    # `legacy_simple`: early prototype features
    # `mimickit_like`: MimicKit-style disc_obs construction from history
    amp_disc_obs_mode: str = "mimickit_like"
    # Keep these aligned with MimicKit AMP env config.
    amp_disc_global_obs: bool = True
    amp_disc_root_height_obs: bool = True
    amp_disc_key_body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "head_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    # Optional override for URDF joint-axis source.
    # Default empty -> auto-resolve from robot cfg spawn.asset_path.
    amp_disc_joint_axis_urdf_path: str = ""
    amp_allow_obs_dim_mismatch: bool = True
    # Router telemetry in metrics (running episode mean per expert).
    enable_router_weight_metrics: bool = False
    router_metric_names: tuple[str, ...] = ()
    
    # 目标小球 Marker 配置 (红色实心球)
    target_sphere_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/target_sphere",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.06,  # 基准半径。若开启 target_sphere_follow_hit_radius，会按该值缩放。
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.2, 0.2),  # 红色
                    opacity=0.9,
                ),
            ),
        },
    )
