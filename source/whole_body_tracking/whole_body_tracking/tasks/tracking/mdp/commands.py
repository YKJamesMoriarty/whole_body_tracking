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
    "x": (0.6, 0.65),   # 前方 50-60cm，有效攻击距离
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
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        # =====================================================================
        # Stage 2: 目标小球 (Target Sphere) 初始化
        # 用于 Task-Oriented RL 的击打目标
        # =====================================================================
        
        # 目标小球的世界坐标位置 (num_envs, 3)
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 获取参考动作第一帧的 Root (Pelvis) 位置和朝向
        # 这是采样区域的基准位置，在整个 episode 中保持不变
        self._reference_root_pos_w = self.motion.body_pos_w[0, 0].clone()  # (3,) 第一帧的 pelvis 位置
        self._reference_root_quat_w = self.motion.body_quat_w[0, 0].clone()  # (4,) 第一帧的 pelvis 朝向
        
        # 采样范围配置 (相对于机器人 Root 的局部坐标)
        self._target_sampling_range = self.cfg.target_sampling_range
        
        # 引导大球半径 (固定值)
        self._guidance_sphere_radius = self.cfg.guidance_sphere_radius
        
        # 初始化目标位置（会在 reset 时被重新采样）
        self._init_target_positions()

    def _init_target_positions(self):
        """初始化所有环境的目标位置"""
        # 在采样范围内随机采样目标位置
        self._resample_target_positions(torch.arange(self.num_envs, device=self.device))
    
    def _resample_target_positions(self, env_ids: torch.Tensor):
        """
        为指定环境重新采样目标位置
        
        采样逻辑:
        1. 在局部坐标系 (相对于参考动作第一帧的 Root) 中采样
        2. 将局部坐标转换到世界坐标系
        3. 加上各环境的 origin 偏移
        
        Args:
            env_ids: 需要重新采样的环境索引
        """
        if len(env_ids) == 0:
            return
        
        num_samples = len(env_ids)
        
        # 在局部坐标系中采样目标位置
        x_range = self._target_sampling_range["x"]
        y_range = self._target_sampling_range["y"]
        z_range = self._target_sampling_range["z"]
        
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

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        
        # Stage 2: 重新采样目标位置
        self._resample_target_positions(torch.tensor(env_ids, device=self.device))

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
        """
        color = [0.3, 0.5, 1.0, 0.8]  # 蓝色 RGBA
        thickness = 2.0
        
        x_range = self._target_sampling_range["x"]
        y_range = self._target_sampling_range["y"]
        z_range = self._target_sampling_range["z"]
        
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
    
    # 目标小球采样范围 (相对于机器人 Root/Pelvis 的局部坐标系)
    # 基于 Unitree G1 机器人的尺寸设计，适合拳击攻击动作
    target_sampling_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: DEFAULT_TARGET_SAMPLING_RANGE.copy()
    )
    
    # 引导大球半径 (固定值，用于可视化奖励生效范围)
    guidance_sphere_radius: float = 0.25
    
    # 目标小球 Marker 配置 (红色实心球)
    target_sphere_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/target_sphere",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.06,  # 半径 6cm
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.2, 0.2),  # 红色
                    opacity=0.9,
                ),
            ),
        },
    )
