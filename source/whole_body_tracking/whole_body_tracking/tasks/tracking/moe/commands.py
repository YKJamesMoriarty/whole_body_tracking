from __future__ import annotations

from dataclasses import MISSING, field
from pathlib import Path

import numpy as np
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms

from .experts import ATTACK_SKILLS
from .utils import DEFAULT_EEF_NAMES, compute_min_eef_distance


class AttackTargetCommand(CommandTerm):
    cfg: AttackTargetCommandCfg

    def __init__(self, cfg: AttackTargetCommandCfg, env):
        super().__init__(cfg, env)

        target_path = Path(cfg.target_file)
        if not target_path.is_absolute():
            target_path = Path.cwd() / target_path
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_path}")

        data = np.load(target_path)
        self._skill_names = list(cfg.skill_names)
        self._skill_points: list[torch.Tensor] = []
        for skill in self._skill_names:
            key = f"{skill}{cfg.target_key_suffix}"
            if key not in data:
                raise KeyError(f"Missing key '{key}' in target file: {target_path}")
            points = torch.tensor(data[key], dtype=torch.float32, device=self.device)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Invalid points shape for {key}: {points.shape}")
            self._skill_points.append(points)

        self.target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_present = torch.ones(self.num_envs, 1, device=self.device)
        self.skill_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._weight_sum: torch.Tensor | None = None
        self._weight_count = torch.zeros(self.num_envs, device=self.device)
        self._expert_names: list[str] | None = None
        self._episode_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_hit_count = torch.zeros(self.num_envs, device=self.device)
        self._hit_rate = torch.zeros(self.num_envs, device=self.device)
        self._global_episode_count = torch.tensor(0.0, device=self.device)
        self._global_hit_count = torch.tensor(0.0, device=self.device)

        self._robot = env.scene["robot"]
        self._pelvis_idx = self._robot.body_names.index("pelvis")

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos_b

    def _resample_command(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if env_ids.numel() == 0:
                return

        self._update_hit_rate(env_ids)

        skill_ids = torch.randint(0, len(self._skill_points), (len(env_ids),), device=self.device)
        self.skill_ids[env_ids] = skill_ids

        for skill_idx in range(len(self._skill_points)):
            mask = skill_ids == skill_idx
            if not torch.any(mask):
                continue
            env_mask = env_ids[mask]
            points = self._skill_points[skill_idx]
            point_ids = torch.randint(0, points.shape[0], (env_mask.numel(),), device=self.device)
            self.target_pos_b[env_mask] = points[point_ids]

        self._elapsed_steps[env_ids] = 0
        self._update_target_present(env_ids)

        # initialize previous distance for progress reward
        eef_names = self.cfg.eef_body_names or DEFAULT_EEF_NAMES
        if not hasattr(self._env, "_moe_prev_dist"):
            self._env._moe_prev_dist = torch.zeros(self.num_envs, device=self.device)
        min_dist = compute_min_eef_distance(self._env, self.target_pos_b, eef_names)
        self._env._moe_prev_dist[env_ids] = min_dist[env_ids]

        if self._weight_sum is not None:
            self._weight_sum[env_ids] = 0.0
            self._weight_count[env_ids] = 0.0

    def _update_command(self):
        self._elapsed_steps += 1
        self._update_target_present(slice(None))

    def _update_target_present(self, env_ids):
        if self.cfg.visible_steps is None or self.cfg.visible_steps < 0:
            self.target_present[env_ids] = 1.0
        else:
            visible = self._elapsed_steps[env_ids] < self.cfg.visible_steps
            self.target_present[env_ids] = visible.float().unsqueeze(-1)

    def _update_metrics(self):
        if self._expert_names is None:
            self._init_expert_metrics()
        if self._expert_names is None:
            return
        if not hasattr(self._env, "_moe_last_weights"):
            return
        weights = self._env._moe_last_weights
        if self._weight_sum is None:
            return
        self._weight_sum += weights
        self._weight_count += 1.0
        denom = self._weight_count.clamp_min(1.0).unsqueeze(-1)
        avg = self._weight_sum / denom
        for idx, name in enumerate(self._expert_names):
            self.metrics[f"weight_{name}"] = avg[:, idx]
        if hasattr(self._env, "_moe_max_weight"):
            self.metrics["moe_max_weight"] = self._env._moe_max_weight
        if hasattr(self._env, "_moe_weight_entropy"):
            self.metrics["moe_weight_entropy"] = self._env._moe_weight_entropy
        if hasattr(self._env, "_moe_one_hot"):
            self.metrics["moe_one_hot"] = self._env._moe_one_hot
        if float(self._global_episode_count.item()) > 0:
            global_rate = (self._global_hit_count / self._global_episode_count.clamp_min(1.0)).item()
        else:
            global_rate = 0.0
        self.metrics["hit_rate"] = torch.full((self.num_envs,), float(global_rate), device=self.device)
        stage = 0.0
        if getattr(self._env, "_moe_lock_skill", False) is False:
            stage = 1.0
        self.metrics["stage"] = torch.full((self.num_envs,), stage, device=self.device)

    def _init_expert_metrics(self):
        try:
            moe_term = self._env.action_manager.get_term(self.cfg.moe_term_name)
        except Exception:
            return
        names = getattr(moe_term, "expert_names", None)
        if not names:
            return
        self._expert_names = list(names)
        self._weight_sum = torch.zeros(self.num_envs, len(self._expert_names), device=self.device)
        for name in self._expert_names:
            self.metrics[f"weight_{name}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["moe_max_weight"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["moe_weight_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["moe_one_hot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hit_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stage"] = torch.zeros(self.num_envs, device=self.device)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_target_vis"):
                marker_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/Command/attack_target")
                if self.cfg.visual_radius is not None:
                    for marker in marker_cfg.markers.values():
                        if hasattr(marker, "radius"):
                            marker.radius = float(self.cfg.visual_radius)
                self._target_vis = VisualizationMarkers(marker_cfg)
            self._target_vis.set_visibility(True)
        else:
            if hasattr(self, "_target_vis"):
                self._target_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_target_vis"):
            return
        pelvis_pos_w = self._robot.data.body_pos_w[:, self._pelvis_idx]
        pelvis_quat_w = self._robot.data.body_quat_w[:, self._pelvis_idx]
        target_pos_w, _ = combine_frame_transforms(pelvis_pos_w, pelvis_quat_w, self.target_pos_b)
        marker_indices = torch.where(
            self.target_present.squeeze(-1) > 0.5,
            torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
            torch.full((self.num_envs,), 2, device=self.device, dtype=torch.long),
        )
        self._target_vis.visualize(translations=target_pos_w, marker_indices=marker_indices)

    def _update_hit_rate(self, env_ids: torch.Tensor):
        if not self.cfg.auto_stage_switch:
            return
        if not hasattr(self._env, "termination_manager"):
            return
        if "hit_target" not in self._env.termination_manager.active_terms:
            return
        term_idx = self._env.termination_manager._term_name_to_term_idx["hit_target"]
        last_episode = self._env.termination_manager._last_episode_dones[:, term_idx]
        hits = last_episode[env_ids].float()
        self._episode_count[env_ids] += 1.0
        self._episode_hit_count[env_ids] += hits
        self._hit_rate = self._episode_hit_count / self._episode_count.clamp_min(1.0)

        self._global_episode_count += float(env_ids.numel())
        self._global_hit_count += hits.sum()
        global_rate = self._global_hit_count / self._global_episode_count.clamp_min(1.0)
        if getattr(self._env, "_moe_lock_skill", False):
            if global_rate >= self.cfg.hit_rate_threshold:
                self._env._moe_lock_skill = False


@configclass
class AttackTargetCommandCfg(CommandTermCfg):
    class_type: type = AttackTargetCommand

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    target_file: str = MISSING
    target_key_suffix: str = "_points"
    skill_names: list[str] = field(default_factory=lambda: list(ATTACK_SKILLS))
    visible_steps: int | None = None
    eef_body_names: list[str] = field(default_factory=lambda: list(DEFAULT_EEF_NAMES))
    moe_term_name: str = "moe"
    visual_radius: float | None = None
    hit_rate_threshold: float = 0.95
    auto_stage_switch: bool = True
