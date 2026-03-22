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
        self._point_attempts: list[torch.Tensor] = []
        self._point_hits: list[torch.Tensor] = []
        for skill in self._skill_names:
            key = f"{skill}{cfg.target_key_suffix}"
            if key not in data:
                raise KeyError(f"Missing key '{key}' in target file: {target_path}")
            points = torch.tensor(data[key], dtype=torch.float32, device=self.device)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Invalid points shape for {key}: {points.shape}")
            self._skill_points.append(points)
            self._point_attempts.append(torch.zeros(points.shape[0], device=self.device))
            self._point_hits.append(torch.zeros(points.shape[0], device=self.device))

        self.target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_present = torch.ones(self.num_envs, 1, device=self.device)
        self.skill_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.point_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._weight_sum: torch.Tensor | None = None
        self._weight_count = torch.zeros(self.num_envs, device=self.device)
        self._expert_names: list[str] | None = None
        self._episode_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_hit_count = torch.zeros(self.num_envs, device=self.device)
        self._hit_rate = torch.zeros(self.num_envs, device=self.device)
        self._global_episode_count = torch.tensor(0.0, device=self.device)
        self._global_hit_count = torch.tensor(0.0, device=self.device)
        # sliding-window hit-rate (episodes)
        self._hit_rate_window = max(1, int(cfg.hit_rate_window))
        self._window_hits = torch.zeros(self._hit_rate_window, device=self.device)
        self._window_ptr = 0
        self._window_count = 0
        self._window_sum = torch.tensor(0.0, device=self.device)
        self._window_rate = torch.tensor(0.0, device=self.device)
        # sampling ratio (EMA over recent resamples)
        self._sample_ratio_ema = torch.zeros(len(self._skill_points), device=self.device)

        self._robot = env.scene["robot"]
        self._pelvis_idx = self._robot.body_names.index("pelvis")

        if not hasattr(env, "_moe_episode_mismatch"):
            env._moe_episode_mismatch = torch.zeros(self.num_envs, device=self.device)
        if not hasattr(env, "_moe_episode_fall"):
            env._moe_episode_fall = torch.zeros(self.num_envs, device=self.device)

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

        self._update_hit_rate_and_point_stats(env_ids)

        skill_ids = torch.randint(0, len(self._skill_points), (len(env_ids),), device=self.device)
        self.skill_ids[env_ids] = skill_ids
        # update sampling ratio EMA
        counts = torch.bincount(skill_ids, minlength=len(self._skill_points)).float()
        total = counts.sum().clamp_min(1.0)
        batch_ratio = counts / total
        beta = float(self.cfg.sample_ratio_ema)
        beta = max(0.0, min(1.0, beta))
        if beta <= 0.0:
            self._sample_ratio_ema = batch_ratio
        else:
            self._sample_ratio_ema = (1.0 - beta) * self._sample_ratio_ema + beta * batch_ratio

        for skill_idx in range(len(self._skill_points)):
            mask = skill_ids == skill_idx
            if not torch.any(mask):
                continue
            env_mask = env_ids[mask]
            points = self._skill_points[skill_idx]
            point_ids = self._sample_point_ids(skill_idx, env_mask.numel())
            self.point_ids[env_mask] = point_ids
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
        # log target sampling ratios (EMA)
        if hasattr(self, "_sample_ratio_ema"):
            for idx, name in enumerate(self._skill_names):
                self.metrics[f"target_sample_ratio_{name}"] = torch.full(
                    (self.num_envs,), float(self._sample_ratio_ema[idx].item()), device=self.device
                )
        if hasattr(self._env, "_moe_max_weight"):
            self.metrics["moe_max_weight"] = self._env._moe_max_weight
        if hasattr(self._env, "_moe_weight_entropy"):
            self.metrics["moe_weight_entropy"] = self._env._moe_weight_entropy
        if hasattr(self._env, "_moe_one_hot"):
            self.metrics["moe_one_hot"] = self._env._moe_one_hot
        window_rate = float(self._window_rate.item()) if hasattr(self, "_window_rate") else 0.0
        self.metrics["hit_rate"] = torch.full((self.num_envs,), window_rate, device=self.device)

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

    def _update_hit_rate_and_point_stats(self, env_ids: torch.Tensor):
        if not hasattr(self._env, "termination_manager"):
            return
        term_mgr = self._env.termination_manager
        if "hit_target" not in term_mgr.active_terms:
            return
        hit_idx = term_mgr._term_name_to_term_idx["hit_target"]
        last_episode = term_mgr._last_episode_dones[:, hit_idx]
        hits = last_episode[env_ids].float()
        # episode-level mismatch penalty (applied next step)
        if getattr(self._env, "_moe_lock_skill", False) and hasattr(self._env, "_moe_current_skill"):
            mismatch = (self._env._moe_current_skill[env_ids] != self.skill_ids[env_ids]).float()
            self._env._moe_episode_mismatch[env_ids] = mismatch
        # episode-level fall flag (applied next step)
        if "fall" in term_mgr.active_terms and hasattr(self._env, "_moe_episode_fall"):
            fall_idx = term_mgr._term_name_to_term_idx["fall"]
            fall_last = term_mgr._last_episode_dones[:, fall_idx]
            self._env._moe_episode_fall[env_ids] = fall_last[env_ids].float()
        # update per-point stats
        for idx, env_id in enumerate(env_ids):
            skill = int(self.skill_ids[env_id].item())
            point = int(self.point_ids[env_id].item())
            self._point_attempts[skill][point] += 1.0
            self._point_hits[skill][point] += hits[idx].item()
        self._episode_count[env_ids] += 1.0
        self._episode_hit_count[env_ids] += hits
        self._hit_rate = self._episode_hit_count / self._episode_count.clamp_min(1.0)

        self._global_episode_count += float(env_ids.numel())
        self._global_hit_count += hits.sum()
        global_rate = self._global_hit_count / self._global_episode_count.clamp_min(1.0)
        # update sliding window hit-rate
        if self._hit_rate_window > 0:
            n = int(hits.numel())
            positions = (self._window_ptr + torch.arange(n, device=self.device)) % self._hit_rate_window
            old = self._window_hits[positions]
            self._window_hits[positions] = hits
            self._window_sum += hits.sum() - old.sum()
            self._window_ptr = int((self._window_ptr + n) % self._hit_rate_window)
            self._window_count = min(self._hit_rate_window, self._window_count + n)
            denom = float(max(1, self._window_count))
            self._window_rate = self._window_sum / denom
        else:
            self._window_rate = global_rate

    def _sample_point_ids(self, skill_idx: int, num_samples: int) -> torch.Tensor:
        points = self._skill_points[skill_idx]
        attempts = self._point_attempts[skill_idx]
        hits = self._point_hits[skill_idx]
        success = torch.zeros_like(attempts)
        nonzero = attempts > 0
        success[nonzero] = hits[nonzero] / attempts[nonzero]
        threshold = float(self.cfg.success_split_threshold)
        threshold = max(0.0, min(1.0, threshold))
        low_mask = success < threshold
        weights = torch.full_like(success, float(self.cfg.high_success_weight))
        weights[low_mask] = float(self.cfg.low_success_weight)
        # avoid all-zero weights
        if torch.sum(weights) <= 0:
            weights = torch.ones_like(weights)
        return torch.multinomial(weights, num_samples=num_samples, replacement=True)


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
    low_success_weight: float = 2.0
    high_success_weight: float = 1.0
    success_split_threshold: float = 0.7
    hit_rate_window: int = 20000
    sample_ratio_ema: float = 0.1
