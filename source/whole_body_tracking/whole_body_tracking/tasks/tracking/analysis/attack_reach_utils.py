from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import time

import isaaclab.utils.math as math_utils


@dataclass(frozen=True)
class ReachConfig:
    x_threshold: float
    sample_every: int
    episodes: int


@dataclass(frozen=True)
class ResolvedSkill:
    name: str
    model_path: Path
    motion_path: Path
    end_effector: str


def resolve_skill_paths(skills, model_dir: str | Path, motion_dir: str | Path) -> list[ResolvedSkill]:
    model_dir = Path(model_dir)
    motion_dir = Path(motion_dir)
    resolved: list[ResolvedSkill] = []
    for skill in skills:
        model_path = model_dir / skill.model_file
        motion_path = motion_dir / skill.motion_file
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not motion_path.is_file():
            raise FileNotFoundError(f"Motion not found: {motion_path}")
        resolved.append(
            ResolvedSkill(
                name=skill.name,
                model_path=model_path,
                motion_path=motion_path,
                end_effector=skill.end_effector,
            )
        )
    return resolved


def _get_body_index(robot, body_name: str) -> int:
    body_ids, _ = robot.find_bodies([body_name], preserve_order=True)
    if not body_ids:
        raise ValueError(f"Body name not found: {body_name}")
    return int(body_ids[0])


def collect_reach_points(
    env,
    policy,
    *,
    end_effector: str,
    root_body: str,
    x_threshold: float,
    sample_every: int,
    episodes: int,
    simulation_app,
    progress_prefix: str = "",
    log_every_steps: int = 0,
) -> dict[str, np.ndarray]:
    """Collect end-effector points for the requested number of episodes.

    Returns a dict with:
      - points_root: Nx3 points in root (pelvis) frame
      - points_world: Nx3 points in world frame
      - root_pos_w: Nx3 pelvis position in world frame (per kept sample)
      - root_quat_w: Nx4 pelvis orientation in world frame (per kept sample)
    """
    if sample_every <= 0:
        raise ValueError("sample_every must be >= 1")
    if episodes <= 0:
        raise ValueError("episodes must be >= 1")

    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    ee_idx = _get_body_index(robot, end_effector)
    root_idx = _get_body_index(robot, root_body)

    obs = env.get_observations()
    # initial root pose per env
    root_pos_w0 = robot.data.body_pos_w[:, root_idx].detach().cpu().numpy()
    root_quat_w0 = robot.data.body_quat_w[:, root_idx].detach().cpu().numpy()
    completed_episodes = 0
    last_completed_per_env = -1
    step_count = 0
    points_root: list[np.ndarray] = []
    points_world: list[np.ndarray] = []
    root_pos_world: list[np.ndarray] = []
    root_quat_world: list[np.ndarray] = []
    attack_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    start_time = time.monotonic()
    prefix = f"{progress_prefix} " if progress_prefix else ""
    num_envs = int(getattr(env, "num_envs", base_env.num_envs))
    target_total_episodes = episodes * num_envs

    while simulation_app.is_running() and completed_episodes < target_total_episodes:
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        if step_count % sample_every == 0:
            ee_pos_w = robot.data.body_pos_w[:, ee_idx]
            ee_vel_w = robot.data.body_lin_vel_w[:, ee_idx]
            root_pos_w = robot.data.body_pos_w[:, root_idx]
            root_quat_w = robot.data.body_quat_w[:, root_idx]
            root_vel_w = robot.data.body_lin_vel_w[:, root_idx]

            ee_pos_b = math_utils.quat_apply_inverse(root_quat_w, ee_pos_w - root_pos_w)
            # Use velocity relative to root to capture limb extension/retraction.
            ee_vel_rel_w = ee_vel_w - root_vel_w
            ee_vel_b = math_utils.quat_apply_inverse(root_quat_w, ee_vel_rel_w)

            # Attack-phase detection
            dot_pos_vel = (ee_pos_b * ee_vel_b).sum(dim=-1)
            enter_attack = torch.logical_and(ee_pos_b[:, 0] > x_threshold, dot_pos_vel > 0.0)
            attack_active = torch.logical_or(attack_active, enter_attack)
            exit_attack = ee_pos_b[:, 0] <= x_threshold
            attack_active = torch.logical_and(attack_active, ~exit_attack)

            mask = attack_active
            if torch.any(mask):
                points_root.append(ee_pos_b[mask].detach().cpu().numpy())
                points_world.append(ee_pos_w[mask].detach().cpu().numpy())
                root_pos_world.append(root_pos_w[mask].detach().cpu().numpy())
                root_quat_world.append(root_quat_w[mask].detach().cpu().numpy())

        step_count += 1

        if dones is not None:
            if isinstance(dones, torch.Tensor):
                dones_bool = dones.to(dtype=torch.bool)
                completed_episodes += int(dones_bool.sum().item())
                attack_active = torch.where(dones_bool, torch.zeros_like(attack_active), attack_active)
            else:
                completed_episodes += int(np.sum(dones))

        completed_per_env = completed_episodes / max(num_envs, 1)
        completed_per_env_int = int(completed_per_env)
        if completed_per_env_int != last_completed_per_env:
            elapsed = time.monotonic() - start_time
            print(
                f"[INFO] {prefix}episodes {completed_per_env_int}/{episodes} "
                f"(steps={step_count}, elapsed={elapsed:.1f}s)"
            )
            last_completed_per_env = completed_per_env_int

        if log_every_steps > 0 and step_count % log_every_steps == 0:
            elapsed = time.monotonic() - start_time
            completed_per_env = completed_episodes / max(num_envs, 1)
            print(
                f"[INFO] {prefix}steps {step_count} (episodes {completed_per_env:.1f}/{episodes}, {elapsed:.1f}s)"
            )

    # final root pose per env
    root_pos_w_end = robot.data.body_pos_w[:, root_idx].detach().cpu().numpy()
    root_quat_w_end = robot.data.body_quat_w[:, root_idx].detach().cpu().numpy()

    if not points_root:
        empty = np.zeros((0, 3), dtype=np.float32)
        return {
            "points_root": empty,
            "points_world": empty,
            "root_pos_w": empty,
            "root_quat_w": np.zeros((0, 4), dtype=np.float32),
            "root_pos_w0": root_pos_w0.astype(np.float32, copy=False),
            "root_quat_w0": root_quat_w0.astype(np.float32, copy=False),
            "root_pos_w_end": root_pos_w_end.astype(np.float32, copy=False),
            "root_quat_w_end": root_quat_w_end.astype(np.float32, copy=False),
        }

    return {
        "points_root": np.concatenate(points_root, axis=0).astype(np.float32, copy=False),
        "points_world": np.concatenate(points_world, axis=0).astype(np.float32, copy=False),
        "root_pos_w": np.concatenate(root_pos_world, axis=0).astype(np.float32, copy=False),
        "root_quat_w": np.concatenate(root_quat_world, axis=0).astype(np.float32, copy=False),
        "root_pos_w0": root_pos_w0.astype(np.float32, copy=False),
        "root_quat_w0": root_quat_w0.astype(np.float32, copy=False),
        "root_pos_w_end": root_pos_w_end.astype(np.float32, copy=False),
        "root_quat_w_end": root_quat_w_end.astype(np.float32, copy=False),
    }
