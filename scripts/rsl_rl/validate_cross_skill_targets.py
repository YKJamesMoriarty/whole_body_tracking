"""Validate whether targets from one skill can be hit by other skills (visual + quantitative)."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Cross-skill target validation (visual + quantitative).")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (recommended 1).")
parser.add_argument("--episodes", type=int, default=5, help="Number of episodes per evaluation skill.")
parser.add_argument("--hit_radius", type=float, default=0.05, help="Hit radius in root frame.")
parser.add_argument(
    "--targets_file",
    type=str,
    default="outputs/attack_target_final/moe_targets.npz",
    help="NPZ containing <skill>_points in root frame.",
)
parser.add_argument(
    "--target_skill",
    type=str,
    default="swing_right_head",
    help="Which skill's target points to evaluate.",
)
parser.add_argument(
    "--eval_skills",
    nargs="*",
    default=None,
    help="Skills to evaluate against the target points (default: all attack skills).",
)
parser.add_argument(
    "--model_dir",
    type=str,
    default="basic_model/Mimic_refine",
    help="Directory containing skill model checkpoints.",
)
parser.add_argument(
    "--motion_dir",
    type=str,
    default="iros_motion/npz",
    help="Directory containing motion npz files.",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="outputs/attack_target_reports_cross",
    help="Output directory for hit reports.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ================= Fix ModuleNotFoundError: No module named 'isaacsim.asset' =================
from isaacsim.core.utils.extensions import enable_extension

try:
    enable_extension("isaacsim.asset.importer.urdf")
except Exception:
    enable_extension("omni.isaac.urdf_importer")
# ============================================================================================

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.analysis.attack_reach_config import filter_skills
from whole_body_tracking.tasks.tracking.analysis.attack_reach_utils import resolve_skill_paths


ROOT_BODY_NAME = "pelvis"


def _get_body_index(robot, body_name: str) -> int:
    body_ids, _ = robot.find_bodies([body_name], preserve_order=True)
    if not body_ids:
        raise ValueError(f"Body name not found: {body_name}")
    return int(body_ids[0])


def _targets_world(root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, points_root: torch.Tensor) -> torch.Tensor:
    """Convert points in root frame to world frame (num_envs must be 1)."""
    quat = root_quat_w[0].unsqueeze(0).repeat(points_root.shape[0], 1)
    pos = root_pos_w[0].unsqueeze(0).repeat(points_root.shape[0], 1)
    return pos + math_utils.quat_apply(quat, points_root)


def _load_targets(targets_file: Path, target_skill: str) -> np.ndarray:
    if not targets_file.is_file():
        raise FileNotFoundError(f"Targets not found: {targets_file}")
    data = np.load(targets_file)
    key = f"{target_skill}_points"
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {targets_file}")
    return data[key]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = agent_cfg.seed

    if env_cfg.scene.num_envs != 1:
        raise ValueError("This validation script currently supports num_envs=1 for clear visualization.")

    skills = filter_skills(args_cli.eval_skills)
    resolved_skills = resolve_skill_paths(skills, args_cli.model_dir, args_cli.motion_dir)

    targets_file = Path(args_cli.targets_file)
    points_root_np = _load_targets(targets_file, args_cli.target_skill)
    if points_root_np.size == 0:
        raise ValueError(f"No targets for {args_cli.target_skill} in {targets_file}")

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = None
    results: dict[str, np.ndarray] = {}

    for skill in resolved_skills:
        print(f"[INFO] Evaluating target '{args_cli.target_skill}' with skill: {skill.name}")

        if env is None:
            env_cfg.commands.motion.motion_file = str(skill.motion_path)
            env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)
            env = RslRlVecEnvWrapper(env)
        else:
            motion_cmd = env.unwrapped.command_manager.get_term("motion")
            motion_cmd.reload_motion(str(skill.motion_path))
            env.reset()

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(skill.model_path))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        ee_idx = _get_body_index(robot, skill.end_effector)
        root_idx = _get_body_index(robot, ROOT_BODY_NAME)

        points_root = torch.tensor(points_root_np, dtype=torch.float32, device=env.unwrapped.device)
        hit_counts = torch.zeros(points_root.shape[0], dtype=torch.int32, device=env.unwrapped.device)
        episode_hit = torch.zeros(points_root.shape[0], dtype=torch.bool, device=env.unwrapped.device)

        # visualization markers (red=0, green=1)
        marker_cfg = POSITION_GOAL_MARKER_CFG.copy()
        marker_cfg.prim_path = f"/World/Visuals/CrossSkillTargets/{args_cli.target_skill}/{skill.name}"
        for marker in marker_cfg.markers.values():
            if hasattr(marker, "radius"):
                marker.radius = float(args_cli.hit_radius)
        markers = VisualizationMarkers(marker_cfg)
        markers.set_visibility(True)

        marker_indices = torch.zeros(points_root.shape[0], dtype=torch.long, device=env.unwrapped.device)

        obs = env.get_observations()
        episodes_done = 0
        step_count = 0

        while simulation_app.is_running() and episodes_done < args_cli.episodes:
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            root_pos_w = robot.data.body_pos_w[:, root_idx]
            root_quat_w = robot.data.body_quat_w[:, root_idx]
            ee_pos_w = robot.data.body_pos_w[:, ee_idx]
            ee_pos_b = math_utils.quat_apply_inverse(root_quat_w, ee_pos_w - root_pos_w)

            # hit test in root frame
            diff = points_root - ee_pos_b[0]
            dist = torch.linalg.norm(diff, dim=1)
            hits = dist <= args_cli.hit_radius
            new_hits = torch.logical_and(hits, ~episode_hit)
            if torch.any(new_hits):
                episode_hit = torch.logical_or(episode_hit, new_hits)
                hit_counts += new_hits.to(dtype=torch.int32)
                marker_indices[new_hits] = 1

            # update markers every step (targets move with pelvis)
            target_world = _targets_world(root_pos_w, root_quat_w, points_root)
            markers.visualize(translations=target_world, marker_indices=marker_indices)

            step_count += 1

            done = bool(dones[0].item()) if isinstance(dones, torch.Tensor) else bool(dones)
            if done:
                episodes_done += 1
                episode_hit.zero_()
                marker_indices.zero_()
                root_pos_w = robot.data.body_pos_w[:, root_idx]
                root_quat_w = robot.data.body_quat_w[:, root_idx]
                target_world = _targets_world(root_pos_w, root_quat_w, points_root)
                markers.visualize(translations=target_world, marker_indices=marker_indices)
                print(f"[INFO] {skill.name}: episode {episodes_done}/{args_cli.episodes} done (steps={step_count})")

        hit_counts_cpu = hit_counts.detach().cpu().numpy()
        results[skill.name] = hit_counts_cpu

        report_path = out_dir / f"{args_cli.target_skill}_by_{skill.name}_hit_report.npz"
        report = {
            "target_skill": args_cli.target_skill,
            "eval_skill": skill.name,
            "points_root": points_root_np,
            "hit_counts": hit_counts_cpu,
            "hit_radius": args_cli.hit_radius,
            "episodes": args_cli.episodes,
        }
        for k in range(args_cli.episodes + 1):
            report[f"points_hit_{k}"] = points_root_np[hit_counts_cpu == k]
        np.savez(report_path, **report)
        print(f"[INFO] Saved report: {report_path}")
        print(
            f"[INFO] {skill.name}: hit>=1 {int((hit_counts_cpu > 0).sum())}/{hit_counts_cpu.shape[0]} "
            f"(missed {int((hit_counts_cpu == 0).sum())})"
        )

        markers.set_visibility(False)

    # Summary: which target points swing cannot hit but other skills can.
    if args_cli.target_skill in results:
        miss_mask = results[args_cli.target_skill] == 0
        miss_count = int(miss_mask.sum())
        print(f"[SUMMARY] {args_cli.target_skill}: missed {miss_count}/{len(miss_mask)}")
        for name, counts in results.items():
            if name == args_cli.target_skill:
                continue
            hit_other = int(((counts > 0) & miss_mask).sum())
            print(f"[SUMMARY] points missed by {args_cli.target_skill} but hit by {name}: {hit_other}")

    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
