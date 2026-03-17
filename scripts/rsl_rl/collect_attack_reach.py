"""Collect reachability points for attack skills using policy rollout."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect attack reachability points from trained policies.")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--episodes", type=int, default=8, help="Number of episodes to sample per run.")
parser.add_argument("--x_threshold", type=float, default=0.2, help="X threshold in root frame for valid points.")
parser.add_argument("--sample_every", type=int, default=1, help="Sample every N steps.")
parser.add_argument(
    "--log_every_steps",
    type=int,
    default=0,
    help="Optional step-based progress logging. 0 disables.",
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
    default="outputs/attack_reach",
    help="Output directory for collected points.",
)
parser.add_argument(
    "--skills",
    nargs="*",
    default=None,
    help="Optional list of skill names to run. Default runs all configured skills.",
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
# Must be enabled after app starts and before other Isaac Lab modules are imported.
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
from whole_body_tracking.tasks.tracking.analysis.attack_reach_utils import (
    ReachConfig,
    collect_reach_points,
    resolve_skill_paths,
)

ROOT_BODY_NAME = "pelvis"


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = agent_cfg.seed

    reach_cfg = ReachConfig(
        x_threshold=args_cli.x_threshold,
        sample_every=args_cli.sample_every,
        episodes=args_cli.episodes,
    )

    skills = filter_skills(args_cli.skills)
    resolved_skills = resolve_skill_paths(skills, args_cli.model_dir, args_cli.motion_dir)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = None
    for skill in resolved_skills:
        print(f"[INFO] Collecting reach points for skill: {skill.name}")

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

        collected = collect_reach_points(
            env,
            policy,
            end_effector=skill.end_effector,
            root_body=ROOT_BODY_NAME,
            x_threshold=reach_cfg.x_threshold,
            sample_every=reach_cfg.sample_every,
            episodes=reach_cfg.episodes,
            simulation_app=simulation_app,
            progress_prefix=skill.name,
            log_every_steps=args_cli.log_every_steps,
        )

        out_path = out_dir / f"{skill.name}_reach.npz"
        np.savez(
            out_path,
            points_root=collected["points_root"],
            points_world=collected["points_world"],
            root_pos_w=collected["root_pos_w"],
            root_quat_w=collected["root_quat_w"],
            root_pos_w0=collected["root_pos_w0"],
            root_quat_w0=collected["root_quat_w0"],
            root_pos_w_end=collected["root_pos_w_end"],
            root_quat_w_end=collected["root_quat_w_end"],
            skill_name=skill.name,
            model_path=str(skill.model_path),
            motion_path=str(skill.motion_path),
            end_effector=skill.end_effector,
            root_body=ROOT_BODY_NAME,
            x_threshold=reach_cfg.x_threshold,
            sample_every=reach_cfg.sample_every,
            episodes=reach_cfg.episodes,
            num_envs=env_cfg.scene.num_envs,
        )
        print(f"[INFO] Saved: {out_path} (points={len(collected['points_root'])})")
        torch.cuda.empty_cache()

    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
