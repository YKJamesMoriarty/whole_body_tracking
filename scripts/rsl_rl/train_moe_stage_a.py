# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train MoE policy (Stage A) with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Train MoE gating policy (Stage A) with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-MoE-StageA-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--model_dir", type=str, default=None, help="Directory containing expert .pt models.")
parser.add_argument("--motion_dir", type=str, default=None, help="Directory containing expert motion .npz files.")
parser.add_argument("--targets_file", type=str, default=None, help="Path to moe_targets.npz.")
parser.add_argument("--temperature", type=float, default=None, help="Softmax temperature for gating.")
parser.add_argument("--entropy_lambda", type=float, default=None, help="Entropy penalty coefficient (lambda).")
parser.add_argument("--hit_radius", type=float, default=None, help="Hit radius in meters.")
parser.add_argument("--progress_weight", type=float, default=None, help="Weight for progress reward.")
parser.add_argument("--alive_weight", type=float, default=None, help="Weight for alive reward.")
parser.add_argument("--skill_match_weight", type=float, default=None, help="Weight for skill-match reward.")
parser.add_argument("--lock_skill", type=int, default=None, help="Lock one skill per episode (1/0).")
parser.add_argument(
    "--success_split_threshold",
    type=float,
    default=None,
    help="Fixed success threshold to separate low/high success points.",
)
parser.add_argument(
    "--hit_rate_window",
    type=int,
    default=None,
    help="Sliding window size (episodes) for hit-rate metric.",
)
parser.add_argument(
    "--sample_ratio_ema",
    type=float,
    default=None,
    help="EMA factor for target sampling ratio metrics (0 disables EMA).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# default to wandb logging for MoE training (can be overridden via --logger)
if args_cli.logger is None:
    args_cli.logger = "wandb"

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ================= 修复 ModuleNotFoundError: No module named 'isaacsim.asset' =================
# 必须在 app 启动后，import 其他 Isaac Lab 模块之前手动加载
from isaacsim.core.utils.extensions import enable_extension
try:
    enable_extension("isaacsim.asset.importer.urdf")
except Exception:
    enable_extension("omni.isaac.urdf_importer")
# ===========================================================================================

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
from datetime import datetime

from whole_body_tracking.utils.my_on_policy_runner import MyOnPolicyRunner as OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.model_dir is not None:
        env_cfg.actions.moe.model_dir = args_cli.model_dir
    if args_cli.motion_dir is not None:
        env_cfg.actions.moe.motion_dir = args_cli.motion_dir
    if args_cli.targets_file is not None:
        env_cfg.commands.attack_target.target_file = args_cli.targets_file
    if args_cli.temperature is not None:
        env_cfg.actions.moe.temperature = args_cli.temperature
    if args_cli.entropy_lambda is not None:
        env_cfg.rewards.entropy.weight = -abs(args_cli.entropy_lambda)
    if args_cli.hit_radius is not None:
        if hasattr(env_cfg.rewards, "hit_stage1"):
            env_cfg.rewards.hit_stage1.params["hit_radius"] = args_cli.hit_radius
        env_cfg.terminations.hit_target.params["hit_radius"] = args_cli.hit_radius
        env_cfg.commands.attack_target.visual_radius = args_cli.hit_radius
    if args_cli.progress_weight is not None:
        if hasattr(env_cfg.rewards, "progress_stage1"):
            env_cfg.rewards.progress_stage1.weight = args_cli.progress_weight
    if args_cli.alive_weight is not None:
        env_cfg.rewards.alive.weight = args_cli.alive_weight
    if args_cli.skill_match_weight is not None:
        if hasattr(env_cfg.rewards, "skill_match_stage1"):
            env_cfg.rewards.skill_match_stage1.weight = args_cli.skill_match_weight
        elif hasattr(env_cfg.rewards, "skill_match"):
            env_cfg.rewards.skill_match.weight = args_cli.skill_match_weight
    if args_cli.lock_skill is not None:
        env_cfg.actions.moe.lock_skill_per_episode = bool(args_cli.lock_skill)
    if args_cli.success_split_threshold is not None:
        env_cfg.commands.attack_target.success_split_threshold = args_cli.success_split_threshold
    if args_cli.hit_rate_window is not None:
        env_cfg.commands.attack_target.hit_rate_window = args_cli.hit_rate_window
    if args_cli.sample_ratio_ema is not None:
        env_cfg.commands.attack_target.sample_ratio_ema = args_cli.sample_ratio_ema

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    train_cfg = agent_cfg.to_dict()
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    def _resolve_resume_path_from_wandb_or_local(wandb_path: str) -> str:
        run_path = wandb_path
        local_path = pathlib.Path(run_path)
        if local_path.exists():
            if local_path.is_file():
                print(f"[INFO]: Loading local model checkpoint: {local_path}")
                return str(local_path)
            candidates = list(local_path.glob("model_*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No model_*.pt found in directory: {local_path}")
            candidates.sort(key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else -1)
            resume = candidates[-1]
            print(f"[INFO]: Loading local model checkpoint: {resume}")
            return str(resume)

        import wandb

        api = wandb.Api()
        if "model" in wandb_path:
            run_path = "/".join(wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        if not files:
            raise FileNotFoundError(f"No model checkpoint found in wandb run: {run_path}")

        if "model" in wandb_path:
            file = wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        temp_dir = pathlib.Path("./logs/rsl_rl/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        wandb_file.download(str(temp_dir), replace=True)

        resume = temp_dir / file
        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        return str(resume)

    if agent_cfg.resume:
        if args_cli.wandb_path:
            resume_path = _resolve_resume_path_from_wandb_or_local(args_cli.wandb_path)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    import pickle
    import yaml

    def _dump_pickle(filename, data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def _dump_yaml(filename, data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            yaml.dump(data, f, sort_keys=False)

    _dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    _dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    _dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    _dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
