# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")
# Stage 2: 支持从 WandB 加载 checkpoint 继续训练
parser.add_argument("--wandb_checkpoint_path", type=str, default=None, 
                    help="WandB run path to load checkpoint from (e.g., 'org/project/run_id' or 'org/project/run_id/model_10000.pt')")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.io import dump_pickle, dump_yaml
# from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# 修复importError: cannot import name 'dump_pickle' from 'isaaclab.utils.io'
import pickle
import yaml
import os
def dump_pickle(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def dump_yaml(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
# === 修复结束 ===

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # --- [新增] 修复 ModuleNotFoundError: No module named 'isaacsim.asset' ---
    from isaacsim.core.utils.extensions import enable_extension
    try:
        # 尝试加载 Isaac Sim 4.0+ 的新版 URDF 插件
        enable_extension("isaacsim.asset.importer.urdf")
    except Exception:
        # 如果失败，尝试旧版名称
        enable_extension("omni.isaac.urdf_importer")
    # -----------------------------------------------------------------------
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # load the motion file from the wandb registry
    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"
    import pathlib

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    
    # =========================================================================
    # Stage 2: 支持从 WandB 或本地加载 checkpoint 继续训练
    # =========================================================================
    
    # 方式 1: 从 WandB 下载 checkpoint (优先级最高)
    if args_cli.wandb_checkpoint_path:
        import wandb as wandb_api
        
        run_path = args_cli.wandb_checkpoint_path
        api = wandb_api.Api()
        
        # 检查是否指定了具体的 model 文件
        if "model" in args_cli.wandb_checkpoint_path:
            run_path = "/".join(args_cli.wandb_checkpoint_path.split("/")[:-1])
            specified_file = args_cli.wandb_checkpoint_path.split("/")[-1]
        else:
            specified_file = None
        
        wandb_run = api.run(run_path)
        
        # 获取所有 model 文件
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        
        if specified_file:
            file = specified_file
        else:
            # 找最新的 model 文件 (model_xxx.pt 中 xxx 最大的)
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        # 下载到临时目录
        wandb_file = wandb_run.file(str(file))
        wandb_download_dir = "./logs/rsl_rl/wandb_checkpoints"
        wandb_file.download(wandb_download_dir, replace=True)
        
        resume_path = os.path.join(wandb_download_dir, file)
        print(f"[INFO]: Loading model checkpoint from WandB: {run_path}/{file}")
        print(f"[INFO]: Downloaded to: {resume_path}")
        
        # 加载 checkpoint
        runner.load(resume_path)
        print(f"[INFO]: Successfully loaded Stage 1 checkpoint, continuing Stage 2 training...")
    
    # 方式 2: 从本地 logs 文件夹恢复 (原有逻辑)
    elif agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
