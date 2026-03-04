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
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Local .npz motion file path (fallback if no registry).")
# Stage4: Frozen experts + trainable router (MoE)
parser.add_argument("--stage4_moe", action="store_true", default=False, help="Enable Stage4 MoE router training mode.")
parser.add_argument(
    "--frozen_model_dir",
    type=str,
    default="basic_model",
    help="Directory containing frozen skill checkpoints for Stage4.",
)
parser.add_argument(
    "--frozen_motion_dir",
    type=str,
    default="iros_motion/npz",
    help="Directory containing per-skill reference motions (.npz) for Stage4 frozen experts.",
)
parser.add_argument(
    "--router_hidden_dims",
    type=str,
    default="256,128",
    help="Router MLP hidden dims, comma-separated (e.g., 256,128).",
)
parser.add_argument("--router_init_noise_std", type=float, default=0.35, help="Initial action noise std for router policy.")
parser.add_argument("--hit_radius_start", type=float, default=0.30, help="Stage4 hit radius curriculum start.")
parser.add_argument("--hit_radius_end", type=float, default=0.06, help="Stage4 hit radius curriculum end.")
parser.add_argument("--hit_curriculum_window", type=int, default=2000, help="Episode window size for hit curriculum.")
parser.add_argument(
    "--hit_curriculum_success_threshold",
    type=float,
    default=0.60,
    help="Hit success threshold to shrink hit radius.",
)
parser.add_argument("--hit_radius_shrink_factor", type=float, default=0.98, help="Hit radius multiplicative shrink step.")
parser.add_argument("--target_visible_time_min", type=float, default=0.3, help="Target visible min seconds.")
parser.add_argument("--target_visible_time_max", type=float, default=0.8, help="Target visible max seconds.")
parser.add_argument("--stage4_episode_length_s", type=float, default=8.0, help="Episode length for Stage4.")
parser.add_argument(
    "--stage4_alive_bonus_weight",
    type=float,
    default=None,
    help="Stage4 survival bonus reward weight.",
)
parser.add_argument(
    "--stage4_posture_penalty_weight",
    type=float,
    default=None,
    help="Stage4 torso tilt penalty weight.",
)
parser.add_argument(
    "--stage4_posture_tilt_threshold",
    type=float,
    default=None,
    help="No posture penalty above this cos(up, world_up) threshold.",
)
parser.add_argument(
    "--stage4_posture_full_penalty_tilt",
    type=float,
    default=None,
    help="Posture penalty saturates near this cos(up, world_up) level.",
)
parser.add_argument(
    "--stage4_posture_penalty_exponent",
    type=float,
    default=None,
    help="Shape exponent for posture penalty curve.",
)
parser.add_argument(
    "--stage4_roll_pitch_rate_penalty_weight",
    type=float,
    default=None,
    help="Stage4 roll/pitch angular-rate penalty weight.",
)
parser.add_argument(
    "--stage4_roll_pitch_rate_deadband",
    type=float,
    default=None,
    help="No roll/pitch rate penalty below this rad/s deadband.",
)
parser.add_argument(
    "--enable_amp_reward",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable AMP style reward in Stage4 (default: enabled). Use --no-enable_amp_reward to disable.",
)
parser.add_argument(
    "--amp_disc_bundle_path",
    type=str,
    default="basic_model/amp_boxing_disc_bundle.pt",
    help="Path to exported AMP discriminator bundle.",
)
parser.add_argument("--amp_disc_activation", type=str, default="elu", help="Discriminator activation: elu/relu/leaky_relu.")
parser.add_argument(
    "--amp_disc_obs_mode",
    type=str,
    default="mimickit_like",
    choices=["legacy_simple", "mimickit_like"],
    help="AMP disc_obs construction mode.",
)
parser.add_argument("--amp_obs_history_steps", type=int, default=10, help="History steps used for runtime AMP obs.")
parser.add_argument(
    "--amp_allow_obs_dim_mismatch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Allow AMP obs dim pad/trim fallback when bundle dim mismatches runtime obs.",
)
parser.add_argument(
    "--amp_reward_weight",
    type=float,
    default=None,
    help="Reward weight for AMP style term. Default: keep tracking_env_cfg.py value.",
)
parser.add_argument(
    "--amp_reward_scale",
    type=float,
    default=None,
    help="Internal scale in AMP style reward. Default: keep tracking_env_cfg.py value.",
)
parser.add_argument(
    "--enable_router_diversity_reward",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable anti-collapse router diversity reward (default: enabled). Use --no-enable_router_diversity_reward to disable.",
)
parser.add_argument(
    "--router_diversity_weight",
    type=float,
    default=None,
    help="Reward weight for router diversity term. Default: keep tracking_env_cfg.py value.",
)
parser.add_argument(
    "--router_diversity_min_entropy",
    type=float,
    default=None,
    help="Entropy floor for router diversity reward (normalized 0~1). Default: keep tracking_env_cfg.py value.",
)
parser.add_argument(
    "--router_diversity_load_balance_coef",
    type=float,
    default=None,
    help="Global load-balance coefficient for router diversity reward. Default: keep tracking_env_cfg.py value.",
)
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
import pathlib
import copy

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
from whole_body_tracking.tasks.tracking.stage4.skill_registry import (
    STAGE4_SKILL_METRIC_NAMES,
    resolve_stage4_model_paths,
    resolve_stage4_motion_paths,
)
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# 修复importError: cannot import name 'dump_pickle' from 'isaaclab.utils.io'
import pickle
import yaml


def dump_pickle(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def dump_yaml(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=False)


# === 修复结束 ===


def _parse_dims(text: str) -> list[int]:
    items = [x.strip() for x in text.split(",") if len(x.strip()) > 0]
    if len(items) == 0:
        raise ValueError(f"无效 hidden dims: '{text}'")
    return [int(x) for x in items]


def _resolve_motion_file(
    registry_name: str | None,
    local_motion_file: str | None,
    require_motion_file: bool = True,
) -> tuple[str, str | None]:
    """Resolve motion source from WandB registry or local file."""
    if registry_name is not None:
        if ":" not in registry_name:
            registry_name = f"{registry_name}:latest"
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        return motion_file, registry_name

    if local_motion_file is not None:
        motion_path = pathlib.Path(local_motion_file).expanduser().resolve()
        if not motion_path.exists():
            raise FileNotFoundError(f"--motion_file 不存在: {motion_path}")
        return str(motion_path), None

    if not require_motion_file:
        return "", None

    raise ValueError("必须提供 --registry_name 或 --motion_file 之一。")


def _print_active_stage4_rewards(env_cfg):
    """Print active non-zero reward terms after Stage4 override for quick sanity check."""
    reward_cfg = env_cfg.rewards
    print("[INFO] Stage4 active rewards (non-zero):")
    for term_name in reward_cfg.__dataclass_fields__.keys():
        term_cfg = getattr(reward_cfg, term_name)
        if term_cfg is None:
            continue
        if abs(float(term_cfg.weight)) < 1e-8:
            continue
        print(f"  - {term_name}: weight={float(term_cfg.weight):.4f}")


def _apply_stage4_overrides(env_cfg, args):
    """Apply Stage4 task overrides on top of base tracking config."""
    motion_cfg = env_cfg.commands.motion
    # Stage4 command no longer uses a shared single motion reference.
    # Frozen experts consume per-skill references via stage4_expert_motion_files.
    motion_cfg.stage4_expert_motion_files = tuple(resolve_stage4_motion_paths(args.frozen_motion_dir))

    # Episode design: no motion-length termination, fixed short episode.
    env_cfg.episode_length_s = float(args.stage4_episode_length_s)
    env_cfg.terminations.motion_completed = None

    # Keep short target visibility from stage2 behavior.
    t_min = float(min(args.target_visible_time_min, args.target_visible_time_max))
    t_max = float(max(args.target_visible_time_min, args.target_visible_time_max))
    motion_cfg.target_visible_time_range_s = (t_min, t_max)

    # Curriculum on hit radius.
    motion_cfg.hit_radius_curriculum_enabled = True
    motion_cfg.hit_radius_start = float(args.hit_radius_start)
    motion_cfg.hit_radius_end = float(args.hit_radius_end)
    motion_cfg.hit_curriculum_window = int(args.hit_curriculum_window)
    motion_cfg.hit_curriculum_success_threshold = float(args.hit_curriculum_success_threshold)
    motion_cfg.hit_radius_shrink_factor = float(args.hit_radius_shrink_factor)
    motion_cfg.hit_distance_threshold = float(args.hit_radius_start)
    motion_cfg.target_sphere_follow_hit_radius = True
    motion_cfg.guidance_sphere_follow_hit_radius = True
    motion_cfg.enable_router_weight_metrics = True
    motion_cfg.router_metric_names = tuple(STAGE4_SKILL_METRIC_NAMES)
    motion_cfg.enable_tracking_error_metrics = False

    # Stage4 stability shaping.
    posture_tilt_threshold = args.stage4_posture_tilt_threshold
    posture_full_penalty_tilt = args.stage4_posture_full_penalty_tilt
    if (posture_tilt_threshold is not None) and (posture_full_penalty_tilt is not None):
        if posture_full_penalty_tilt >= posture_tilt_threshold:
            raise ValueError("--stage4_posture_full_penalty_tilt 必须小于 --stage4_posture_tilt_threshold。")

    if hasattr(env_cfg.rewards, "alive_bonus") and (args.stage4_alive_bonus_weight is not None):
        env_cfg.rewards.alive_bonus.weight = float(args.stage4_alive_bonus_weight)
    if hasattr(env_cfg.rewards, "posture_unstable"):
        if args.stage4_posture_penalty_weight is not None:
            env_cfg.rewards.posture_unstable.weight = float(args.stage4_posture_penalty_weight)
        if posture_tilt_threshold is not None:
            env_cfg.rewards.posture_unstable.params["tilt_threshold"] = float(posture_tilt_threshold)
        if posture_full_penalty_tilt is not None:
            env_cfg.rewards.posture_unstable.params["full_penalty_tilt"] = float(posture_full_penalty_tilt)
        if args.stage4_posture_penalty_exponent is not None:
            env_cfg.rewards.posture_unstable.params["penalty_exponent"] = float(args.stage4_posture_penalty_exponent)
    if hasattr(env_cfg.rewards, "root_roll_pitch_rate"):
        if args.stage4_roll_pitch_rate_penalty_weight is not None:
            env_cfg.rewards.root_roll_pitch_rate.weight = float(args.stage4_roll_pitch_rate_penalty_weight)
        if args.stage4_roll_pitch_rate_deadband is not None:
            env_cfg.rewards.root_roll_pitch_rate.params["deadband"] = float(args.stage4_roll_pitch_rate_deadband)

    # AMP style reward (optional, switchable for ablation).
    motion_cfg.amp_reward_enabled = bool(args.enable_amp_reward)
    motion_cfg.amp_disc_activation = args.amp_disc_activation
    motion_cfg.amp_disc_obs_mode = args.amp_disc_obs_mode
    motion_cfg.amp_obs_history_steps = int(args.amp_obs_history_steps)
    motion_cfg.amp_allow_obs_dim_mismatch = bool(args.amp_allow_obs_dim_mismatch)
    if args.amp_reward_scale is not None:
        motion_cfg.amp_disc_reward_scale = float(args.amp_reward_scale)
    if args.enable_amp_reward:
        if len(args.amp_disc_bundle_path) == 0:
            raise ValueError("开启 --enable_amp_reward 时必须提供 --amp_disc_bundle_path。")
        amp_bundle_path = pathlib.Path(args.amp_disc_bundle_path).expanduser().resolve()
        if not amp_bundle_path.exists():
            raise FileNotFoundError(f"AMP bundle 不存在: {amp_bundle_path}")
        motion_cfg.amp_disc_bundle_path = str(amp_bundle_path)
    else:
        motion_cfg.amp_disc_bundle_path = ""

    if hasattr(env_cfg.rewards, "amp_style_reward"):
        if args.enable_amp_reward:
            if args.amp_reward_weight is not None:
                env_cfg.rewards.amp_style_reward.weight = float(args.amp_reward_weight)
        else:
            env_cfg.rewards.amp_style_reward = None

    # Anti-collapse router diversity reward (optional, switchable for ablation).
    if hasattr(env_cfg.rewards, "router_diversity"):
        if args.enable_router_diversity_reward:
            if args.router_diversity_weight is not None:
                env_cfg.rewards.router_diversity.weight = float(args.router_diversity_weight)
            if args.router_diversity_min_entropy is not None:
                env_cfg.rewards.router_diversity.params["min_entropy"] = float(args.router_diversity_min_entropy)
            if args.router_diversity_load_balance_coef is not None:
                env_cfg.rewards.router_diversity.params["load_balance_coef"] = float(
                    args.router_diversity_load_balance_coef
                )
        else:
            env_cfg.rewards.router_diversity = None

    _print_active_stage4_rewards(env_cfg)

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

    # load motion from registry or local file
    require_motion_file = not args_cli.stage4_moe
    motion_file, registry_name = _resolve_motion_file(
        args_cli.registry_name,
        args_cli.motion_file,
        require_motion_file=require_motion_file,
    )
    env_cfg.commands.motion.motion_file = motion_file

    # Stage4 overrides before env creation
    if args_cli.stage4_moe:
        _apply_stage4_overrides(env_cfg, args_cli)

    # build mutable runner cfg dict (used for both default and Stage4)
    runner_cfg_dict = copy.deepcopy(agent_cfg.to_dict())
    if args_cli.stage4_moe:
        frozen_ckpts = resolve_stage4_model_paths(args_cli.frozen_model_dir)
        policy_cfg = dict(runner_cfg_dict.get("policy", {}))
        policy_cfg["class_name"] = "whole_body_tracking.learning.moe_actor_critic.MoEActorCritic"
        policy_cfg["frozen_skill_ckpts"] = frozen_ckpts
        policy_cfg["router_hidden_dims"] = _parse_dims(args_cli.router_hidden_dims)
        policy_cfg["init_noise_std"] = float(args_cli.router_init_noise_std)
        runner_cfg_dict["policy"] = policy_cfg
        print("[INFO] Stage4 MoE enabled.")
        print(f"[INFO] Stage4 frozen expert motions: {args_cli.frozen_motion_dir}")
        print(f"[INFO] Frozen experts: {len(frozen_ckpts)}")
        print(f"[INFO] AMP reward enabled: {args_cli.enable_amp_reward}")
        print(f"[INFO] Router diversity reward enabled: {args_cli.enable_router_diversity_reward}")

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
        env, runner_cfg_dict, log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
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
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), runner_cfg_dict)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), runner_cfg_dict)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
