"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from dataclasses import MISSING

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--registry_name", type=str, default=None, help="WandB registry artifact name for motion.")
parser.add_argument("--eval_strict", action="store_true", default=False, help="Disable eval-time noise/push randomization.")
# Stage4: MoE (router + 7 frozen experts) play-time overrides.
parser.add_argument("--stage4_moe", action="store_true", default=False, help="Enable Stage4 MoE play mode.")
parser.add_argument(
    "--frozen_model_dir",
    type=str,
    default="basic_model",
    help="Directory containing frozen expert checkpoints for Stage4 MoE.",
)
parser.add_argument(
    "--frozen_motion_dir",
    type=str,
    default="iros_motion/npz",
    help="Directory containing per-expert reference motions (.npz) for Stage4 MoE.",
)
parser.add_argument(
    "--router_hidden_dims",
    type=str,
    default="256,128",
    help="Router MLP hidden dims used to construct MoE policy when agent.yaml is unavailable.",
)
parser.add_argument(
    "--use_grouped_router",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use two-level grouped router for Stage4 MoE play (default: enabled).",
)
parser.add_argument(
    "--grouped_router_stance_init_bias",
    type=float,
    default=2.0,
    help="Initial stance-group bias when reconstructing grouped router without agent.yaml.",
)
parser.add_argument("--target_visible_time_min", type=float, default=0.3, help="Target visible min seconds for Stage4.")
parser.add_argument("--target_visible_time_max", type=float, default=0.8, help="Target visible max seconds for Stage4.")
parser.add_argument("--stage4_episode_length_s", type=float, default=8.0, help="Episode length for Stage4 play.")
parser.add_argument(
    "--play_hit_radius",
    type=float,
    default=None,
    help="Optional fixed hit radius for play visualization/evaluation. If set, disables hit-radius curriculum.",
)
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

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
import yaml

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
from whole_body_tracking.tasks.tracking.stage4.skill_registry import (
    STAGE4_SKILL_METRIC_NAMES,
    resolve_stage4_model_paths,
    resolve_stage4_motion_paths,
)
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner


def _parse_dims(text: str) -> list[int]:
    items = [x.strip() for x in text.split(",") if len(x.strip()) > 0]
    if len(items) == 0:
        raise ValueError(f"Invalid hidden dims: '{text}'")
    return [int(x) for x in items]


def _apply_stage4_play_overrides(env_cfg, args):
    """Apply Stage4 MoE command/env overrides so play matches Stage4 training setup."""
    motion_cfg = env_cfg.commands.motion
    motion_cfg.stage4_expert_motion_files = tuple(resolve_stage4_motion_paths(args.frozen_motion_dir))
    env_cfg.episode_length_s = float(args.stage4_episode_length_s)
    if hasattr(env_cfg.terminations, "motion_completed"):
        env_cfg.terminations.motion_completed = None
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    t_min = float(min(args.target_visible_time_min, args.target_visible_time_max))
    t_max = float(max(args.target_visible_time_min, args.target_visible_time_max))
    motion_cfg.target_visible_time_range_s = (t_min, t_max)
    motion_cfg.enable_router_weight_metrics = True
    motion_cfg.router_metric_names = tuple(STAGE4_SKILL_METRIC_NAMES)
    motion_cfg.enable_tracking_error_metrics = False

    # Keep Stage4 behavior explicit: no shared motion reference required.
    # The actual expert command bank comes from stage4_expert_motion_files.
    motion_cfg.motion_file = ""

    if args.play_hit_radius is not None:
        radius = float(args.play_hit_radius)
        if radius <= 0.0:
            raise ValueError("--play_hit_radius must be > 0.")
        motion_cfg.hit_radius_curriculum_enabled = False
        motion_cfg.hit_distance_threshold = radius
        motion_cfg.hit_radius_start = radius
        motion_cfg.hit_radius_end = radius
        motion_cfg.target_sphere_follow_hit_radius = True
        motion_cfg.guidance_sphere_follow_hit_radius = True
        print(f"[INFO]: Stage4 play fixed hit radius = {radius:.4f}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if args_cli.eval_strict:
        if hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg.observations, "critic"):
            env_cfg.observations.critic.enable_corruption = False
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    motion_file_path = args_cli.motion_file
    if args_cli.registry_name:
        import wandb

        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        artifact = wandb.Api().artifact(registry_name)
        motion_file_path = str(pathlib.Path(artifact.download()) / "motion.npz")
        print(f"[INFO]: Using motion from registry: {registry_name}")

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"
        if motion_file_path is None:
            arts = [a for a in wandb_run.used_artifacts() if "motion" in a.type.lower()]
            art = arts[0] if arts else None
            if art is not None:
                motion_file_path = str(pathlib.Path(art.download()) / "motion.npz")
            else:
                print("[WARN] No motion artifact found in run; keep CLI/config motion file.")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if motion_file_path is not None:
        print(f"[INFO]: Using motion file: {motion_file_path}")
        env_cfg.commands.motion.motion_file = motion_file_path

    if args_cli.stage4_moe:
        _apply_stage4_play_overrides(env_cfg, args_cli)
        print("[INFO]: Stage4 MoE play overrides enabled.")
        print(f"[INFO]: Stage4 frozen expert motions dir: {args_cli.frozen_motion_dir}")
    elif env_cfg.commands.motion.motion_file is MISSING:
        raise ValueError(
            "No motion file configured. Please provide one of: "
            "--motion_file, --registry_name, or a wandb run with used motion artifact."
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    # Prefer run-specific agent config (supports custom Stage4 policy classes).
    runner_cfg_dict = agent_cfg.to_dict()
    loaded_runner_cfg_from_file = False
    agent_cfg_yaml = os.path.join(os.path.dirname(resume_path), "params", "agent.yaml")
    if os.path.exists(agent_cfg_yaml):
        with open(agent_cfg_yaml, "r") as f:
            loaded_cfg = yaml.safe_load(f)
        if isinstance(loaded_cfg, dict):
            runner_cfg_dict = loaded_cfg
            loaded_runner_cfg_from_file = True
            print(f"[INFO]: Loaded runner config from: {agent_cfg_yaml}")

    # For wandb-only checkpoint play, params/agent.yaml may be unavailable.
    # In that case reconstruct Stage4 policy config from CLI so MoE checkpoint can still be loaded.
    if args_cli.stage4_moe:
        policy_cfg = dict(runner_cfg_dict.get("policy", {}))
        if not loaded_runner_cfg_from_file:
            policy_cfg["class_name"] = "whole_body_tracking.learning.moe_actor_critic.MoEActorCritic"
            policy_cfg["frozen_skill_ckpts"] = resolve_stage4_model_paths(args_cli.frozen_model_dir)
            policy_cfg["router_hidden_dims"] = _parse_dims(args_cli.router_hidden_dims)
            policy_cfg["use_grouped_router"] = bool(args_cli.use_grouped_router)
            policy_cfg["grouped_router_stance_init_bias"] = float(args_cli.grouped_router_stance_init_bias)
            print("[INFO]: Stage4 MoE policy config reconstructed from CLI.")
        else:
            # Keep run config as source-of-truth to avoid checkpoint mismatch.
            # Only fill optional missing keys for backward compatibility.
            policy_cfg.setdefault("class_name", "whole_body_tracking.learning.moe_actor_critic.MoEActorCritic")
            policy_cfg.setdefault("frozen_skill_ckpts", resolve_stage4_model_paths(args_cli.frozen_model_dir))
            policy_cfg.setdefault("use_grouped_router", bool(args_cli.use_grouped_router))
            policy_cfg.setdefault("grouped_router_stance_init_bias", float(args_cli.grouped_router_stance_init_bias))
            print("[INFO]: Stage4 MoE policy config loaded from run's agent.yaml.")
        runner_cfg_dict["policy"] = policy_cfg
        print("[INFO]: Stage4 MoE policy config applied.")
        print(f"[INFO]: Stage4 frozen expert models dir: {args_cli.frozen_model_dir}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # 已经修复
    try:
        export_motion_policy_as_onnx(
            env.unwrapped,
            ppo_runner.alg.policy,
            normalizer=getattr(ppo_runner, "obs_normalizer", None),
            path=export_model_dir,
            filename="policy.onnx",
        )
        attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    except Exception as exc:
        print(f"[WARN] ONNX export skipped during play: {exc}")
    # reset environment
    obs = env.get_observations() #已经修复
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
