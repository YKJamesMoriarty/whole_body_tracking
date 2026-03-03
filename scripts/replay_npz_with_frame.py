"""Replay a motion npz in Isaac Sim and print current frame index.

Examples:
    python scripts/replay_npz_with_frame.py --motion_file /path/to/motion.npz
    python scripts/replay_npz_with_frame.py --registry_name your-org/wandb-registry-Motions/foo:v0
"""

import argparse
import pathlib

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion and print frame index.")
parser.add_argument("--motion_file", type=str, default=None, help="Local path to motion.npz.")
parser.add_argument("--registry_name", type=str, default=None, help="WandB registry artifact path.")
parser.add_argument("--start_frame", type=int, default=0, help="Start frame index.")
parser.add_argument("--print_every", type=int, default=1, help="Print frame every N simulation steps.")
parser.add_argument(
    "--stop_after_cycle",
    action="store_true",
    default=False,
    help="Stop automatically after one full cycle of the motion.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if (args_cli.motion_file is None) == (args_cli.registry_name is None):
    raise ValueError("Provide exactly one of --motion_file or --registry_name.")
if args_cli.print_every < 1:
    raise ValueError("--print_every must be >= 1.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for replay scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _resolve_motion_file() -> str:
    if args_cli.motion_file is not None:
        motion_file = pathlib.Path(args_cli.motion_file).expanduser().resolve()
        if not motion_file.is_file():
            raise FileNotFoundError(f"Invalid --motion_file: {motion_file}")
        return str(motion_file)

    registry_name = args_cli.registry_name
    if ":" not in registry_name:
        registry_name += ":latest"

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    return str(pathlib.Path(artifact.download()) / "motion.npz")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion_file = _resolve_motion_file()
    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )

    total_frames = int(motion.time_step_total)
    if total_frames <= 0:
        raise ValueError(f"Invalid motion with zero frame count: {motion_file}")

    fps_value = float(np.array(motion.fps).reshape(-1)[0])
    start_frame = int(np.clip(args_cli.start_frame, 0, total_frames - 1))
    time_steps = torch.full((scene.num_envs,), start_frame, dtype=torch.long, device=sim.device)
    cycle_count = 0
    last_frame = start_frame

    print(
        f"[INFO] motion_file={motion_file}, frames={total_frames}, fps={fps_value:.3f}, "
        f"duration={(total_frames - 1) / fps_value:.3f}s, start_frame={start_frame}"
    )
    print("[INFO] Press Ctrl+C to stop and record current frame.")

    try:
        while simulation_app.is_running():
            frame_steps = time_steps.clone()
            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion.body_pos_w[frame_steps][:, 0] + scene.env_origins[:, None, :]
            root_states[:, 3:7] = motion.body_quat_w[frame_steps][:, 0]
            root_states[:, 7:10] = motion.body_lin_vel_w[frame_steps][:, 0]
            root_states[:, 10:] = motion.body_ang_vel_w[frame_steps][:, 0]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(motion.joint_pos[frame_steps], motion.joint_vel[frame_steps])
            scene.write_data_to_sim()
            sim.render()
            scene.update(sim_dt)

            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

            last_frame = int(frame_steps[0].item())
            if last_frame % args_cli.print_every == 0:
                current_time = last_frame / fps_value
                print(
                    f"\rframe={last_frame}/{total_frames - 1} cycle={cycle_count} t={current_time:.3f}s",
                    end="",
                    flush=True,
                )

            time_steps += 1
            reset_ids = time_steps >= total_frames
            if torch.any(reset_ids):
                time_steps[reset_ids] = 0
                cycle_count += int(torch.sum(reset_ids).item())
                if args_cli.stop_after_cycle and cycle_count >= 1:
                    break
    except KeyboardInterrupt:
        pass

    print()
    return last_frame, total_frames, fps_value


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    last_frame = None
    total_frames = None
    fps_value = None
    last_frame, total_frames, fps_value = run_simulator(sim, scene)
    print(
        f"[INFO] Last displayed frame: {last_frame}/{total_frames - 1}, "
        f"time={last_frame / fps_value:.3f}s"
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
