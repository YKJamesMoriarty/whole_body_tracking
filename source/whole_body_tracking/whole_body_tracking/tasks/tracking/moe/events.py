from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from whole_body_tracking.tasks.tracking.mdp.commands import MotionLoader
from .experts import EXPERTS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_stance_spec():
    for spec in EXPERTS:
        if spec.name == "stance":
            return spec
    raise ValueError("Stance expert not found in EXPERTS")


def reset_to_stance_motion(env: "ManagerBasedEnv", env_ids: torch.Tensor):
    """Reset robot state to the first frame of the stance motion."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return

    if not hasattr(env, "_moe_stance_loader"):
        stance_spec = _get_stance_spec()
        motion_path = Path(env.cfg.actions.moe.motion_dir) / stance_spec.motion_file
        if not motion_path.is_absolute():
            motion_path = Path.cwd() / motion_path
        if not motion_path.exists():
            raise FileNotFoundError(f"Stance motion not found: {motion_path}")

        robot = env.scene["robot"]
        body_names = list(env.cfg.actions.moe.body_names)
        body_indexes = torch.tensor(
            robot.find_bodies(body_names, preserve_order=True)[0], dtype=torch.long, device=env.device
        )

        env._moe_stance_loader = MotionLoader(str(motion_path), body_indexes, device=env.device)
        env._moe_stance_robot = robot

    motion = env._moe_stance_loader
    robot = env._moe_stance_robot

    root_pos = motion.body_pos_w[0, 0]
    root_quat = motion.body_quat_w[0, 0]
    root_lin_vel = motion.body_lin_vel_w[0, 0]
    root_ang_vel = motion.body_ang_vel_w[0, 0]

    root_pos = root_pos.unsqueeze(0) + env.scene.env_origins[env_ids]
    root_quat = root_quat.unsqueeze(0).repeat(env_ids.numel(), 1)
    root_lin_vel = root_lin_vel.unsqueeze(0).repeat(env_ids.numel(), 1)
    root_ang_vel = root_ang_vel.unsqueeze(0).repeat(env_ids.numel(), 1)

    joint_pos = motion.joint_pos[0].unsqueeze(0).repeat(env_ids.numel(), 1)
    joint_vel = motion.joint_vel[0].unsqueeze(0).repeat(env_ids.numel(), 1)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.write_root_state_to_sim(
        torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1),
        env_ids=env_ids,
    )
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
