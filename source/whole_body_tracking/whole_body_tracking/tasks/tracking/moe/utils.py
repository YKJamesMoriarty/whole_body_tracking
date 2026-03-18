from __future__ import annotations

import torch
from isaaclab.utils.math import subtract_frame_transforms


DEFAULT_EEF_NAMES = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]


def compute_min_eef_distance(
    env,
    target_pos_b: torch.Tensor,
    body_names: list[str] | None = None,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Compute the minimum end-effector distance to target in pelvis frame.

    Args:
        env: ManagerBasedRLEnv.
        target_pos_b: Target positions in pelvis frame. Shape (num_envs, 3).
        body_names: End-effector body names. Defaults to wrists and ankles.
        asset_name: Name of the robot asset.

    Returns:
        Tensor of shape (num_envs,) with min distance per env.
    """
    robot = env.scene[asset_name]
    eef_names = body_names or DEFAULT_EEF_NAMES

    pelvis_idx = robot.body_names.index("pelvis")
    eef_ids, _ = robot.find_bodies(eef_names, preserve_order=True)

    pelvis_pos_w = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_quat_w = robot.data.body_quat_w[:, pelvis_idx]

    eef_pos_w = robot.data.body_pos_w[:, eef_ids]
    eef_quat_w = robot.data.body_quat_w[:, eef_ids]

    eef_pos_b, _ = subtract_frame_transforms(
        pelvis_pos_w[:, None, :].repeat(1, len(eef_ids), 1),
        pelvis_quat_w[:, None, :].repeat(1, len(eef_ids), 1),
        eef_pos_w,
        eef_quat_w,
    )

    diff = eef_pos_b - target_pos_b[:, None, :]
    dist = torch.norm(diff, dim=-1)
    min_dist, _ = dist.min(dim=1)
    return min_dist
