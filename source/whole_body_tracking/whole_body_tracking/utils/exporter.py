# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def _safe_get_default_joint_pos(env: ManagerBasedRLEnv):
    robot_data = env.scene["robot"].data
    if hasattr(robot_data, "default_joint_pos_nominal"):
        return robot_data.default_joint_pos_nominal
    if hasattr(robot_data, "default_joint_pos"):
        return robot_data.default_joint_pos
    return None


def _safe_get_action_scale(env: ManagerBasedRLEnv):
    try:
        term = env.action_manager.get_term("joint_pos")
        return term._scale[0]
    except Exception:
        return None


def _safe_get_anchor_body_info(env: ManagerBasedRLEnv):
    # Prefer motion command if present (mimic policy), otherwise fall back to MoE action term.
    try:
        cmd = env.command_manager.get_term("motion")
        return cmd.cfg.anchor_body_name, cmd.cfg.body_names
    except Exception:
        pass
    try:
        act = env.action_manager.get_term("moe")
        return act.cfg.anchor_body_name, act.cfg.body_names
    except Exception:
        return None, None


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    metadata: dict[str, object] = {"run_path": run_path}

    # Observation metadata (optional but useful).
    if hasattr(env, "observation_manager"):
        observation_names = env.observation_manager.active_terms.get("policy", [])
        observation_history_lengths: list[int] = []
        if getattr(env.observation_manager.cfg.policy, "history_length", None) is not None:
            observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
        else:
            for name in observation_names:
                term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
                history_length = term_cfg["history_length"]
                observation_history_lengths.append(1 if history_length == 0 else history_length)
        metadata["observation_names"] = observation_names
        metadata["observation_history_lengths"] = observation_history_lengths

    # Robot metadata.
    robot_data = env.scene["robot"].data
    metadata["joint_names"] = robot_data.joint_names
    metadata["joint_stiffness"] = robot_data.joint_stiffness[0].cpu().tolist()
    metadata["joint_damping"] = robot_data.joint_damping[0].cpu().tolist()
    default_joint_pos = _safe_get_default_joint_pos(env)
    if default_joint_pos is not None:
        metadata["default_joint_pos"] = default_joint_pos.cpu().tolist()

    # Command/action info (may differ across tasks).
    if hasattr(env, "command_manager"):
        metadata["command_names"] = env.command_manager.active_terms
    action_scale = _safe_get_action_scale(env)
    if action_scale is not None:
        metadata["action_scale"] = action_scale.cpu().tolist()
    anchor_body_name, body_names = _safe_get_anchor_body_info(env)
    if anchor_body_name is not None:
        metadata["anchor_body_name"] = anchor_body_name
    if body_names is not None:
        metadata["body_names"] = body_names

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
