# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import copy
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
        if hasattr(actor_critic, "actor") or hasattr(actor_critic, "student"):
            super().__init__(actor_critic, normalizer, verbose)
        else:
            # Fallback for custom policies (e.g. MoEActorCritic) without actor/student attribute.
            torch.nn.Module.__init__(self)
            self.verbose = verbose
            self.is_recurrent = bool(getattr(actor_critic, "is_recurrent", False))
            if self.is_recurrent:
                raise ValueError("ONNX 导出暂不支持自定义 recurrent policy。")
            self.actor = _MoEActorForExport(actor_critic)
            self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def _infer_actor_obs_dim(self) -> int:
        if hasattr(self.actor, "in_features"):
            return int(self.actor.in_features)
        if hasattr(self.actor, "__getitem__"):
            first = self.actor[0]
            if hasattr(first, "in_features"):
                return int(first.in_features)
        raise ValueError("无法推断 actor 输入维度，请在 actor 上提供 in_features。")

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
        obs = torch.zeros(1, self._infer_actor_obs_dim())
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


class _MoEActorForExport(torch.nn.Module):
    """Export-only actor wrapper for MoE policies.

    Inputs:
      actor_obs: flattened policy observation tensor.
    Outputs:
      weighted action mean from frozen skill actors + router.
    """

    def __init__(self, actor_critic: object):
        super().__init__()
        if not hasattr(actor_critic, "skill_actors") or not hasattr(actor_critic, "router"):
            raise ValueError("Policy 缺少 skill_actors/router，无法构建 MoE 导出 actor。")
        self.skill_actors = copy.deepcopy(actor_critic.skill_actors)
        self.router = copy.deepcopy(actor_critic.router)
        self.skill_weight_temperature = float(max(1e-6, getattr(actor_critic, "skill_weight_temperature", 1.0)))
        for p in self.skill_actors.parameters():
            p.requires_grad_(False)
        self.skill_actors.eval()
        self.router.eval()

        # Used by exporter to build dummy input.
        self.in_features = int(self.router[0].in_features)

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            per_skill_actions = [skill_actor(actor_obs) for skill_actor in self.skill_actors]
            # [B, num_skills, num_actions]
            skill_means = torch.stack(per_skill_actions, dim=1)
        logits = self.router(actor_obs) / self.skill_weight_temperature
        weights = torch.softmax(logits, dim=-1)
        return torch.sum(skill_means * weights.unsqueeze(-1), dim=1)


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    robot_data = env.scene["robot"].data
    default_joint_pos = getattr(robot_data, "default_joint_pos_nominal", None)
    if default_joint_pos is None:
        default_joint_pos = getattr(robot_data, "default_joint_pos", None)
    if default_joint_pos is None:
        default_joint_pos = robot_data.joint_pos[0]
    elif default_joint_pos.ndim == 2:
        default_joint_pos = default_joint_pos[0]

    metadata = {
        "run_path": run_path,
        "joint_names": robot_data.joint_names,
        "joint_stiffness": robot_data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": robot_data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": default_joint_pos.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
