from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import torch
from tensordict import TensorDict

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from rsl_rl.modules import ActorCritic

from whole_body_tracking.tasks.tracking.mdp.commands import MotionLoader

from .experts import filter_experts


def _policy_obs_dim(num_joints: int) -> int:
    return 2 * num_joints + 3 + 6 + 3 + 3 + num_joints + num_joints + num_joints


def _critic_obs_dim(num_joints: int, num_bodies: int) -> int:
    return 2 * num_joints + 3 + 6 + num_bodies * 3 + num_bodies * 6 + 3 + 3 + num_joints + num_joints + num_joints


class _ExpertRuntime:
    def __init__(
        self,
        name: str,
        model_path: Path,
        motion_path: Path,
        env,
        body_names: list[str],
        anchor_body_name: str,
        device: str,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str,
    ) -> None:
        self.name = name
        self.device = device
        self.env = env
        self.robot = env.scene["robot"]

        self.body_names = list(body_names)
        self.anchor_body_name = anchor_body_name

        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.body_names, preserve_order=True)[0], dtype=torch.long, device=device
        )
        self.robot_anchor_body_index = self.robot.body_names.index(self.anchor_body_name)
        self.motion_anchor_body_index = self.body_names.index(self.anchor_body_name)

        self.motion = MotionLoader(str(motion_path), self.body_indexes, device=device)
        self.time_steps = torch.zeros(self.env.num_envs, dtype=torch.long, device=device)

        self.policy = self._load_policy(
            model_path,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        )

    def _load_policy(
        self,
        model_path: Path,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str,
    ) -> ActorCritic:
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("actor_critic", checkpoint)

        has_actor_norm = any("actor_obs_normalizer" in k for k in state_dict.keys())
        has_critic_norm = any("critic_obs_normalizer" in k for k in state_dict.keys())

        num_joints = self.robot.num_joints
        num_bodies = len(self.body_names)
        obs_td = TensorDict(
            {
                "policy": torch.zeros(1, _policy_obs_dim(num_joints), device=self.device),
                "critic": torch.zeros(1, _critic_obs_dim(num_joints, num_bodies), device=self.device),
            },
            batch_size=[1],
        )
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}

        policy = ActorCritic(
            obs=obs_td,
            obs_groups=obs_groups,
            num_actions=num_joints,
            actor_obs_normalization=has_actor_norm,
            critic_obs_normalization=has_critic_norm,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=1.0,
            noise_std_type="scalar",
            state_dependent_std=False,
        ).to(self.device)
        policy.load_state_dict(state_dict, strict=True)
        policy.eval()
        for param in policy.parameters():
            param.requires_grad_(False)
        return policy

    def reset(self, env_ids):
        if env_ids is None:
            self.time_steps.zero_()
        else:
            self.time_steps[env_ids] = 0

    def advance(self):
        self.time_steps += 1
        overflow = self.time_steps >= self.motion.time_step_total
        if torch.any(overflow):
            self.time_steps[overflow] = 0


class MoEActionTerm(ActionTerm):
    cfg: "MoEActionCfg"

    def __init__(self, cfg: "MoEActionCfg", env) -> None:
        super().__init__(cfg, env)

        self._device = env.device
        self._temperature = cfg.temperature
        self._hard_gate = cfg.hard_gate
        self._lock_skill_per_episode = cfg.lock_skill_per_episode

        # underlying joint-position action
        joint_cfg = JointPositionActionCfg(
            asset_name=cfg.asset_name,
            joint_names=cfg.joint_names,
            scale=cfg.scale,
            use_default_offset=cfg.use_default_offset,
        )
        self._joint_action = JointPositionAction(joint_cfg, env)

        self._joint_ids = self._joint_action._joint_ids  # slice or tensor
        self._num_joints = self._joint_action.action_dim

        experts = filter_experts(cfg.expert_names)
        self._experts: list[_ExpertRuntime] = []
        for expert in experts:
            model_path = Path(cfg.model_dir) / expert.model_file
            motion_path = Path(cfg.motion_dir) / expert.motion_file
            if not model_path.exists():
                raise FileNotFoundError(f"Expert model not found: {model_path}")
            if not motion_path.exists():
                raise FileNotFoundError(f"Expert motion not found: {motion_path}")
            self._experts.append(
                _ExpertRuntime(
                    name=expert.name,
                    model_path=model_path,
                    motion_path=motion_path,
                    env=env,
                    body_names=cfg.body_names,
                    anchor_body_name=cfg.anchor_body_name,
                    device=self._device,
                    actor_hidden_dims=list(cfg.actor_hidden_dims),
                    critic_hidden_dims=list(cfg.critic_hidden_dims),
                    activation=cfg.activation,
                )
            )

        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self._device)
        self._processed_actions = torch.zeros(env.num_envs, self._num_joints, device=self._device)
        self._prev_combined_action = torch.zeros(env.num_envs, self._num_joints, device=self._device)

        # initialize per-env tracking for gating
        if not hasattr(env, "_moe_time_since_switch"):
            env._moe_time_since_switch = torch.zeros(env.num_envs, device=self._device)
        if not hasattr(env, "_moe_prev_skill"):
            env._moe_prev_skill = torch.full((env.num_envs,), -1, dtype=torch.long, device=self._device)
        if not hasattr(env, "_moe_last_weights") or env._moe_last_weights.shape[1] != self.action_dim:
            env._moe_last_weights = torch.zeros(env.num_envs, self.action_dim, device=self._device)
        if not hasattr(env, "_moe_lock_skill"):
            env._moe_lock_skill = bool(self._lock_skill_per_episode)
        if not hasattr(env, "_moe_locked_skill"):
            env._moe_locked_skill = torch.full((env.num_envs,), -1, dtype=torch.long, device=self._device)

    @property
    def action_dim(self) -> int:
        return len(self._experts)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def set_temperature(self, temperature: float) -> None:
        self._temperature = max(temperature, 1.0e-6)

    @property
    def expert_names(self) -> list[str]:
        return [expert.name for expert in self._experts]

    def reset(self, env_ids=None):
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._prev_combined_action.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._prev_combined_action[env_ids] = 0.0

        for expert in self._experts:
            expert.reset(env_ids)

        if hasattr(self._env, "_moe_time_since_switch"):
            if env_ids is None:
                self._env._moe_time_since_switch.zero_()
            else:
                self._env._moe_time_since_switch[env_ids] = 0.0
        if hasattr(self._env, "_moe_prev_skill"):
            if env_ids is None:
                self._env._moe_prev_skill.fill_(-1)
            else:
                self._env._moe_prev_skill[env_ids] = -1
        if hasattr(self._env, "_moe_locked_skill"):
            if env_ids is None:
                self._env._moe_locked_skill.fill_(-1)
            else:
                self._env._moe_locked_skill[env_ids] = -1

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        logits = actions / max(self._temperature, 1.0e-6)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        weights = torch.softmax(logits, dim=-1)
        if self._hard_gate:
            hard_idx = torch.argmax(weights, dim=-1, keepdim=True)
            weights = torch.zeros_like(weights).scatter_(1, hard_idx, 1.0)
        if self._lock_skill_per_episode and getattr(self._env, "_moe_lock_skill", False):
            locked = self._env._moe_locked_skill
            unset = locked < 0
            if torch.any(unset):
                choose = torch.argmax(weights[unset], dim=-1)
                locked[unset] = choose
            weights = torch.zeros_like(weights).scatter_(1, locked.unsqueeze(-1), 1.0)

        # stats for logging
        max_weight = weights.max(dim=-1).values.detach()
        entropy = -(weights * (weights + 1.0e-6).log()).sum(dim=-1).detach()
        one_hot = (max_weight > 0.999).float()
        self._env._moe_max_weight = max_weight
        self._env._moe_weight_entropy = entropy
        self._env._moe_one_hot = one_hot

        expert_actions = []
        with torch.inference_mode():
            for expert in self._experts:
                obs = self._build_expert_obs(expert)
                obs_td = TensorDict({"policy": obs}, batch_size=[obs.shape[0]])
                act = expert.policy.act_inference(obs_td)
                expert_actions.append(act)

        actions_stack = torch.stack(expert_actions, dim=1)
        combined = torch.sum(weights.unsqueeze(-1) * actions_stack, dim=1)

        self._processed_actions[:] = combined
        self._joint_action.process_actions(combined)

        self._prev_combined_action = combined.detach()

        self._update_switch_state(weights)
        self._env._moe_last_weights = weights.detach()

        for expert in self._experts:
            expert.advance()

    def apply_actions(self):
        self._joint_action.apply_actions()

    def _build_expert_obs(self, expert: _ExpertRuntime) -> torch.Tensor:
        robot = expert.robot
        time_steps = expert.time_steps

        joint_pos = expert.motion.joint_pos[time_steps]
        joint_vel = expert.motion.joint_vel[time_steps]
        command = torch.cat([joint_pos, joint_vel], dim=-1)

        anchor_pos_w = expert.motion.body_pos_w[time_steps, expert.motion_anchor_body_index]
        anchor_pos_w = anchor_pos_w + self._env.scene.env_origins
        anchor_quat_w = expert.motion.body_quat_w[time_steps, expert.motion_anchor_body_index]

        robot_anchor_pos_w = robot.data.body_pos_w[:, expert.robot_anchor_body_index]
        robot_anchor_quat_w = robot.data.body_quat_w[:, expert.robot_anchor_body_index]

        anchor_pos_b, anchor_quat_b = subtract_frame_transforms(
            robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w
        )
        anchor_ori_mat = matrix_from_quat(anchor_quat_b)
        anchor_ori_b = anchor_ori_mat[..., :2].reshape(anchor_ori_mat.shape[0], -1)

        base_lin_vel = robot.data.root_lin_vel_b
        base_ang_vel = robot.data.root_ang_vel_b

        joint_pos_rel = robot.data.joint_pos[:, self._joint_ids] - robot.data.default_joint_pos[:, self._joint_ids]
        joint_vel_rel = robot.data.joint_vel[:, self._joint_ids] - robot.data.default_joint_vel[:, self._joint_ids]

        obs = torch.cat(
            [
                command,
                anchor_pos_b,
                anchor_ori_b,
                base_lin_vel,
                base_ang_vel,
                joint_pos_rel,
                joint_vel_rel,
                self._prev_combined_action,
            ],
            dim=-1,
        )
        return obs

    def _update_switch_state(self, weights: torch.Tensor):
        current_skill = torch.argmax(weights, dim=-1)
        prev_skill = self._env._moe_prev_skill
        changed = current_skill != prev_skill

        time_since = self._env._moe_time_since_switch
        time_since = time_since + 1.0
        time_since[changed] = 0.0

        self._env._moe_time_since_switch = time_since
        self._env._moe_prev_skill = current_skill
        self._env._moe_current_skill = current_skill
        self._env._moe_skill_changed = changed.float()


@configclass
class MoEActionCfg(ActionTermCfg):
    class_type: type = MoEActionTerm

    asset_name: str = "robot"
    joint_names: list[str] = MISSING
    scale: float | dict = 1.0
    use_default_offset: bool = True

    model_dir: str = MISSING
    motion_dir: str = MISSING
    expert_names: list[str] | None = None

    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    temperature: float = 0.05
    hard_gate: bool = False
    lock_skill_per_episode: bool = True

    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"


@configclass
class MoEActionsCfg:
    moe: MoEActionCfg = MoEActionCfg()
