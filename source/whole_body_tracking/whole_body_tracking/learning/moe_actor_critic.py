from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP
from whole_body_tracking.learning.expert_command_telemetry import get_expert_commands
from whole_body_tracking.learning.router_telemetry import set_router_weights


def _extract_actor_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Extract actor-only state dict from an RSL-RL checkpoint-like dict."""
    state_dict = checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    candidate_prefixes = (
        "actor.",
        "alg.policy.actor.",
        "policy.actor.",
    )

    for prefix in candidate_prefixes:
        sub = {
            k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if len(sub) > 0:
            return sub

    # Fallback: if keys look like plain MLP layer keys, use directly.
    if any(key.split(".")[0].isdigit() for key in state_dict.keys()):
        return state_dict

    raise RuntimeError(
        "无法从 checkpoint 中解析 actor 参数。"
        "请确认文件来自 RSL-RL policy checkpoint。"
    )


class MoEActorCritic(nn.Module):
    """Router-based MoE policy with frozen single-skill actors.

    - Frozen experts: pre-trained skill actors (joint action heads)
    - Trainable router: outputs skill mixture weights
    - Trainable critic: standard value network
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        frozen_skill_ckpts: list[str],
        frozen_actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        router_hidden_dims: tuple[int] | list[int] = (256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        activation: str = "elu",
        actor_hidden_dims: tuple[int] | list[int] | None = None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        state_dependent_std: bool | None = None,
        skill_weight_temperature: float = 1.0,
        use_grouped_router: bool = True,
        grouped_router_stance_init_bias: float = 2.0,
        **kwargs: dict[str, Any],
    ) -> None:
        # Keep compatibility with default RSL-RL policy kwargs.
        _ = state_dependent_std
        if actor_hidden_dims is not None:
            frozen_actor_hidden_dims = actor_hidden_dims
        if kwargs:
            print(
                "MoEActorCritic.__init__ got unexpected arguments, ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        if len(frozen_skill_ckpts) == 0:
            raise ValueError("MoEActorCritic 需要至少一个 frozen skill checkpoint。")

        self.obs_groups = obs_groups
        self.noise_std_type = noise_std_type
        self.skill_weight_temperature = float(max(1e-6, skill_weight_temperature))
        self.use_grouped_router = bool(use_grouped_router)

        # Infer observation dimensions.
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "MoEActorCritic 仅支持 1D 观测。"
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "MoEActorCritic 仅支持 1D 观测。"
            num_critic_obs += obs[obs_group].shape[-1]

        # Build frozen skill experts.
        self.skill_actors = nn.ModuleList(
            [
                MLP(num_actor_obs, num_actions, frozen_actor_hidden_dims, activation)
                for _ in frozen_skill_ckpts
            ]
        )
        self._load_and_freeze_skill_actors(frozen_skill_ckpts)
        self.num_skills = len(self.skill_actors)

        # Trainable router:
        # - Grouped two-level router (default): choose limb-group, then choose skill in group.
        # - Flat router (fallback): directly output logits over skills.
        self.group_names: list[str] = []
        self.group_to_skill_indices: dict[str, list[int]] = {}
        self.group_router: MLP | None = None
        self.intra_routers = nn.ModuleDict()
        self.router: MLP | None = None
        self._grouped_router_stance_init_bias = float(grouped_router_stance_init_bias)
        if self.use_grouped_router:
            self._build_grouped_router(num_actor_obs, router_hidden_dims, activation)
        else:
            self.router = MLP(num_actor_obs, self.num_skills, router_hidden_dims, activation)

        # Trainable critic.
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        if self.use_grouped_router:
            print(f"MoE group router MLP: {self.group_router}")
            for group_name, intra_router in self.intra_routers.items():
                print(f"MoE intra router [{group_name}]: {intra_router}")
        else:
            print(f"MoE router MLP: {self.router}")
        print(f"MoE critic MLP: {self.critic}")
        print(f"Frozen skills loaded: {self.num_skills}")

        # Observation normalization.
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )

        # Action noise.
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"未知噪声类型: {self.noise_std_type}")

        self.distribution: Normal | None = None
        self._last_skill_weights: torch.Tensor | None = None
        self._warned_expert_command_mismatch = False
        Normal.set_default_validate_args(False)

    def _load_and_freeze_skill_actors(self, ckpt_paths: list[str]) -> None:
        for skill_actor, ckpt_path in zip(self.skill_actors, ckpt_paths):
            path = pathlib.Path(ckpt_path)
            if not path.exists():
                raise FileNotFoundError(f"冻结模型不存在: {path}")
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            actor_state = _extract_actor_state_dict(checkpoint)
            missing, unexpected = skill_actor.load_state_dict(actor_state, strict=False)
            if missing:
                raise RuntimeError(f"加载 {path.name} 失败，缺失参数: {missing}")
            if unexpected:
                print(f"[MoEActorCritic] {path.name} 有未使用参数: {unexpected}")
            for p in skill_actor.parameters():
                p.requires_grad_(False)
            skill_actor.eval()

    @staticmethod
    def _find_last_linear(module: nn.Module) -> nn.Linear | None:
        for layer in reversed(list(module.modules())):
            if isinstance(layer, nn.Linear):
                return layer
        return None

    def _build_grouped_router(self, num_actor_obs: int, router_hidden_dims, activation: str) -> None:
        """Build two-level grouped router.

        Groups are fixed to the Stage4 skill order:
            0 cross, 1 swing, 2 hook_left, 3 frontkick,
            4 roundhouse_right, 5 roundhouse_left, 6 stance
        """
        if self.num_skills != 7:
            raise ValueError(
                "Grouped router expects exactly 7 skills in Stage4 order "
                f"(got {self.num_skills})."
            )

        self.group_names = ["right_hand", "left_hand", "right_foot", "left_foot", "stance"]
        self.group_to_skill_indices = {
            "right_hand": [0, 1],   # cross, swing
            "left_hand": [2],       # hook_left
            "right_foot": [3, 4],   # frontkick, roundhouse_right
            "left_foot": [5],       # roundhouse_left
            "stance": [6],          # stance
        }

        self.group_router = MLP(num_actor_obs, len(self.group_names), router_hidden_dims, activation)
        self.intra_routers = nn.ModuleDict()
        for group_name, skill_indices in self.group_to_skill_indices.items():
            if len(skill_indices) > 1:
                self.intra_routers[group_name] = MLP(
                    num_actor_obs,
                    len(skill_indices),
                    router_hidden_dims,
                    activation,
                )

        # Bias grouped-router initialization toward stance so early training
        # starts from more stable behavior.
        last_linear = self._find_last_linear(self.group_router)
        if last_linear is not None and last_linear.bias is not None:
            with torch.no_grad():
                last_linear.bias.zero_()
                stance_group_idx = self.group_names.index("stance")
                last_linear.bias[stance_group_idx] = self._grouped_router_stance_init_bias

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore[return-value]

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev  # type: ignore[return-value]

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)  # type: ignore[return-value]

    @property
    def last_skill_weights(self) -> torch.Tensor | None:
        return self._last_skill_weights

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def _compute_skill_means(self, actor_obs: torch.Tensor) -> torch.Tensor:
        # No gradient through frozen experts.
        with torch.no_grad():
            expert_commands = get_expert_commands()
            if (
                expert_commands is not None
                and expert_commands.ndim == 3
                and expert_commands.shape[0] == actor_obs.shape[0]
                and expert_commands.shape[1] == self.num_skills
            ):
                command_dim = int(expert_commands.shape[-1])
                if 0 < command_dim <= actor_obs.shape[-1]:
                    expert_obs = actor_obs.unsqueeze(1).repeat(1, self.num_skills, 1)
                    expert_obs[:, :, :command_dim] = expert_commands.to(
                        device=actor_obs.device,
                        dtype=actor_obs.dtype,
                    )
                    per_skill = [actor(expert_obs[:, idx, :]) for idx, actor in enumerate(self.skill_actors)]
                else:
                    if not self._warned_expert_command_mismatch:
                        print(
                            "[MoEActorCritic] expert command dim mismatch: "
                            f"command_dim={command_dim}, actor_obs_dim={actor_obs.shape[-1]}. "
                            "Fallback to shared actor_obs for all experts."
                        )
                        self._warned_expert_command_mismatch = True
                    per_skill = [actor(actor_obs) for actor in self.skill_actors]
            else:
                per_skill = [actor(actor_obs) for actor in self.skill_actors]
            # [batch, num_skills, num_actions]
            return torch.stack(per_skill, dim=1)

    def _compute_router_weights(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if self.use_grouped_router:
            if self.group_router is None:
                raise RuntimeError("Grouped router is enabled but group_router is None.")
            group_logits = self.group_router(actor_obs) / self.skill_weight_temperature
            group_weights = torch.softmax(group_logits, dim=-1)  # [batch, num_groups]

            weights = torch.zeros(
                actor_obs.shape[0],
                self.num_skills,
                dtype=actor_obs.dtype,
                device=actor_obs.device,
            )

            for group_idx, group_name in enumerate(self.group_names):
                skill_indices = self.group_to_skill_indices[group_name]
                group_w = group_weights[:, group_idx : group_idx + 1]  # [batch, 1]
                if len(skill_indices) == 1:
                    weights[:, skill_indices[0]] = group_w.squeeze(-1)
                    continue

                intra_router = self.intra_routers[group_name]
                intra_logits = intra_router(actor_obs) / self.skill_weight_temperature
                intra_weights = torch.softmax(intra_logits, dim=-1)  # [batch, n_skill_in_group]
                weights[:, skill_indices] = group_w * intra_weights
        else:
            if self.router is None:
                raise RuntimeError("Flat router is enabled but router is None.")
            logits = self.router(actor_obs)
            logits = logits / self.skill_weight_temperature
            weights = torch.softmax(logits, dim=-1)

        self._last_skill_weights = weights
        set_router_weights(weights)
        return weights

    def _compute_action_mean(self, actor_obs: torch.Tensor) -> torch.Tensor:
        skill_means = self._compute_skill_means(actor_obs)
        skill_weights = self._compute_router_weights(actor_obs)
        # Weighted sum over experts -> [batch, num_actions]
        return torch.sum(skill_means * skill_weights.unsqueeze(-1), dim=1)

    def _update_distribution(self, obs: TensorDict) -> None:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        mean = self._compute_action_mean(actor_obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()  # type: ignore[return-value]

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self._compute_action_mean(actor_obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)  # type: ignore[return-value]

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
