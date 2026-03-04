from __future__ import annotations

import pathlib
from dataclasses import dataclass

import torch
import torch.nn as nn


def _activation_from_name(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class DiscBundleMeta:
    obs_dim: int
    num_hidden_layers: int
    activation: str


class AmpDiscriminator(nn.Module):
    """Inference-only AMP discriminator loaded from exported MimicKit bundle."""

    def __init__(
        self,
        linears: list[nn.Linear],
        logit: nn.Linear,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        activation: str = "elu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(linears)
        self.logit = logit
        self.norm_mean = nn.Parameter(norm_mean.clone().float(), requires_grad=False)
        self.norm_std = nn.Parameter(norm_std.clone().float(), requires_grad=False)
        self.activation = _activation_from_name(activation)
        self.meta = DiscBundleMeta(
            obs_dim=int(norm_mean.numel()),
            num_hidden_layers=len(linears),
            activation=activation,
        )

    @staticmethod
    def from_bundle(bundle_path: str | pathlib.Path, activation: str = "elu") -> "AmpDiscriminator":
        bundle_path = pathlib.Path(bundle_path).expanduser().resolve()
        if not bundle_path.exists():
            raise FileNotFoundError(f"AMP bundle not found: {bundle_path}")

        bundle = torch.load(str(bundle_path), map_location="cpu")
        if "disc_state_dict" not in bundle:
            raise KeyError("Invalid AMP bundle: missing 'disc_state_dict'")

        disc_state = bundle["disc_state_dict"]
        norm_mean = bundle["disc_obs_norm_mean"].float()
        norm_std = bundle["disc_obs_norm_std"].float().clamp_min(1e-6)

        hidden_layer_ids = sorted(
            {
                int(key.split(".")[1])
                for key in disc_state.keys()
                if key.startswith("disc_layers.") and key.endswith(".weight")
            }
        )
        if len(hidden_layer_ids) == 0:
            raise ValueError("Invalid AMP bundle: no hidden layer weights found.")

        linears: list[nn.Linear] = []
        for layer_id in hidden_layer_ids:
            w = disc_state[f"disc_layers.{layer_id}.weight"]
            b = disc_state[f"disc_layers.{layer_id}.bias"]
            linear = nn.Linear(w.shape[1], w.shape[0], bias=True)
            linear.weight.data.copy_(w)
            linear.bias.data.copy_(b)
            linears.append(linear)

        logit_w = disc_state["disc_logits.weight"]
        logit_b = disc_state["disc_logits.bias"]
        logit = nn.Linear(logit_w.shape[1], logit_w.shape[0], bias=True)
        logit.weight.data.copy_(logit_w)
        logit.bias.data.copy_(logit_b)

        model = AmpDiscriminator(
            linears=linears,
            logit=logit,
            norm_mean=norm_mean,
            norm_std=norm_std,
            activation=activation,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def normalize_obs(self, disc_obs: torch.Tensor) -> torch.Tensor:
        return (disc_obs - self.norm_mean) / self.norm_std

    def forward(self, disc_obs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.normalize_obs(disc_obs) if normalize else disc_obs
        for linear in self.layers:
            x = self.activation(linear(x))
        return self.logit(x).squeeze(-1)

    @torch.no_grad()
    def style_reward(self, disc_obs: torch.Tensor, disc_reward_scale: float = 2.0, eps: float = 1e-4) -> torch.Tensor:
        logits = self.forward(disc_obs, normalize=True)
        prob = torch.sigmoid(logits)
        reward = -torch.log(torch.clamp(1.0 - prob, min=eps))
        return reward * disc_reward_scale
