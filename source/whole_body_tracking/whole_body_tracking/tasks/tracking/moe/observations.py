from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from .commands import AttackTargetCommand


def target_pos_b(env: ManagerBasedEnv, command_name: str = "attack_target") -> torch.Tensor:
    command: AttackTargetCommand = env.command_manager.get_term(command_name)
    return command.target_pos_b


def target_present(env: ManagerBasedEnv, command_name: str = "attack_target") -> torch.Tensor:
    command: AttackTargetCommand = env.command_manager.get_term(command_name)
    return command.target_present


def time_since_switch(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "_moe_time_since_switch"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env._moe_time_since_switch.unsqueeze(-1)
