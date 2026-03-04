from __future__ import annotations

import torch

_LAST_EXPERT_COMMANDS: torch.Tensor | None = None


def set_expert_commands(commands: torch.Tensor):
    """Store latest per-skill command bank.

    Expected shape: [num_envs, num_skills, command_dim]
    """
    global _LAST_EXPERT_COMMANDS
    _LAST_EXPERT_COMMANDS = commands.detach()


def get_expert_commands() -> torch.Tensor | None:
    """Get latest per-skill command bank."""
    return _LAST_EXPERT_COMMANDS


def clear_expert_commands():
    global _LAST_EXPERT_COMMANDS
    _LAST_EXPERT_COMMANDS = None

