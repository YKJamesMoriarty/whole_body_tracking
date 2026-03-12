from __future__ import annotations

import torch

_LAST_EXPERT_COMMANDS: torch.Tensor | None = None
_LAST_EXPERT_TARGET_REL_POS: torch.Tensor | None = None


def set_expert_commands(commands: torch.Tensor):
    """Store latest per-skill command bank.

    Expected shape: [num_envs, num_skills, command_dim]
    """
    global _LAST_EXPERT_COMMANDS
    _LAST_EXPERT_COMMANDS = commands.detach()


def set_expert_target_rel_pos(target_rel_pos: torch.Tensor):
    """Store latest per-skill target_rel_pos (local frame).

    Expected shape: [num_envs, num_skills, 3]
    """
    global _LAST_EXPERT_TARGET_REL_POS
    _LAST_EXPERT_TARGET_REL_POS = target_rel_pos.detach()


def get_expert_commands() -> torch.Tensor | None:
    """Get latest per-skill command bank."""
    return _LAST_EXPERT_COMMANDS


def get_expert_target_rel_pos() -> torch.Tensor | None:
    """Get latest per-skill target_rel_pos bank (local frame)."""
    return _LAST_EXPERT_TARGET_REL_POS


def clear_expert_commands():
    global _LAST_EXPERT_COMMANDS
    _LAST_EXPERT_COMMANDS = None
    global _LAST_EXPERT_TARGET_REL_POS
    _LAST_EXPERT_TARGET_REL_POS = None
