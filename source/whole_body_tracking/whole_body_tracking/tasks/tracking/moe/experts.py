from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    model_file: str
    motion_file: str


EXPERTS: list[ExpertSpec] = [
    ExpertSpec(
        name="cross_right_body",
        model_file="trim_cross_right_normal_body_2_150.pt",
        motion_file="trim_cross_right_normal_body_2_150.npz",
    ),
    ExpertSpec(
        name="frontkick_right_body",
        model_file="trim_frontkick_right_fast_body_no_bag_2_150.pt",
        motion_file="trim_frontkick_right_fast_body_no_bag_2_150.npz",
    ),
    ExpertSpec(
        name="roundhouse_left_mid",
        model_file="trim_roundhouse_left_normal_mid_no_bag_1_150.pt",
        motion_file="trim_roundhouse_left_normal_mid_no_bag_1_150.npz",
    ),
    ExpertSpec(
        name="roundhouse_right_mid",
        model_file="trim_roundhouse_right_normal_mid_no_bag_2_150.pt",
        motion_file="trim_roundhouse_right_normal_mid_no_bag_2_150.npz",
    ),
    ExpertSpec(
        name="swing_right_head",
        model_file="trim_swing_right_normal_head2_150.pt",
        motion_file="trim_swing_right_normal_head2_150.npz",
    ),
    ExpertSpec(
        name="stance",
        model_file="trim_stance_orthodox_idle_normal_2_150.pt",
        motion_file="trim_stance_orthodox_idle_normal_2_150.npz",
    ),
]

ATTACK_SKILLS: list[str] = [
    "cross_right_body",
    "frontkick_right_body",
    "roundhouse_left_mid",
    "roundhouse_right_mid",
    "swing_right_head",
]


def filter_experts(names: list[str] | None) -> list[ExpertSpec]:
    if not names:
        return EXPERTS
    name_set = {n.strip() for n in names if n.strip()}
    selected = [s for s in EXPERTS if s.name in name_set]
    missing = sorted(name_set - {s.name for s in selected})
    if missing:
        raise ValueError(f"Unknown expert names: {missing}")
    return selected
