from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillSpec:
    name: str
    model_file: str
    motion_file: str
    end_effector: str


SKILLS: list[SkillSpec] = [
    SkillSpec(
        name="cross_right_body",
        model_file="trim_cross_right_normal_body_2_150.pt",
        motion_file="trim_cross_right_normal_body_2_150.npz",
        end_effector="right_wrist_yaw_link",
    ),
    SkillSpec(
        name="frontkick_right_body",
        model_file="trim_frontkick_right_fast_body_no_bag_2_150.pt",
        motion_file="trim_frontkick_right_fast_body_no_bag_2_150.npz",
        end_effector="right_ankle_roll_link",
    ),
    SkillSpec(
        name="roundhouse_left_mid",
        model_file="trim_roundhouse_left_normal_mid_no_bag_1_150.pt",
        motion_file="trim_roundhouse_left_normal_mid_no_bag_1_150.npz",
        end_effector="left_ankle_roll_link",
    ),
    SkillSpec(
        name="roundhouse_right_mid",
        model_file="trim_roundhouse_right_normal_mid_no_bag_2_150.pt",
        motion_file="trim_roundhouse_right_normal_mid_no_bag_2_150.npz",
        end_effector="right_ankle_roll_link",
    ),
    SkillSpec(
        name="swing_right_head",
        model_file="trim_swing_right_normal_head2_150.pt",
        motion_file="trim_swing_right_normal_head2_150.npz",
        end_effector="right_wrist_yaw_link",
    ),
]


def filter_skills(names: list[str] | None) -> list[SkillSpec]:
    if not names:
        return SKILLS
    name_set = {n.strip() for n in names if n.strip()}
    selected = [s for s in SKILLS if s.name in name_set]
    missing = sorted(name_set - {s.name for s in selected})
    if missing:
        raise ValueError(f"Unknown skill names: {missing}")
    return selected
