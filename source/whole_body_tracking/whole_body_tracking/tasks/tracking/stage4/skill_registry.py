from __future__ import annotations

import pathlib

# NOTE:
# Keep this order stable: router weight i always maps to the same frozen expert.
STAGE4_SKILL_MODEL_FILENAMES: list[str] = [
    "trim_cross_right_normal_body_2_150.pt",
    "trim_swing_right_normal_head2_150.pt",
    "trim_hook_left_normal_body2_150.pt",
    "trim_frontkick_right_fast_body_no_bag_2_150.pt",
    "trim_roundhouse_right_normal_mid_no_bag_2_150.pt",
    "trim_roundhouse_left_normal_mid_no_bag_1_150.pt",
    "trim_stance_orthodox_idle_normal_2_150.pt",
]
# Keep names aligned one-by-one with STAGE4_SKILL_MODEL_FILENAMES.
STAGE4_SKILL_MOTION_FILENAMES: list[str] = [
    "trim_cross_right_normal_body_2_150.npz",
    "trim_swing_right_normal_head2_150.npz",
    "trim_hook_left_normal_body2_150.npz",
    "trim_frontkick_right_fast_body_no_bag_2_150.npz",
    "trim_roundhouse_right_normal_mid_no_bag_2_150.npz",
    "trim_roundhouse_left_normal_mid_no_bag_1_150.npz",
    "trim_stance_orthodox_idle_normal_2_150.npz",
]
# Keep names aligned one-by-one with STAGE4_SKILL_MODEL_FILENAMES.
STAGE4_SKILL_METRIC_NAMES: list[str] = [
    "cross",
    "swing",
    "hook_left",
    "frontkick",
    "roundhouse_right",
    "roundhouse_left",
    "stance",
]

if not (
    len(STAGE4_SKILL_MODEL_FILENAMES)
    == len(STAGE4_SKILL_MOTION_FILENAMES)
    == len(STAGE4_SKILL_METRIC_NAMES)
):
    raise RuntimeError(
        "Stage4 skill registry 配置长度不一致："
        f"models={len(STAGE4_SKILL_MODEL_FILENAMES)}, "
        f"motions={len(STAGE4_SKILL_MOTION_FILENAMES)}, "
        f"metrics={len(STAGE4_SKILL_METRIC_NAMES)}"
    )


def resolve_stage4_model_paths(model_dir: str | pathlib.Path) -> list[str]:
    """Resolve Stage4 frozen expert checkpoints in a deterministic order."""
    model_dir = pathlib.Path(model_dir)
    paths = [model_dir / name for name in STAGE4_SKILL_MODEL_FILENAMES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage4 冻结模型缺失，请检查 basic_model 目录：\n" + "\n".join(missing)
        )
    return [str(path) for path in paths]


def resolve_stage4_motion_paths(motion_dir: str | pathlib.Path) -> list[str]:
    """Resolve Stage4 per-skill reference motions in deterministic order."""
    motion_dir = pathlib.Path(motion_dir)
    paths = [motion_dir / name for name in STAGE4_SKILL_MOTION_FILENAMES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage4 专家参考动作缺失，请检查 iros_motion/npz 目录：\n" + "\n".join(missing)
        )
    return [str(path) for path in paths]
