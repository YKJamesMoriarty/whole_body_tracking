"""Stage 3 技能注册表 (Skill Registry)

统一管理所有格斗技能的配置信息，供打标脚本、高层分类器和推理脚本使用。

新7技能说明 (基于 basic_model/Mimic/ 重训模型):
    所有技能的 skill_type_idx 和 effector_one_hot_idx 在训练时已统一，
    observations.py 中直接硬编码固定值:
        active_effector_one_hot  → [0, 0, 0, 1] (index=3)
        skill_type_one_hot       → [:, 7] = 1.0

    skill_id 用于标签文件（labels.npy）的整数标识，不影响模型输入。

G1 body_names 中对应肢体:
    右手: right_wrist_yaw_link   (index 13 in body_names)
    左手: left_wrist_yaw_link    (index 10 in body_names)
    右脚: right_ankle_roll_link  (index  6 in body_names)
    左脚: left_ankle_roll_link   (index  3 in body_names)
"""

from __future__ import annotations

import pathlib
from typing import TypedDict


class SkillConfig(TypedDict):
    """单个技能的完整配置信息"""
    skill_id: int               # 技能 ID，用于 labels.npy 标识
    model_filename: str         # .pt 文件名 (位于 model_dir/)
    motion_filename: str        # .npz 文件名 (位于 motion_dir/)
    effector_body_name: str     # 攻击肢体名称 (用于命中检测，必须在 G1FlatEnvCfg.body_names 中)
    skill_type_idx: int         # skill_type_one_hot 中置 1 的维度（所有新技能统一=7）
    effector_one_hot_idx: int   # active_effector_one_hot 中置 1 的维度（所有新技能统一=0，即 [1,0,0,0]）
    description: str            # 中文描述


# ============================================================
# 技能注册表 (7 个新技能，基于 basic_model/Mimic/ 模型)
# 新增技能只需在此添加一项，其余代码自动适配
# ============================================================

SKILL_CONFIGS: dict[str, SkillConfig] = {
    "cross_right": {
        "skill_id": 0,
        "model_filename": "trim_cross_right_normal_body_2_150.pt",
        "motion_filename": "trim_cross_right_normal_body_2_150.npz",
        "effector_body_name": "right_wrist_yaw_link",
        "skill_type_idx": 7,    # 所有新技能统一
        "effector_one_hot_idx": 0,  # 所有新技能统一（含stance），训练时固定为 [1,0,0,0]
        "description": "右手直拳",
    },
    "swing_right": {
        "skill_id": 1,
        "model_filename": "trim_swing_right_normal_head2_150.pt",
        "motion_filename": "trim_swing_right_normal_head2_150.npz",
        "effector_body_name": "right_wrist_yaw_link",
        "skill_type_idx": 7,
        "effector_one_hot_idx": 0,
        "description": "右手摆拳",
    },
    "hook_left": {
        "skill_id": 2,
        "model_filename": "trim_hook_left_normal_body2_150.pt",
        "motion_filename": "trim_hook_left_normal_body2_150.npz",
        "effector_body_name": "left_wrist_yaw_link",
        "skill_type_idx": 7,
        "effector_one_hot_idx": 0,
        "description": "左手勾拳",
    },
    "roundhouse_right": {
        "skill_id": 3,
        "model_filename": "trim_roundhouse_right_normal_mid_no_bag_2_150.pt",
        "motion_filename": "trim_roundhouse_right_normal_mid_no_bag_2_150.npz",
        "effector_body_name": "right_ankle_roll_link",
        "skill_type_idx": 7,
        "effector_one_hot_idx": 0,
        "description": "右脚中位鞭腿",
    },
    "roundhouse_left": {
        "skill_id": 4,
        "model_filename": "trim_roundhouse_left_normal_mid_no_bag_1_150.pt",
        "motion_filename": "trim_roundhouse_left_normal_mid_no_bag_1_150.npz",
        "effector_body_name": "left_ankle_roll_link",
        "skill_type_idx": 7,
        "effector_one_hot_idx": 0,
        "description": "左脚中位鞭腿",
    },
    "frontkick_right": {
        "skill_id": 5,
        "model_filename": "trim_frontkick_right_fast_body_no_bag_2_150.pt",
        "motion_filename": "trim_frontkick_right_fast_body_no_bag_2_150.npz",
        "effector_body_name": "right_ankle_roll_link",
        "skill_type_idx": 7,
        "effector_one_hot_idx": 0,
        "description": "右脚正蹬",
    },
    "stance": {
        "skill_id": 6,
        "model_filename": "trim_stance_orthodox_idle_normal_2_150.pt",
        "motion_filename": "trim_stance_orthodox_idle_normal_2_150.npz",
        "effector_body_name": "right_wrist_yaw_link",  # stance 无攻击，占位用右手
        "skill_type_idx": 7,    # 与攻击技能统一（训练时已确认）
        "effector_one_hot_idx": 0,  # 与攻击技能统一（训练时已确认）
        "description": "防守站姿 (无目标时默认技能)",
    },
}

# ============================================================
# 常用分组
# ============================================================

# 参与打标测试的进攻技能（不含 stance，stance 作为兜底标签）
ATTACK_SKILLS: list[str] = [
    "cross_right",
    "swing_right",
    "hook_left",
    "roundhouse_right",
    "roundhouse_left",
    "frontkick_right",
]

# 所有技能（含 stance）
ALL_SKILLS: list[str] = ATTACK_SKILLS + ["stance"]

# 无进攻技能能命中目标时的兜底技能 ID
STANCE_SKILL_ID: int = SKILL_CONFIGS["stance"]["skill_id"]  # 6

# 腿部技能集合（鞭腿/正蹬），用于需要区分手/脚的场景
KICK_SKILLS: list[str] = ["roundhouse_right", "roundhouse_left", "frontkick_right"]

# ============================================================
# 路径辅助函数
# ============================================================

def get_model_path(skill_name: str, model_dir: str | pathlib.Path) -> pathlib.Path:
    """返回技能 .pt 文件的完整路径"""
    cfg = SKILL_CONFIGS[skill_name]
    return pathlib.Path(model_dir) / cfg["model_filename"]


def get_motion_path(skill_name: str, motion_dir: str | pathlib.Path) -> pathlib.Path:
    """返回技能参考动作 .npz 文件的完整路径"""
    cfg = SKILL_CONFIGS[skill_name]
    return pathlib.Path(motion_dir) / cfg["motion_filename"]
