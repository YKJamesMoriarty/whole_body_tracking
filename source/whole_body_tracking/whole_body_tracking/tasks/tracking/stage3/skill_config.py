"""Stage 3 技能注册表 (Skill Registry)

统一管理所有格斗技能的配置信息，供打标脚本、高层分类器和推理脚本使用。

技能 ID 规则（与 observations.py skill_type_one_hot 对照表一致）:
    0: r-Cross (右直拳)
    1: r-swing (右摆拳)
    2: roundhouse_right_normal_low  (不好用，暂无 model)
    3: roundhouse_right_fast_high   (右高位鞭腿)
    4: frontkick_right_normal_body  (右脚正蹬)
    5: stance (摆架子/防守，无目标时默认)
    6: roundhouse_right_normal_mid  (暂无 model)
    7: hook_left_normal_body        (暂无 model)
    8: roundhouse_left_normal_mid   (暂无 model)
    9-15: 预留

effector_one_hot_idx 格式: [左手=0, 右手=1, 左脚=2, 右脚=3]

G1 body_names 中对应肢体:
    右手: right_wrist_yaw_link  (index 13 in body_names)
    右脚: right_ankle_roll_link (index 6 in body_names)
"""

from __future__ import annotations

import pathlib
from typing import TypedDict


class SkillConfig(TypedDict):
    """单个技能的完整配置信息"""
    skill_id: int               # 技能 ID (与 skill_type_one_hot 索引一致)
    model_filename: str         # .pt 文件名 (位于 model_dir/)
    motion_filename: str        # .npz 文件名 (位于 motion_dir/)
    effector_body_name: str     # 攻击肢体名称 (必须在 G1FlatEnvCfg.body_names 中)
    skill_type_idx: int         # skill_type_one_hot 中置 1 的维度
    effector_one_hot_idx: int   # active_effector_one_hot 中置 1 的维度
    description: str            # 中文描述


# ============================================================
# 技能注册表 (目前已完成 Stage 2 训练的 5 个技能)
# 新增技能只需在此添加一项，其余代码自动适配
# ============================================================

SKILL_CONFIGS: dict[str, SkillConfig] = {
    "cross": {
        "skill_id": 0,
        "model_filename": "cross_108500.pt",
        "motion_filename": "cross_right_normal_body2.npz",
        "effector_body_name": "right_wrist_yaw_link",
        "skill_type_idx": 0,
        "effector_one_hot_idx": 1,  # 右手
        "description": "右手直拳",
    },
    "swing": {
        "skill_id": 1,
        "model_filename": "swing_145500.pt",
        "motion_filename": "swing_right_normal_head2.npz",
        "effector_body_name": "right_wrist_yaw_link",
        "skill_type_idx": 1,
        "effector_one_hot_idx": 1,  # 右手
        "description": "右手摆拳",
    },
    "roundhouse": {
        "skill_id": 3,
        "model_filename": "r_h_roundhouse_144000.pt",
        "motion_filename": "roundhouse_right_fast_high2.npz",
        "effector_body_name": "right_ankle_roll_link",
        "skill_type_idx": 3,
        "effector_one_hot_idx": 3,  # 右脚
        "description": "右脚高位鞭腿",
    },
    "frontkick": {
        "skill_id": 4,
        "model_filename": "r_frontkick_150000.pt",
        "motion_filename": "frontkick_right_normal_body2.npz",
        "effector_body_name": "right_ankle_roll_link",
        "skill_type_idx": 4,
        "effector_one_hot_idx": 3,  # 右脚
        "description": "右脚正蹬",
    },
    "stance": {
        "skill_id": 5,
        "model_filename": "stance_9500.pt",
        "motion_filename": "stance_orthodox_idle_normal_2_100.npz",
        "effector_body_name": "right_wrist_yaw_link",  # stance 无攻击，占位
        "skill_type_idx": 5,
        "effector_one_hot_idx": -1,  # -1 = 无进攻肢体，active_effector_one_hot 返回全零
        "description": "防守站姿 (无目标时默认技能)",
    },
}

# ============================================================
# 常用分组
# ============================================================

# 参与打标测试的进攻技能（不含 stance，stance 作为兜底标签）
ATTACK_SKILLS: list[str] = ["cross", "swing", "roundhouse", "frontkick"]

# 所有技能（含 stance）
ALL_SKILLS: list[str] = ATTACK_SKILLS + ["stance"]

# 无进攻技能能命中目标时的兜底技能 ID
STANCE_SKILL_ID: int = SKILL_CONFIGS["stance"]["skill_id"]  # 5


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
