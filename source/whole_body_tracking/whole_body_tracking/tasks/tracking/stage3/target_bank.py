from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

from whole_body_tracking.tasks.tracking.stage3.skill_config import SKILL_CONFIGS


@dataclass(frozen=True)
class TargetCluster:
    """One skill's target cluster filtered from offline label results."""

    skill_name: str
    skill_id: int
    points: np.ndarray
    best_accuracies: np.ndarray

    @property
    def size(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class LabelData:
    """Raw label artifacts from scripts/rsl_rl/label_skills.py."""

    grid_points: np.ndarray
    all_accuracies: np.ndarray
    labels: np.ndarray
    metadata: dict

    @classmethod
    def from_dir(cls, labels_dir: str | pathlib.Path) -> "LabelData":
        labels_dir = pathlib.Path(labels_dir)
        if not labels_dir.exists():
            raise FileNotFoundError(f"labels 目录不存在: {labels_dir}")

        grid_points = np.load(labels_dir / "grid_points.npy")
        all_accuracies = np.load(labels_dir / "all_accuracies.npy")
        labels = np.load(labels_dir / "labels.npy")
        metadata = {}
        meta_path = labels_dir / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        if grid_points.ndim != 2 or grid_points.shape[1] != 3:
            raise ValueError(f"grid_points.npy 期望形状 (N, 3)，实际: {grid_points.shape}")
        if all_accuracies.ndim != 2:
            raise ValueError(f"all_accuracies.npy 期望二维，实际: {all_accuracies.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels.npy 期望一维，实际: {labels.shape}")
        if len(grid_points) != len(labels) or len(grid_points) != all_accuracies.shape[0]:
            raise ValueError(
                "labels 文件长度不一致: "
                f"grid={len(grid_points)}, labels={len(labels)}, acc={all_accuracies.shape[0]}"
            )

        return cls(
            grid_points=grid_points.astype(np.float32),
            all_accuracies=all_accuracies.astype(np.float32),
            labels=labels.astype(np.int32),
            metadata=metadata,
        )


class LabeledTargetBank:
    """Build skill-specific target clusters from offline label artifacts."""

    def __init__(
        self,
        label_data: LabelData,
        attack_skills: list[str],
        min_accuracy: float,
        min_distance: float | None = None,
        max_distance: float | None = None,
    ):
        self.label_data = label_data
        self.attack_skills = attack_skills
        self.min_accuracy = float(min_accuracy)
        self.min_distance = float(min_distance) if min_distance is not None else None
        self.max_distance = float(max_distance) if max_distance is not None else None

        if self.min_distance is not None and self.min_distance < 0.0:
            raise ValueError(f"min_distance 必须 >= 0，实际: {self.min_distance}")
        if self.max_distance is not None and self.max_distance < 0.0:
            raise ValueError(f"max_distance 必须 >= 0，实际: {self.max_distance}")
        if (
            self.min_distance is not None
            and self.max_distance is not None
            and self.min_distance > self.max_distance
        ):
            raise ValueError(
                f"min_distance ({self.min_distance}) 不能大于 max_distance ({self.max_distance})"
            )

        skills_tested = label_data.metadata.get("skills_tested")
        if not isinstance(skills_tested, list) or len(skills_tested) != label_data.all_accuracies.shape[1]:
            skills_tested = [f"skill_{k}" for k in range(label_data.all_accuracies.shape[1])]

        best_col = np.argmax(label_data.all_accuracies, axis=1)
        best_acc = np.max(label_data.all_accuracies, axis=1)
        best_skill_names = np.array([skills_tested[int(i)] for i in best_col], dtype=object)
        point_distance = np.linalg.norm(label_data.grid_points, axis=1)

        distance_mask = np.ones(len(label_data.grid_points), dtype=bool)
        if self.min_distance is not None:
            distance_mask &= point_distance >= self.min_distance
        if self.max_distance is not None:
            distance_mask &= point_distance <= self.max_distance

        self._clusters: dict[str, TargetCluster] = {}
        for skill_name in attack_skills:
            if skill_name not in SKILL_CONFIGS:
                raise ValueError(f"未知技能: {skill_name}, 可用: {list(SKILL_CONFIGS.keys())}")

            skill_id = SKILL_CONFIGS[skill_name]["skill_id"]
            mask = (best_skill_names == skill_name) & (best_acc >= self.min_accuracy) & distance_mask
            points = label_data.grid_points[mask]
            acc = best_acc[mask]
            self._clusters[skill_name] = TargetCluster(
                skill_name=skill_name,
                skill_id=skill_id,
                points=points,
                best_accuracies=acc,
            )

    @classmethod
    def from_dir(
        cls,
        labels_dir: str | pathlib.Path,
        attack_skills: list[str],
        min_accuracy: float,
        min_distance: float | None = None,
        max_distance: float | None = None,
    ) -> "LabeledTargetBank":
        return cls(
            LabelData.from_dir(labels_dir),
            attack_skills,
            min_accuracy,
            min_distance=min_distance,
            max_distance=max_distance,
        )

    def summary(self) -> dict[str, int]:
        return {skill_name: cluster.size for skill_name, cluster in self._clusters.items()}

    def sample_target(self, skill_name: str, rng: np.random.Generator) -> np.ndarray:
        cluster = self._clusters[skill_name]
        if cluster.size == 0:
            raise RuntimeError(
                f"技能 {skill_name} 在过滤条件下无可用点: "
                f"min_accuracy={self.min_accuracy:.3f}, "
                f"min_distance={self.min_distance}, max_distance={self.max_distance}"
            )
        idx = int(rng.integers(0, cluster.size))
        return cluster.points[idx].copy()
