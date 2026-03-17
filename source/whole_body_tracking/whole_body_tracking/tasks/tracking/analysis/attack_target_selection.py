from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    points: np.ndarray
    num_source_points: int
    num_voxel_points: int
    method: str
    voxel_size: float


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return points[unique_idx]


def farthest_point_sampling(points: np.ndarray, num_points: int, seed: int | None = None) -> np.ndarray:
    if points.shape[0] <= num_points:
        return points
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    selected = np.empty(num_points, dtype=np.int64)
    # start from a random point
    selected[0] = int(rng.integers(0, n))
    dists = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, num_points):
        p = points[selected[i - 1]]
        diff = points - p
        dist2 = np.einsum("ij,ij->i", diff, diff)
        dists = np.minimum(dists, dist2)
        selected[i] = int(np.argmax(dists))
    return points[selected]


def random_sampling(points: np.ndarray, num_points: int, seed: int | None = None) -> np.ndarray:
    if points.shape[0] <= num_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=num_points, replace=False)
    return points[idx]


def select_points(
    points: np.ndarray,
    *,
    voxel_size: float,
    num_points: int,
    method: str = "fps",
    seed: int | None = None,
) -> SelectionResult:
    num_source = points.shape[0]
    voxel_points = voxel_downsample(points, voxel_size)
    num_voxel = voxel_points.shape[0]

    if method == "fps":
        selected = farthest_point_sampling(voxel_points, num_points, seed)
    elif method == "random":
        selected = random_sampling(voxel_points, num_points, seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    return SelectionResult(
        points=selected.astype(np.float32, copy=False),
        num_source_points=num_source,
        num_voxel_points=num_voxel,
        method=method,
        voxel_size=voxel_size,
    )


def process_npz(npz_path: Path, out_path: Path, *, voxel_size: float, num_points: int, method: str, seed: int):
    data = np.load(npz_path)
    if "points_root" not in data:
        raise KeyError(f"points_root not found in {npz_path}")
    points = data["points_root"]
    result = select_points(
        points,
        voxel_size=voxel_size,
        num_points=num_points,
        method=method,
        seed=seed,
    )
    np.savez(
        out_path,
        points_root=result.points,
        source_file=str(npz_path),
        voxel_size=result.voxel_size,
        method=result.method,
        num_points=result.points.shape[0],
        num_source_points=result.num_source_points,
        num_voxel_points=result.num_voxel_points,
        seed=seed,
    )
    return result
