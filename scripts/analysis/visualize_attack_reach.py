"""Visualize attack reach points as 3D point clouds and heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def _load_points(npz_path: Path) -> tuple[str, dict[str, np.ndarray | None]]:
    data = np.load(npz_path)
    skill_name = data["skill_name"].item() if "skill_name" in data else npz_path.stem.replace("_reach", "")
    points_root = data["points_root"]
    points_world = data["points_world"] if "points_world" in data else None
    root_pos_w = data["root_pos_w"] if "root_pos_w" in data else None
    root_quat_w = data["root_quat_w"] if "root_quat_w" in data else None
    root_pos_w0 = data["root_pos_w0"] if "root_pos_w0" in data else None
    root_quat_w0 = data["root_quat_w0"] if "root_quat_w0" in data else None
    root_pos_w_end = data["root_pos_w_end"] if "root_pos_w_end" in data else None
    root_quat_w_end = data["root_quat_w_end"] if "root_quat_w_end" in data else None
    return skill_name, {
        "root": points_root,
        "world": points_world,
        "root_pos_w": root_pos_w,
        "root_quat_w": root_quat_w,
        "root_pos_w0": root_pos_w0,
        "root_quat_w0": root_quat_w0,
        "root_pos_w_end": root_pos_w_end,
        "root_quat_w_end": root_quat_w_end,
    }


def _downsample_indices(count: int, max_points: int) -> np.ndarray | None:
    if max_points <= 0 or count <= max_points:
        return None
    return np.random.choice(count, size=max_points, replace=False)


def _apply_indices(points: np.ndarray | None, idx: np.ndarray | None) -> np.ndarray | None:
    if points is None or idx is None:
        return points
    return points[idx]


def _compute_bounds(points_list: list[np.ndarray], margin_ratio: float = 0.05):
    all_points = np.concatenate(points_list, axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    # include origin
    mins = np.minimum(mins, np.zeros(3))
    maxs = np.maximum(maxs, np.zeros(3))
    span = np.maximum(maxs - mins, 1e-3)
    mins = mins - span * margin_ratio
    maxs = maxs + span * margin_ratio
    return mins, maxs


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    # q is wxyz
    w, x, y, z = q
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float32,
    )


def _draw_pose_axes(
    ax,
    pos: np.ndarray,
    quat: np.ndarray,
    axis_scale: float,
    marker: str,
    facecolor: str,
    edgecolor: str,
    size: float = 70,
):
    r = _quat_to_rotmat(quat)
    ax.scatter(
        [pos[0]],
        [pos[1]],
        [pos[2]],
        s=size,
        marker=marker,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=1.2,
    )
    x_axis = r[:, 0] * axis_scale
    y_axis = r[:, 1] * axis_scale
    z_axis = r[:, 2] * axis_scale
    ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], color="r", length=1.0, normalize=False)
    ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], color="g", length=1.0, normalize=False)
    ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], color="b", length=1.0, normalize=False)


def _plot_root_poses(
    ax,
    root_pos_start: np.ndarray | None,
    root_quat_start: np.ndarray | None,
    root_pos_end: np.ndarray | None,
    root_quat_end: np.ndarray | None,
    root_pos_attack: np.ndarray | None,
    root_quat_attack: np.ndarray | None,
    axis_scale: float,
):
    # episode start/end
    if root_pos_start is not None and root_quat_start is not None and root_pos_start.size > 0:
        for i in range(root_pos_start.shape[0]):
            _draw_pose_axes(ax, root_pos_start[i], root_quat_start[i], axis_scale, "o", "#000000", "#000000")

    if root_pos_end is not None and root_quat_end is not None and root_pos_end.size > 0:
        for i in range(root_pos_end.shape[0]):
            _draw_pose_axes(ax, root_pos_end[i], root_quat_end[i], axis_scale, "s", "#555555", "#555555")

    # attack sampling start/end (use first/last sample)
    if root_pos_attack is not None and root_quat_attack is not None and root_pos_attack.size > 0:
        _draw_pose_axes(ax, root_pos_attack[0], root_quat_attack[0], axis_scale, "^", "none", "#f58231", size=80)
        _draw_pose_axes(ax, root_pos_attack[-1], root_quat_attack[-1], axis_scale, "D", "none", "#911eb4", size=85)


def _plot_pointcloud(
    skills: list[tuple[str, np.ndarray]],
    root_poses: list[
        tuple[
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ]
    ]
    | None,
    out_path: Path,
    point_size: float,
    alpha: float,
    title: str,
    show_root_pose: bool = False,
    pose_axis_scale: float = 0.1,
):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = [
        "#e6194B",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
    ]

    all_points = [pts for _, pts in skills if pts is not None and pts.size > 0]
    if all_points:
        mins, maxs = _compute_bounds(all_points)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

    handles = []
    for i, (name, pts) in enumerate(skills):
        if pts is None or pts.size == 0:
            continue
        c = colors[i % len(colors)]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, alpha=alpha, color=c, label=name)
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=name))

        if show_root_pose and root_poses is not None:
            (
                root_pos_w0,
                root_quat_w0,
                root_pos_w_end,
                root_quat_w_end,
                root_pos_w,
                root_quat_w,
            ) = root_poses[i]
            _plot_root_poses(
                ax,
                root_pos_w0,
                root_quat_w0,
                root_pos_w_end,
                root_quat_w_end,
                root_pos_w,
                root_quat_w,
                pose_axis_scale,
            )

    # origin
    ax.scatter([0.0], [0.0], [0.0], s=60, color="black", marker="x", label="origin")
    handles.append(Line2D([0], [0], marker="x", color="black", markersize=8, label="origin"))

    if show_root_pose:
        handles.extend(
            [
                Line2D([0], [0], marker="o", color="black", markerfacecolor="black", markersize=8, label="Episode Start"),
                Line2D([0], [0], marker="s", color="#555555", markerfacecolor="#555555", markersize=8, label="Episode End"),
                Line2D([0], [0], marker="^", color="#f58231", markerfacecolor="none", markersize=8, label="Attack Start"),
                Line2D([0], [0], marker="D", color="#911eb4", markerfacecolor="none", markersize=8, label="Attack End"),
            ]
        )

    ax.set_xlabel("x (root frame)")
    ax.set_ylabel("y (root frame)")
    ax.set_zlabel("z (root frame)")
    ax.legend(handles=handles, loc="best", fontsize=12, frameon=True)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_heatmap(
    points: np.ndarray | None,
    out_path: Path,
    bins: int,
    alpha: float,
    title: str,
    root_pos_w0: np.ndarray | None = None,
    root_quat_w0: np.ndarray | None = None,
    root_pos_w_end: np.ndarray | None = None,
    root_quat_w_end: np.ndarray | None = None,
    root_pos_w: np.ndarray | None = None,
    root_quat_w: np.ndarray | None = None,
    pose_axis_scale: float = 0.1,
):
    if points is None or points.size == 0:
        return

    hist, edges = np.histogramdd(points, bins=bins)
    idx = np.argwhere(hist > 0)
    if idx.size == 0:
        return

    centers = []
    weights = []
    for ix, iy, iz in idx:
        cx = 0.5 * (edges[0][ix] + edges[0][ix + 1])
        cy = 0.5 * (edges[1][iy] + edges[1][iy + 1])
        cz = 0.5 * (edges[2][iz] + edges[2][iz + 1])
        centers.append([cx, cy, cz])
        weights.append(hist[ix, iy, iz])

    centers = np.asarray(centers)
    weights = np.asarray(weights)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=weights,
        cmap="hot",
        s=10,
        alpha=alpha,
    )
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1, label="density")
    ax.scatter([0.0], [0.0], [0.0], s=60, color="black", marker="x", label="origin")
    _plot_root_poses(
        ax,
        root_pos_w0,
        root_quat_w0,
        root_pos_w_end,
        root_quat_w_end,
        root_pos_w,
        root_quat_w,
        pose_axis_scale,
    )
    handles = [
        Line2D([0], [0], marker="x", color="black", markersize=8, label="origin"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="black", markersize=8, label="Episode Start"),
        Line2D([0], [0], marker="s", color="#555555", markerfacecolor="#555555", markersize=8, label="Episode End"),
        Line2D([0], [0], marker="^", color="#f58231", markerfacecolor="none", markersize=8, label="Attack Start"),
        Line2D([0], [0], marker="D", color="#911eb4", markerfacecolor="none", markersize=8, label="Attack End"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=10, frameon=True)
    ax.set_xlabel("x (root frame)")
    ax.set_ylabel("y (root frame)")
    ax.set_zlabel("z (root frame)")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize attack reach points.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *_reach.npz files.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save images.")
    parser.add_argument("--max_points", type=int, default=20000, help="Max points per skill for point cloud.")
    parser.add_argument("--bins", type=int, default=30, help="Number of bins per axis for heatmap.")
    parser.add_argument("--point_size", type=float, default=1.0, help="Point size for point cloud.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha for scatter plots.")
    parser.add_argument("--pose_axis_scale", type=float, default=0.14, help="Axis length for root pose axes.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*_reach.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_reach.npz files found in {input_dir}")

    skills_root: list[tuple[str, np.ndarray]] = []
    skills_world: list[tuple[str, np.ndarray]] = []
    root_poses_world: list[
        tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]
    ] = []
    for npz_path in npz_files:
        name, points = _load_points(npz_path)
        idx = _downsample_indices(points["root"].shape[0], args.max_points)
        points_root = _apply_indices(points["root"], idx)
        points_world = _apply_indices(points["world"], idx)
        root_pos_w0 = points["root_pos_w0"]
        root_quat_w0 = points["root_quat_w0"]
        root_pos_w_end = points["root_pos_w_end"]
        root_quat_w_end = points["root_quat_w_end"]
        root_pos_w = points["root_pos_w"]
        root_quat_w = points["root_quat_w"]
        skills_root.append((name, points_root))
        skills_world.append((name, points_world))
        root_poses_world.append((root_pos_w0, root_quat_w0, root_pos_w_end, root_quat_w_end, root_pos_w, root_quat_w))

    _plot_pointcloud(
        skills_root,
        None,
        out_dir / "attack_reach_pointcloud_root.png",
        args.point_size,
        args.alpha,
        "Attack Reach Point Cloud (root frame)",
    )
    _plot_pointcloud(
        skills_world,
        root_poses_world,
        out_dir / "attack_reach_pointcloud_world.png",
        args.point_size,
        args.alpha,
        "Attack Reach Point Cloud (world frame)",
        show_root_pose=True,
        pose_axis_scale=args.pose_axis_scale,
    )

    for name, points in skills_root:
        heat_out = out_dir / f"{name}_heatmap_root.png"
        _plot_heatmap(points, heat_out, args.bins, args.alpha, f"{name} heatmap (root frame)")

    for (name, points), (root_pos_w0, root_quat_w0, root_pos_w_end, root_quat_w_end, root_pos_w, root_quat_w) in zip(
        skills_world, root_poses_world
    ):
        heat_out = out_dir / f"{name}_heatmap_world.png"
        _plot_heatmap(
            points,
            heat_out,
            args.bins,
            args.alpha,
            f"{name} heatmap (world frame)",
            root_pos_w0=root_pos_w0,
            root_quat_w0=root_quat_w0,
            root_pos_w_end=root_pos_w_end,
            root_quat_w_end=root_quat_w_end,
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            pose_axis_scale=args.pose_axis_scale,
        )

    print(f"[INFO] Saved point clouds: {out_dir / 'attack_reach_pointcloud_root.png'}")
    print(f"[INFO] Saved point clouds: {out_dir / 'attack_reach_pointcloud_world.png'}")
    print(f"[INFO] Saved per-skill heatmaps in: {out_dir}")


if __name__ == "__main__":
    main()
