"""Stage 3 打标结果可视化脚本

将目标空间内各采样点的最优技能标签以 3D 散点图呈现。
不同颜色代表不同技能，直观展示每个空间位置应该使用哪种进攻技能。

使用方法:
    python scripts/rsl_rl/visualize_labels.py --labels_dir labels/

    # 只看命中率矩阵热图（不显示 3D 图）:
    python scripts/rsl_rl/visualize_labels.py --labels_dir labels/ --mode heatmap

    # 过滤掉命中率低于阈值的点:
    python scripts/rsl_rl/visualize_labels.py --labels_dir labels/ --min_accuracy 0.2
"""

import argparse
import json
import pathlib

import numpy as np


# ====================================================================
# 技能颜色表 (与 SKILL_CONFIGS 的 skill_id 对应)
# ====================================================================
SKILL_COLORS = {
    0: ("#4878CF", "cross (右直拳)"),
    1: ("#6ACC65", "swing (右摆拳)"),
    3: ("#D65F5F", "roundhouse (右高位鞭腿)"),
    4: ("#B47CC7", "frontkick (右脚正蹬)"),
    5: ("#C4AD66", "stance (防守/无目标)"),
    # 未来新技能可继续添加
}
DEFAULT_COLOR = ("#888888", "unknown")


def load_labels(labels_dir: pathlib.Path):
    """加载打标结果文件"""
    grid_points    = np.load(labels_dir / "grid_points.npy")     # (N, 3)
    all_accuracies = np.load(labels_dir / "all_accuracies.npy")  # (N, K)
    final_labels   = np.load(labels_dir / "labels.npy")          # (N,)

    meta_path = labels_dir / "metadata.json"
    metadata  = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    print(f"[可视化] 加载 {len(grid_points)} 个采样点，{all_accuracies.shape[1]} 个技能")
    return grid_points, all_accuracies, final_labels, metadata


def plot_3d_labels(
    grid_points: np.ndarray,
    final_labels: np.ndarray,
    all_accuracies: np.ndarray,
    metadata: dict,
    min_accuracy: float = 0.0,
    save_path: pathlib.Path | None = None,
):
    """绘制 3D 散点图：每个点的颜色代表最优技能"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("[可视化] 需要安装 matplotlib: pip install matplotlib")
        return

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")

    # 按技能分组绘制
    unique_ids = sorted(set(final_labels.tolist()))
    handles    = []

    for skill_id in unique_ids:
        mask = (final_labels == skill_id)

        # 过滤低命中率点（stance 标签不过滤）
        if min_accuracy > 0.0 and skill_id != 5:
            max_acc = all_accuracies[mask].max(axis=1)
            mask = mask & np.zeros(len(mask), dtype=bool)  # 重建 mask
            # 重新计算
            base_mask = (final_labels == skill_id)
            max_acc   = all_accuracies[base_mask].max(axis=1)
            keep      = max_acc >= min_accuracy
            indices   = np.where(base_mask)[0][keep]
            if len(indices) == 0:
                continue
            pts = grid_points[indices]
        else:
            pts = grid_points[mask]

        if len(pts) == 0:
            continue

        color, label = SKILL_COLORS.get(skill_id, DEFAULT_COLOR)
        sc = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=color, label=f"[{skill_id}] {label}",
            s=30, alpha=0.7, edgecolors="none",
        )
        handles.append(sc)

    # 标注轴
    ax.set_xlabel("X — 前方 (m)", fontsize=11)
    ax.set_ylabel("Y — 左右 (m)", fontsize=11)
    ax.set_zlabel("Z — 高低 (m)", fontsize=11)  # type: ignore[attr-defined]
    ax.set_title(
        f"目标空间技能分布图\n"
        f"采样间隔 {metadata.get('grid_spacing', '?')}m | "
        f"{metadata.get('n_episodes', '?')} 次/点 | "
        f"{len(grid_points)} 个采样点",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=9)

    # 添加机器人站位示意（原点）
    ax.scatter([0], [0], [0], c="black", s=200, marker="*", label="机器人原点 (Pelvis)", zorder=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[可视化] 3D 图保存至: {save_path}")
    plt.show()


def plot_heatmaps(
    grid_points: np.ndarray,
    all_accuracies: np.ndarray,
    metadata: dict,
    skills_tested: list[str],
    save_dir: pathlib.Path | None = None,
):
    """绘制每个技能的命中率分布热图（按 x-z 平面投影，y 取均值）"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("[可视化] 需要安装 matplotlib")
        return

    spacing = metadata.get("grid_spacing", 0.05)
    half    = spacing / 2

    x_vals = np.arange(0.0,  1.2  + half, spacing)
    z_vals = np.arange(-0.4, 0.4  + half, spacing)

    n_skills = len(skills_tested)
    fig, axes = plt.subplots(1, n_skills, figsize=(5 * n_skills, 5))
    if n_skills == 1:
        axes = [axes]

    for k, skill_name in enumerate(skills_tested):
        acc_k = all_accuracies[:, k]  # (N,)

        # 对 y 轴取平均，投影到 x-z 平面
        heatmap = np.zeros((len(z_vals), len(x_vals)))
        counts  = np.zeros_like(heatmap)
        for i, (x, y, z) in enumerate(grid_points):
            xi = int(round((x - 0.0)  / spacing))
            zi = int(round((z - (-0.4)) / spacing))
            xi = min(xi, len(x_vals) - 1)
            zi = min(zi, len(z_vals) - 1)
            heatmap[zi, xi] += acc_k[i]
            counts[zi, xi]  += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            heatmap = np.where(counts > 0, heatmap / counts, 0)

        im = axes[k].imshow(
            heatmap, origin="lower", cmap="RdYlGn",
            vmin=0.0, vmax=1.0,
            extent=[x_vals[0], x_vals[-1], z_vals[0], z_vals[-1]],
            aspect="auto",
        )
        axes[k].set_title(f"{skill_name}\n(y 轴平均)", fontsize=11)
        axes[k].set_xlabel("X 前方 (m)")
        axes[k].set_ylabel("Z 高低 (m)")
        plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04, label="命中率")

    fig.suptitle("各技能命中率热图 (x-z 投影，y 均值)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_dir:
        save_path = save_dir / "heatmaps.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[可视化] 热图保存至: {save_path}")
    plt.show()


def print_statistics(
    grid_points: np.ndarray,
    all_accuracies: np.ndarray,
    final_labels: np.ndarray,
    metadata: dict,
    skills_tested: list[str],
):
    """打印命中率统计信息"""
    from whole_body_tracking.tasks.tracking.stage3.skill_config import SKILL_CONFIGS, STANCE_SKILL_ID

    n_points = len(grid_points)
    print(f"\n{'='*55}")
    print(f"打标结果统计 ({n_points} 个采样点)")
    print(f"{'='*55}")
    print(f"{'技能':<12} {'平均命中率':>10} {'命中率>0':>10} {'被选为最优':>12}")
    print(f"{'-'*55}")

    for k, skill_name in enumerate(skills_tested):
        acc_k   = all_accuracies[:, k]
        mean_a  = acc_k.mean()
        nonzero = np.sum(acc_k > 0)
        sid     = SKILL_CONFIGS[skill_name]["skill_id"]
        n_best  = np.sum(final_labels == sid)
        print(f"{skill_name:<12} {mean_a:>10.3f} {nonzero:>10d} {n_best:>12d}")

    # stance 兜底
    n_stance = np.sum(final_labels == STANCE_SKILL_ID)
    print(f"{'stance(兜底)':<12} {'---':>10} {'---':>10} {n_stance:>12d}")
    print(f"{'='*55}\n")


# ====================================================================
# 入口
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3 打标结果可视化")
    parser.add_argument("--labels_dir", type=str, default="labels", help="label_skills.py 输出目录")
    parser.add_argument(
        "--mode", type=str, default="3d",
        choices=["3d", "heatmap", "both"],
        help="可视化模式: 3d=3D散点图, heatmap=命中率热图, both=两者都显示"
    )
    parser.add_argument("--min_accuracy", type=float, default=0.0, help="过滤命中率低于此值的点（0=不过滤）")
    parser.add_argument("--save", action="store_true", help="将图表保存到 labels_dir/")
    args = parser.parse_args()

    labels_dir = pathlib.Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"目录不存在: {labels_dir}，请先运行 label_skills.py")

    grid_points, all_accuracies, final_labels, metadata = load_labels(labels_dir)
    skills_tested = metadata.get("skills_tested", [f"skill_{k}" for k in range(all_accuracies.shape[1])])

    print_statistics(grid_points, all_accuracies, final_labels, metadata, skills_tested)

    save_dir = labels_dir if args.save else None

    if args.mode in ("3d", "both"):
        plot_3d_labels(
            grid_points, final_labels, all_accuracies, metadata,
            min_accuracy=args.min_accuracy,
            save_path=(save_dir / "label_3d.png") if save_dir else None,
        )

    if args.mode in ("heatmap", "both"):
        plot_heatmaps(
            grid_points, all_accuracies, metadata, skills_tested,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
