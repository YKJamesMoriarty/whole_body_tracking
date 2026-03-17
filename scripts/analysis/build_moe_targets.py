"""Build consolidated MoE target set from hit reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_points(report_path: Path, min_hits: int) -> np.ndarray:
    data = np.load(report_path)
    hit_counts = data["hit_counts"]
    points = data["points_root"]
    mask = hit_counts >= min_hits
    return points[mask]


def main():
    parser = argparse.ArgumentParser(description="Build MoE targets from hit reports.")
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="outputs/attack_target_reports",
        help="Directory containing *_hit_report.npz files.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="outputs/moe_targets.npz",
        help="Output NPZ file.",
    )
    parser.add_argument(
        "--swing_min_hits",
        type=int,
        default=3,
        help="Minimum hits for swing_right_head.",
    )
    parser.add_argument(
        "--default_min_hits",
        type=int,
        default=5,
        help="Minimum hits for other skills.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    out_path = Path(args.out_path)
    if not reports_dir.is_dir():
        raise FileNotFoundError(f"Reports dir not found: {reports_dir}")

    report_map = {
        "cross_right_body": reports_dir / "cross_right_body_hit_report.npz",
        "frontkick_right_body": reports_dir / "frontkick_right_body_hit_report.npz",
        "roundhouse_left_mid": reports_dir / "roundhouse_left_mid_hit_report.npz",
        "roundhouse_right_mid": reports_dir / "roundhouse_right_mid_hit_report.npz",
        "swing_right_head": reports_dir / "swing_right_head_hit_report.npz",
    }

    payload = {}
    for skill, path in report_map.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing report: {path}")
        min_hits = args.swing_min_hits if skill == "swing_right_head" else args.default_min_hits
        points = _load_points(path, min_hits)
        payload[f"{skill}_points"] = points.astype(np.float32, copy=False)
        payload[f"{skill}_min_hits"] = np.array(min_hits, dtype=np.int32)
        print(f"[INFO] {skill}: kept={points.shape[0]} (min_hits={min_hits})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()
