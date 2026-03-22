"""Filter MoE target points using cross-skill hit reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _build_index(points: np.ndarray, decimals: int = 6) -> dict[tuple[float, float, float], int]:
    rounded = np.round(points, decimals=decimals)
    return {tuple(p.tolist()): i for i, p in enumerate(rounded)}


def _align_hits(
    ref_points: np.ndarray, report_points: np.ndarray, report_hits: np.ndarray, *, decimals: int = 6
) -> np.ndarray:
    """Align report hit_counts to ref_points order using rounded matching."""
    if report_points.shape == ref_points.shape and np.allclose(report_points, ref_points):
        return report_hits
    index = _build_index(report_points, decimals=decimals)
    aligned = np.zeros(ref_points.shape[0], dtype=report_hits.dtype)
    ref_rounded = np.round(ref_points, decimals=decimals)
    missing = 0
    for i, p in enumerate(ref_rounded):
        key = tuple(p.tolist())
        if key in index:
            aligned[i] = report_hits[index[key]]
        else:
            missing += 1
    if missing > 0:
        print(f"[WARN] {missing} points could not be matched; treating as zero hits.")
    return aligned


def main():
    parser = argparse.ArgumentParser(description="Filter MoE targets using cross-skill hit reports.")
    parser.add_argument(
        "--targets_file",
        type=str,
        required=True,
        help="Path to moe_targets.npz (contains <skill>_points).",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        required=True,
        help="Directory containing <target>_by_<eval>_hit_report.npz files.",
    )
    parser.add_argument(
        "--target_skill",
        type=str,
        default="swing_right_head",
        help="Skill to filter (default: swing_right_head).",
    )
    parser.add_argument(
        "--exclude_skill",
        type=str,
        default="cross_right_body",
        help="Skill whose hits should exclude target points (default: cross_right_body).",
    )
    parser.add_argument(
        "--min_target_hits",
        type=int,
        default=1,
        help="Minimum hits by target skill to keep a point.",
    )
    parser.add_argument(
        "--max_exclude_hits",
        type=int,
        default=0,
        help="Maximum hits by exclude skill to keep a point.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output NPZ path.",
    )
    args = parser.parse_args()

    targets_path = Path(args.targets_file)
    reports_dir = Path(args.reports_dir)
    out_path = Path(args.out_path)
    if not targets_path.is_file():
        raise FileNotFoundError(f"Targets file not found: {targets_path}")
    if not reports_dir.is_dir():
        raise FileNotFoundError(f"Reports dir not found: {reports_dir}")

    data = np.load(targets_path)
    target_key = f"{args.target_skill}_points"
    if target_key not in data:
        raise KeyError(f"Missing key '{target_key}' in {targets_path}")

    target_points = data[target_key]
    report_target = reports_dir / f"{args.target_skill}_by_{args.target_skill}_hit_report.npz"
    report_exclude = reports_dir / f"{args.target_skill}_by_{args.exclude_skill}_hit_report.npz"
    if not report_target.is_file():
        raise FileNotFoundError(f"Missing report: {report_target}")
    if not report_exclude.is_file():
        raise FileNotFoundError(f"Missing report: {report_exclude}")

    report_t = np.load(report_target)
    report_e = np.load(report_exclude)
    hits_target = _align_hits(target_points, report_t["points_root"], report_t["hit_counts"])
    hits_exclude = _align_hits(target_points, report_e["points_root"], report_e["hit_counts"])

    keep = (hits_target >= args.min_target_hits) & (hits_exclude <= args.max_exclude_hits)
    filtered = target_points[keep]
    removed = int((~keep).sum())

    payload = {k: data[k] for k in data.files}
    payload[target_key] = filtered.astype(np.float32, copy=False)
    payload[f"{args.target_skill}_filter_min_hits"] = np.array(args.min_target_hits, dtype=np.int32)
    payload[f"{args.target_skill}_filter_max_exclude_hits"] = np.array(args.max_exclude_hits, dtype=np.int32)
    payload[f"{args.target_skill}_filter_exclude_skill"] = np.array(args.exclude_skill)
    payload[f"{args.target_skill}_filter_removed"] = np.array(removed, dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    print(
        f"[INFO] {args.target_skill}: kept={filtered.shape[0]} removed={removed} "
        f"(exclude={args.exclude_skill}, min_hits={args.min_target_hits}, max_exclude_hits={args.max_exclude_hits})"
    )
    print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()
