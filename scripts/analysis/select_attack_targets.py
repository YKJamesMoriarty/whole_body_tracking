"""Select target points per skill from collected reach points."""

from __future__ import annotations

import argparse
from pathlib import Path

import importlib.util
import sys


def _load_selector():
    """Load selector module by path to avoid importing full IsaacLab stack."""
    module_path = Path(__file__).resolve().parents[2] / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "analysis" / "attack_target_selection.py"
    spec = importlib.util.spec_from_file_location("attack_target_selection", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Select target points from reach data.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with *_reach.npz files.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for selected targets.")
    parser.add_argument("--voxel_size", type=float, default=0.03, help="Voxel size for downsampling.")
    parser.add_argument("--num_points", type=int, default=400, help="Number of points to keep per skill.")
    parser.add_argument(
        "--method",
        type=str,
        default="fps",
        choices=["fps", "random"],
        help="Sampling method after voxel downsampling.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    args = parser.parse_args()

    selector = _load_selector()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*_reach.npz"))
    if not files:
        raise FileNotFoundError(f"No *_reach.npz files found in {input_dir}")

    for npz_path in files:
        skill_name = npz_path.stem.replace("_reach", "")
        out_path = out_dir / f"{skill_name}_targets_root.npz"
        result = selector.process_npz(
            npz_path,
            out_path,
            voxel_size=args.voxel_size,
            num_points=args.num_points,
            method=args.method,
            seed=args.seed,
        )
        print(
            f"[INFO] {skill_name}: source={result.num_source_points}, "
            f"voxel={result.num_voxel_points} (voxel_size={args.voxel_size}), "
            f"kept={result.points.shape[0]}"
        )


if __name__ == "__main__":
    main()
