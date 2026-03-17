"""Verify that points_root and points_world are consistent via root pose."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion(s) q (wxyz)."""
    # q: (N,4), v: (N,3)
    q_vec = q[:, 1:4]
    w = q[:, 0:1]
    t = 2.0 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)


def verify_file(npz_path: Path, tol: float) -> bool:
    data = np.load(npz_path)
    if "points_root" not in data or "points_world" not in data:
        print(f"[WARN] Missing points_root/points_world in {npz_path}")
        return False
    if "root_pos_w" not in data or "root_quat_w" not in data:
        print(f"[WARN] Missing root_pos_w/root_quat_w in {npz_path}")
        return False

    points_root = data["points_root"]
    points_world = data["points_world"]
    root_pos_w = data["root_pos_w"]
    root_quat_w = data["root_quat_w"]

    if points_root.shape[0] != points_world.shape[0] or points_root.shape[0] != root_pos_w.shape[0]:
        print(f"[WARN] Shape mismatch in {npz_path}")
        print(f"       points_root: {points_root.shape}, points_world: {points_world.shape}, root_pos_w: {root_pos_w.shape}")
        return False

    pred_world = root_pos_w + quat_apply(root_quat_w, points_root)
    diff = np.linalg.norm(pred_world - points_world, axis=1)
    max_err = float(diff.max()) if diff.size else 0.0
    mean_err = float(diff.mean()) if diff.size else 0.0

    ok = max_err <= tol
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {npz_path.name}: mean_err={mean_err:.6f}, max_err={max_err:.6f}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify reach point transforms.")
    parser.add_argument("--input", type=str, required=True, help="NPZ file or directory with *_reach.npz files.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Max allowed error.")
    args = parser.parse_args()

    path = Path(args.input)
    if path.is_file():
        verify_file(path, args.tol)
        return

    files = sorted(path.glob("*_reach.npz"))
    if not files:
        raise FileNotFoundError(f"No *_reach.npz files found in {path}")

    all_ok = True
    for f in files:
        if not verify_file(f, args.tol):
            all_ok = False

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
