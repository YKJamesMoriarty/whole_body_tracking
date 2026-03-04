#!/usr/bin/env python3
"""Build a MimicKit motion-list YAML from a folder of .pkl motions."""

from __future__ import annotations

import argparse
import pathlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MimicKit motion yaml from pkl folder.")
    parser.add_argument("--motion_dir", type=str, required=True, help="Directory containing .pkl motion files.")
    parser.add_argument("--output_yaml", type=str, required=True, help="Output yaml path.")
    parser.add_argument("--weight", type=float, default=1.0, help="Uniform weight for each motion.")
    return parser.parse_args()


def main():
    args = parse_args()
    motion_dir = pathlib.Path(args.motion_dir).expanduser().resolve()
    output_yaml = pathlib.Path(args.output_yaml).expanduser().resolve()
    if not motion_dir.exists():
        raise FileNotFoundError(f"motion_dir 不存在: {motion_dir}")

    files = sorted(motion_dir.glob("*.pkl"))
    if len(files) == 0:
        raise ValueError(f"{motion_dir} 下未找到 .pkl 文件。")

    motions = [{"file": str(p), "weight": float(args.weight)} for p in files]
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        f.write("motions:\n")
        for motion in motions:
            f.write(f"  - file: \"{motion['file']}\"\n")
            f.write(f"    weight: {motion['weight']}\n")

    print(f"[OK] Saved: {output_yaml}")
    print(f"[Info] motions: {len(motions)}")


if __name__ == "__main__":
    main()
