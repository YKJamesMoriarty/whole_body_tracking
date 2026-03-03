"""Trim motion.npz by frame range, save locally, and upload to WandB registry.

Default output file name is:
    trim_{source_motion_name}.npz

For example, input:
    artifacts/stance_orthodox_idle_normal_2_100:v0/motion.npz
will create:
    iros_motion/npz/trim_stance_orthodox_idle_normal_2_100.npz
and upload artifact name:
    trim_stance_orthodox_idle_normal_2_100
"""

import argparse
import pathlib
import re
import tempfile

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Trim motion.npz by frame range and upload to WandB.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to source motion.npz.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index (inclusive).")
    parser.add_argument("--end_frame", type=int, required=True, help="End frame index (inclusive).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="iros_motion/npz",
        help="Directory to save trimmed npz.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output artifact/file base name without extension. Defaults to trim_{source_motion_name}.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="csv_to_npz",
        help="WandB project name used for upload.",
    )
    parser.add_argument(
        "--registry_type",
        type=str,
        default="motions",
        help="WandB artifact type and registry collection suffix.",
    )
    parser.add_argument(
        "--no_upload",
        action="store_true",
        default=False,
        help="Skip WandB upload (local save only).",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Use np.savez_compressed for smaller output.",
    )
    return parser.parse_args()


def infer_source_motion_name(input_file: pathlib.Path) -> str:
    # Prefer parent folder name when file is the default "motion.npz" downloaded from WandB.
    if input_file.name == "motion.npz":
        raw_name = input_file.parent.name
    else:
        raw_name = input_file.stem

    # Strip artifact alias suffix, e.g. "stance_xxx:v0" -> "stance_xxx".
    source_name = raw_name.split(":", 1)[0]
    source_name = source_name.strip()
    if not source_name:
        source_name = "motion"

    # Keep names filesystem-safe and registry-safe.
    source_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name).strip("._-")
    return source_name or "motion"


def main():
    args = parse_args()
    input_file = pathlib.Path(args.input_file).expanduser().resolve()

    if not input_file.is_file():
        raise FileNotFoundError(f"Invalid --input_file: {input_file}")

    source_motion_name = infer_source_motion_name(input_file)
    output_name = args.output_name or f"trim_{source_motion_name}"
    output_name = re.sub(r"[^A-Za-z0-9._-]+", "_", output_name).strip("._-")
    if not output_name:
        raise ValueError("Invalid output name after sanitization.")

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_file = output_dir / f"{output_name}.npz"

    data = np.load(str(input_file))
    required_keys = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    for key in required_keys:
        if key not in data.files:
            raise KeyError(f"Required key missing in input npz: {key}")

    total_frames = int(data["joint_pos"].shape[0])
    start_frame = args.start_frame
    end_frame = args.end_frame
    if start_frame < 0:
        raise ValueError("--start_frame must be >= 0.")
    if end_frame < 0:
        raise ValueError("--end_frame must be >= 0.")
    if start_frame > end_frame:
        raise ValueError("--start_frame must be <= --end_frame.")
    if end_frame >= total_frames:
        raise ValueError(f"--end_frame must be <= {total_frames - 1}, got {end_frame}.")

    trimmed = {}
    slice_end = end_frame + 1
    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == total_frames and key != "fps":
            trimmed[key] = arr[start_frame:slice_end]
        else:
            trimmed[key] = arr

    trimmed["trim_start_frame"] = np.array([start_frame], dtype=np.int64)
    trimmed["trim_end_frame"] = np.array([end_frame], dtype=np.int64)
    trimmed["source_motion_name"] = np.array([source_motion_name])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if args.compress else np.savez
    save_fn(str(output_file), **trimmed)

    fps_value = float(np.array(data["fps"]).reshape(-1)[0])
    trimmed_frames = int(trimmed["joint_pos"].shape[0])
    original_duration = (total_frames - 1) / fps_value
    trimmed_duration = (trimmed_frames - 1) / fps_value

    print(f"[INFO] input={input_file}")
    print(f"[INFO] output={output_file}")
    print(
        f"[INFO] trim_range=[{start_frame}, {end_frame}] "
        f"frames={trimmed_frames}/{total_frames} fps={fps_value:.3f}"
    )
    print(f"[INFO] duration: original={original_duration:.3f}s trimmed={trimmed_duration:.3f}s")

    if args.no_upload:
        print("[INFO] WandB upload skipped (--no_upload).")
        return

    import wandb

    run = wandb.init(project=args.wandb_project, name=output_name)
    print(f"[INFO] Uploading artifact to WandB: {output_name}")
    # Keep artifact internal file name as "motion.npz" for compatibility with replay/train scripts.
    with tempfile.TemporaryDirectory(prefix="trim_motion_upload_") as tmp_dir:
        upload_motion_file = pathlib.Path(tmp_dir) / "motion.npz"
        save_fn(str(upload_motion_file), **trimmed)
        logged_artifact = run.log_artifact(artifact_or_path=str(upload_motion_file), name=output_name, type=args.registry_type)
    run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{args.registry_type}/{output_name}")
    print(f"[INFO] Uploaded and linked: wandb-registry-{args.registry_type}/{output_name}")
    run.finish()


if __name__ == "__main__":
    main()
