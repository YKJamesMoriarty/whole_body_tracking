#!/usr/bin/env python3
"""Export AMP discriminator weights/normalizer from MimicKit checkpoints.

The exported file is a lightweight bundle for downstream integration in
whole_body_tracking:
  - disc_state_dict: discriminator-only parameters
  - disc_obs_norm_mean/std/count: normalizer statistics
  - disc_obs_dim: input dimension inferred from normalizer
"""

from __future__ import annotations

import argparse
import pathlib
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MimicKit AMP discriminator bundle.")
    parser.add_argument("--input_ckpt", type=str, required=True, help="Path to MimicKit AMP checkpoint (.pt).")
    parser.add_argument(
        "--output_bundle",
        type=str,
        default=None,
        help="Output .pt path. Default: <input_parent>/<input_stem>_disc_bundle.pt",
    )
    return parser.parse_args()


def _extract_disc_bundle(state_dict: dict) -> dict:
    disc_prefix = "_model._disc_layers."
    logit_prefix = "_model._disc_logits."
    norm_prefix = "_disc_obs_norm."

    disc_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(disc_prefix):
            disc_state_dict["disc_layers." + key[len(disc_prefix) :]] = value
        elif key.startswith(logit_prefix):
            disc_state_dict["disc_logits." + key[len(logit_prefix) :]] = value

    if len(disc_state_dict) == 0:
        raise KeyError("未找到判别器参数，checkpoint 可能不是 AMP 模型。")

    norm_count_key = norm_prefix + "_count"
    norm_mean_key = norm_prefix + "_mean"
    norm_std_key = norm_prefix + "_std"
    if norm_mean_key not in state_dict or norm_std_key not in state_dict:
        raise KeyError("未找到 _disc_obs_norm 统计量，无法导出 AMP 归一化参数。")

    norm_count = state_dict.get(norm_count_key, torch.tensor([0.0], dtype=torch.float32))
    norm_mean = state_dict[norm_mean_key]
    norm_std = state_dict[norm_std_key]

    bundle = {
        "disc_state_dict": disc_state_dict,
        "disc_obs_norm_count": norm_count,
        "disc_obs_norm_mean": norm_mean,
        "disc_obs_norm_std": norm_std,
        "disc_obs_dim": int(norm_mean.numel()),
    }
    return bundle


def main():
    args = _parse_args()
    in_path = pathlib.Path(args.input_ckpt).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"input checkpoint 不存在: {in_path}")

    out_path = (
        pathlib.Path(args.output_bundle).expanduser().resolve()
        if args.output_bundle is not None
        else in_path.with_name(f"{in_path.stem}_disc_bundle.pt")
    )

    state_dict = torch.load(str(in_path), map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint 不是 dict 格式，无法解析。")

    bundle = _extract_disc_bundle(state_dict)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, str(out_path))

    print(f"[OK] Exported AMP disc bundle: {out_path}")
    print(f"[Info] disc_obs_dim: {bundle['disc_obs_dim']}")
    print(f"[Info] disc params: {len(bundle['disc_state_dict'])} tensors")


if __name__ == "__main__":
    main()
