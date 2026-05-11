#!/usr/bin/env python3
"""Inspect saved LIBERO rollout latent-feature pickle files."""

import argparse
import pickle
from pathlib import Path

import torch


EXPECTED_OPENVLA_32_LAYERS = [1, 4, 8, 12, 16, 20, 24, 28, 32]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def inspect_file(path: Path) -> bool:
    ok = True
    rollout = load_pickle(path)

    hidden_states = rollout.get("hidden_states")
    if hidden_states is None:
        print(f"[FAIL] {path}: missing `hidden_states`")
        return False

    if isinstance(hidden_states, list):
        hidden_states = torch.stack(hidden_states, dim=0)
    elif not isinstance(hidden_states, torch.Tensor):
        hidden_states = torch.as_tensor(hidden_states)

    layers = rollout.get("hidden_state_layers")
    dim_per_layer = rollout.get("hidden_state_dim_per_layer")
    expected_feature_dim = None
    if layers is not None and dim_per_layer is not None:
        expected_feature_dim = len(layers) * int(dim_per_layer)

    print(f"\n{path}")
    print(f"  hidden_states shape: {tuple(hidden_states.shape)}")
    print(f"  dtype: {hidden_states.dtype}")
    print(f"  layers: {layers}")
    print(f"  dim_per_layer: {dim_per_layer}")
    print(f"  layout: {rollout.get('hidden_state_layout')}")
    print(f"  task_id: {rollout.get('task_id')}")
    print(f"  episode_success: {rollout.get('episode_success')}")
    print(f"  mp4_path in pickle: {rollout.get('mp4_path')}")

    if hidden_states.ndim != 3:
        print("  [FAIL] expected hidden_states to have shape (T, action_tokens, feature_dim)")
        ok = False

    if hidden_states.shape[0] <= 0:
        print("  [FAIL] no rollout timesteps saved")
        ok = False

    if hidden_states.shape[-1] <= 0:
        print("  [FAIL] empty feature dimension")
        ok = False

    if expected_feature_dim is not None and hidden_states.shape[-1] != expected_feature_dim:
        print(f"  [FAIL] feature dim {hidden_states.shape[-1]} != {expected_feature_dim} from metadata")
        ok = False

    if layers is not None and layers != EXPECTED_OPENVLA_32_LAYERS:
        print(f"  [WARN] layers differ from expected OpenVLA-32 selection: {EXPECTED_OPENVLA_32_LAYERS}")

    if not torch.isfinite(hidden_states).all():
        print("  [FAIL] hidden_states contains NaN or inf")
        ok = False

    if torch.count_nonzero(hidden_states).item() == 0:
        print("  [FAIL] hidden_states is all zeros")
        ok = False

    print(f"  min/max: {hidden_states.min().item():.4f} / {hidden_states.max().item():.4f}")
    print(f"  mean/std: {hidden_states.float().mean().item():.4f} / {hidden_states.float().std().item():.4f}")

    csv_path = path.with_suffix(".csv")
    mp4_path = path.with_suffix(".mp4")
    print(f"  matching csv: {csv_path.exists()} ({csv_path})")
    print(f"  matching mp4: {mp4_path.exists()} ({mp4_path})")
    if not csv_path.exists():
        ok = False
        print("  [FAIL] missing matching .csv log")
    if not mp4_path.exists():
        ok = False
        print("  [FAIL] missing matching .mp4 video")

    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def find_pickle_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.pkl"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Saved rollout .pkl file or directory containing rollout .pkl files")
    parser.add_argument("--max-files", type=int, default=None, help="Inspect at most this many pickle files")
    args = parser.parse_args()

    paths = find_pickle_files(args.path)
    if args.max_files is not None:
        paths = paths[: args.max_files]

    if not paths:
        print(f"No .pkl files found under {args.path}")
        return 1

    passed = 0
    for path in paths:
        if inspect_file(path):
            passed += 1

    print(f"\nSummary: {passed}/{len(paths)} files passed")
    return 0 if passed == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
