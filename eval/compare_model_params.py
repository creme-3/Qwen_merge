from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


@dataclass
class DiffStats:
    l2_sum: float = 0.0
    l2_base_sum: float = 0.0
    max_abs: float = 0.0
    count: int = 0

    def update(self, diff: torch.Tensor, base: torch.Tensor) -> None:
        diff_flat = diff.float().flatten()
        base_flat = base.float().flatten()
        self.l2_sum += float(torch.sum(diff_flat * diff_flat))
        self.l2_base_sum += float(torch.sum(base_flat * base_flat))
        self.max_abs = max(self.max_abs, float(torch.max(torch.abs(diff_flat))))
        self.count += diff_flat.numel()

    def l2_ratio(self) -> float:
        if self.l2_base_sum == 0.0:
            return float("nan")
        return (self.l2_sum ** 0.5) / (self.l2_base_sum ** 0.5)

    def mean_abs(self) -> float:
        if self.count == 0:
            return 0.0
        return (self.l2_sum / self.count) ** 0.5


def iter_keys(path: Path) -> Iterable[str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return list(handle.keys())


def compare_models(base_path: Path, other_path: Path, include_pattern: str | None) -> DiffStats:
    stats = DiffStats()
    with safe_open(base_path, framework="pt", device="cpu") as base_handle, safe_open(other_path, framework="pt", device="cpu") as other_handle:
        base_keys = set(base_handle.keys())
        other_keys = set(other_handle.keys())
        missing = base_keys - other_keys
        extra = other_keys - base_keys
        if missing or extra:
            raise ValueError(f"Tensor keys mismatch. Missing={len(missing)}, Extra={len(extra)}")

        for key in base_keys:
            if include_pattern and (include_pattern not in key):
                continue
            base_tensor = base_handle.get_tensor(key)
            other_tensor = other_handle.get_tensor(key)
            if base_tensor.shape != other_tensor.shape:
                raise ValueError(f"Shape mismatch for {key}: {base_tensor.shape} vs {other_tensor.shape}")
            diff = other_tensor - base_tensor
            stats.update(diff, base_tensor)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare model parameter differences between two safetensors.")
    parser.add_argument("--base", type=Path, required=True, help="Path to base model.safetensors")
    parser.add_argument("--other", type=Path, required=True, help="Path to merged model.safetensors")
    parser.add_argument("--only-mlp", action="store_true", help="Only compare MLP-related tensors")
    args = parser.parse_args()

    pattern = None
    if args.only_mlp:
        pattern = "mlp"

    stats = compare_models(args.base, args.other, pattern)
    print(f"Compared tensors: {stats.count} parameters")
    print(f"L2 ratio (||delta|| / ||base||): {stats.l2_ratio():.6f}")
    print(f"Max abs diff: {stats.max_abs:.6f}")
    print(f"RMS diff per param: {stats.mean_abs():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
