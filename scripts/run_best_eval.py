from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Build and invoke evaluation for the best merged checkpoint."""
    parser = argparse.ArgumentParser(description="Evaluate the best recorded merged model.")
    parser.add_argument("--config", type=Path, default=Path("configs/paths.yml"), help="Path configuration YAML")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for a smoke test")
    parser.add_argument("--offline", action="store_true", help="Use local Hugging Face caches only")
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}. Copy configs/paths.example.yml to configs/paths.yml first.")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install requirements-eval.txt first") from exc

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    root = args.config.resolve().parents[1]
    model_name = config.get("best_model")
    if not model_name:
        raise ValueError("best_model is missing from the path config")
    models_dir = (root / config.get("merge_outputs_dir", "merge_outputs")).resolve()
    output_dir = (root / config.get("eval_results_dir", "eval_results")).resolve()
    command = [
        sys.executable, str(root / "eval" / "evaluate_models.py"),
        "--scan", "--models-dir", str(models_dir), "--models", model_name,
        "--tasks", "gsm8k", "mmlu", "humaneval",
        "--output", str(output_dir / f"{model_name}-all.json"),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.offline:
        command.append("--offline")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
