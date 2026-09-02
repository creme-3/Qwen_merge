from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run the best-model evaluation once for each requested seed."""
    parser = argparse.ArgumentParser(description="Repeat evaluation of the best merged model with multiple seeds.")
    parser.add_argument("--config", type=Path, default=Path("configs/paths.yml"), help="Path configuration YAML")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Evaluation seeds")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--offline", action="store_true", help="Use local Hugging Face caches only")
    args = parser.parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    root = args.config.resolve().parents[1]
    model_name = config.get("best_model")
    models_dir = (root / config.get("merge_outputs_dir", "merge_outputs")).resolve()
    output_dir = (root / config.get("eval_results_dir", "eval_results")).resolve()
    for seed in args.seeds:
        command = [
            sys.executable, str(root / "eval" / "evaluate_models.py"),
            "--scan", "--models-dir", str(models_dir), "--models", model_name,
            "--tasks", "gsm8k", "mmlu", "humaneval", "--seed", str(seed),
            "--output", str(output_dir / f"{model_name}-seed{seed}.json"),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.offline:
            command.append("--offline")
        code = subprocess.run(command, check=False).returncode
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
