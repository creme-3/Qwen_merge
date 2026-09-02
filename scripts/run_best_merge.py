from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Build and invoke the best recorded TIES merge command."""
    parser = argparse.ArgumentParser(description="Run the best recorded TIES merge configuration.")
    parser.add_argument("--config", type=Path, default=Path("configs/paths.yml"), help="Path configuration YAML")
    parser.add_argument("--write-only", action="store_true", help="Generate the recipe without running mergekit")
    parser.add_argument("--dry-run", action="store_true", help="Print the merge command without running mergekit")
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}. Copy configs/paths.example.yml to configs/paths.yml first.")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install requirements-merge.txt first") from exc

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    root = args.config.resolve().parents[1]
    models = config.get("models", {})
    required = ["math", "coder"]
    missing = [name for name in required if name not in models]
    if missing:
        raise ValueError(f"Missing model paths in config: {missing}")

    command = [
        sys.executable, str(root / "merge" / "single_expert_merge.py"),
        "--method", "ties", "--base-name", "math", "--expert", "coder",
        "--preset", "mlp", "--layers", "9-18", "--lambda", "0.05", "--density", "0.85",
        "--base", str((root / models["math"]).resolve()),
        "--expert-path", str((root / models["coder"]).resolve()),
        "--recipes-dir", str((root / config.get("recipes_dir", "merge/experiments")).resolve()),
        "--outputs-dir", str((root / config.get("merge_outputs_dir", "merge_outputs")).resolve()),
    ]
    if args.write_only:
        command.append("--write-only")
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
