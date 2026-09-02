from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def main() -> int:
    """Run the best merge and evaluation steps in sequence."""
    parser = argparse.ArgumentParser(description="Reproduce the best merge and evaluation workflow.")
    parser.add_argument("--config", type=Path, default=Path("configs/paths.yml"), help="Path configuration YAML")
    parser.add_argument("--merge-only", action="store_true", help="Stop after model merging")
    parser.add_argument("--eval-only", action="store_true", help="Skip model merging")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for evaluation")
    parser.add_argument("--offline", action="store_true", help="Use local Hugging Face caches only")
    args = parser.parse_args()
    if args.merge_only and args.eval_only:
        parser.error("--merge-only and --eval-only cannot be combined")

    root = Path(__file__).resolve().parents[1]
    merge_command = [sys.executable, str(root / "scripts" / "run_best_merge.py"), "--config", str(args.config)]
    eval_command = [sys.executable, str(root / "scripts" / "run_best_eval.py"), "--config", str(args.config)]
    if args.limit is not None:
        eval_command.extend(["--limit", str(args.limit)])
    if args.offline:
        eval_command.append("--offline")
    if not args.eval_only:
        code = run(merge_command)
        if code:
            return code
    if not args.merge_only:
        return run(eval_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
