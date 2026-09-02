from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    """Export the Markdown report when Pandoc is available."""
    parser = argparse.ArgumentParser(description="Export the Markdown experiment report to PDF with Pandoc.")
    parser.add_argument("--input", type=Path, default=Path("outputs/merge_experiment_report.md"), help="Markdown report")
    parser.add_argument("--output", type=Path, default=Path("outputs/merge_experiment_report.pdf"), help="PDF output")
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Report not found: {args.input}")
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("Pandoc is required for PDF export. Install Pandoc and a LaTeX engine, then rerun this command.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run([pandoc, str(args.input), "-o", str(args.output)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
