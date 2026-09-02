from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "PROJECT_ROADMAP.md",
    "PROJECT_SUMMARY.md",
    "ENVIRONMENT_SETUP.md",
    "merge/single_expert_merge.py",
    "merge/slerp_experiment.py",
    "eval/evaluate_models.py",
    "eval/normalized_score.py",
    "outputs/merge_results_summary.csv",
]
REQUIRED_TASK_COLUMNS = {
    "gsm8k": "exact_match,flexible-extract",
    "mmlu": "acc,none",
    "humaneval": "pass@1,create_test",
}


def check_files(root: Path) -> list[str]:
    return [path for path in REQUIRED_FILES if not (root / path).exists()]


def check_markdown_images(root: Path) -> list[str]:
    missing = []
    pattern = re.compile(r"!\[[^\]]*\]\(([^)#]+)(?:#[^)]+)?\)")
    for markdown in root.rglob("*.md"):
        for target in pattern.findall(markdown.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://")):
                continue
            if not (markdown.parent / target).exists():
                missing.append(f"{markdown.relative_to(root)} -> {target}")
    return missing


def check_csv(root: Path) -> list[str]:
    errors = []
    for path in (root / "eval_results").glob("*.csv"):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"model", "task"}.issubset(reader.fieldnames):
                errors.append(f"{path.relative_to(root)}: missing model/task columns")
                continue
            for row in reader:
                task = row.get("task", "")
                metric = REQUIRED_TASK_COLUMNS.get(task)
                if metric and metric not in reader.fieldnames:
                    errors.append(f"{path.relative_to(root)}: missing {metric} for {task}")
    summary = root / "outputs" / "merge_results_summary.csv"
    with summary.open(newline="", encoding="utf-8-sig") as handle:
        fields = set(csv.DictReader(handle).fieldnames or [])
    missing = {"method", "model", "gsm8k", "mmlu", "humaneval"} - fields
    if missing:
        errors.append(f"outputs/merge_results_summary.csv: missing {sorted(missing)}")
    return errors


def main() -> int:
    """Run repository checks that do not require model weights."""
    parser = argparse.ArgumentParser(description="Run lightweight repository integrity checks without model weights.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    args = parser.parse_args()
    root = args.root.resolve()
    errors = []
    errors.extend(f"missing file: {path}" for path in check_files(root))
    errors.extend(f"broken image: {item}" for item in check_markdown_images(root))
    errors.extend(check_csv(root))
    try:
        import yaml
        for path in (root / "merge" / "experiments").glob("*.yml"):
            yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        errors.append("PyYAML is required to validate merge YAML files")
    except Exception as exc:
        errors.append(f"invalid merge YAML: {exc}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Project integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
