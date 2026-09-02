from __future__ import annotations

import argparse
import csv
from pathlib import Path


TASK_METRICS = {
    "gsm8k": "exact_match,flexible-extract",
    "mmlu": "acc,none",
    "humaneval": "pass@1,create_test",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def collect_scores(results_dir: Path) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for path in sorted(results_dir.glob("*.csv")):
        for row in read_rows(path):
            model, task = row.get("model"), row.get("task")
            metric = TASK_METRICS.get(task or "")
            if not model or not task or not metric or not row.get(metric):
                continue
            value = float(row[metric])
            key = (model, task)
            previous = scores.get(key)
            if previous is not None and abs(previous - value) > 1e-9:
                raise ValueError(f"Conflicting values for {model}/{task}: {previous} vs {value} in {path}")
            scores[key] = value
    return scores


def main() -> int:
    """Validate raw evaluation metrics against the committed summary."""
    parser = argparse.ArgumentParser(description="Check evaluation CSV consistency and the committed result summary.")
    parser.add_argument("--results-dir", type=Path, default=Path("eval_results"), help="Evaluation CSV directory")
    parser.add_argument("--summary", type=Path, default=Path("outputs/merge_results_summary.csv"), help="Committed summary CSV")
    args = parser.parse_args()
    scores = collect_scores(args.results_dir)
    missing_base = [task for task in TASK_METRICS if ("base", task) not in scores]
    if missing_base:
        raise ValueError(f"Base model is missing metrics: {missing_base}")
    summary_rows = read_rows(args.summary)
    required = {"method", "model", "gsm8k", "mmlu", "humaneval"}
    if not summary_rows or not required.issubset(summary_rows[0]):
        raise ValueError(f"Summary must contain columns: {sorted(required)}")
    unverifiable = []
    for row in summary_rows:
        model = row["model"]
        if not any((model, task) in scores for task in TASK_METRICS):
            unverifiable.append(model)
            continue
        for task, column in (("gsm8k", "gsm8k"), ("mmlu", "mmlu"), ("humaneval", "humaneval")):
            expected = scores.get((model, task))
            if expected is None:
                unverifiable.append(f"{model}/{task}")
                continue
            actual = float(row[column])
            if abs(actual - expected) > 1e-6:
                raise ValueError(f"Summary mismatch for {model}/{task}: {actual} vs {expected}")
    best = "ties_math_plus_coder_mlp_layers_9_18_lam_0p05_dens_0p85"
    if not all((best, task) in scores for task in TASK_METRICS):
        raise ValueError(f"Best model is missing one or more task metrics: {best}")
    print(f"Checked {len(scores)} model/task scores and {len(summary_rows)} summary rows.")
    if unverifiable:
        print(f"Skipped {len(unverifiable)} summary entries without matching raw CSV rows.")
    print(f"Best model metrics: {best} | GSM8K={scores[(best, 'gsm8k')]:.4f} | MMLU={scores[(best, 'mmlu')]:.4f} | HumanEval={scores[(best, 'humaneval')]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
