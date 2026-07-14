from __future__ import annotations

import argparse
import csv
from pathlib import Path


TASK_METRICS = {
    "gsm8k": ["exact_match,flexible-extract", "exact_match,strict-match"],
    "mmlu": ["acc,none"],
    "humaneval": ["pass@1,create_test", "pass@1,none"],
}


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def metric_value(row: dict[str, str]) -> float | None:
    task = row.get("task", "")
    for metric_name in TASK_METRICS.get(task, []):
        value = parse_float(row.get(metric_name))
        if value is not None:
            return value
    return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def collect_scores(results_dir: Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for csv_path in sorted(results_dir.glob("*.csv")):
        for row in read_csv_rows(csv_path):
            model = row.get("model")
            task = row.get("task")
            value = metric_value(row)
            if not model or not task or value is None:
                continue
            scores.setdefault(model, {})[task] = value
    return scores


def build_base_scores(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    if "base" not in scores:
        raise ValueError("Could not find model named 'base' in CSV results")
    missing = [task for task in TASK_METRICS if task not in scores["base"]]
    if missing:
        raise ValueError(f"Base scores are missing tasks: {missing}")
    return {task: scores["base"][task] for task in TASK_METRICS}


def normalized_total(model_scores: dict[str, float], base_scores: dict[str, float]) -> float | None:
    if any(task not in model_scores for task in TASK_METRICS):
        return None
    return sum(model_scores[task] / base_scores[task] for task in TASK_METRICS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize normalized GSM8K/MMLU/HumanEval scores against the base model.")
    parser.add_argument("--results-dir", type=Path, default=Path("eval_results"), help="Directory containing evaluation CSV files")
    args = parser.parse_args()

    scores = collect_scores(args.results_dir)
    base_scores = build_base_scores(scores)

    rows = []
    for model, model_scores in scores.items():
        total = normalized_total(model_scores, base_scores)
        if total is None:
            continue
        rows.append(
            {
                "model": model,
                "score": total,
                "gsm8k": model_scores["gsm8k"],
                "mmlu": model_scores["mmlu"],
                "humaneval": model_scores["humaneval"],
            }
        )

    rows.sort(key=lambda row: row["score"], reverse=True)
    print("model,normalized_total,gsm8k,mmlu,humaneval")
    for row in rows:
        print(
            f"{row['model']},{row['score']:.6f},"
            f"{row['gsm8k']:.6f},{row['mmlu']:.6f},{row['humaneval']:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
