from __future__ import annotations

import argparse
import csv
from pathlib import Path


TASK_COLUMNS = ("gsm8k", "mmlu", "humaneval")
COLORS = {
    "gsm8k": "#4C78A8",
    "mmlu": "#F58518",
    "humaneval": "#54A24B",
    "core_score": "#B279A2",
    "Baseline": "#666666",
    "Task Arithmetic": "#4C78A8",
    "TIES": "#E45756",
    "SLERP": "#54A24B",
}


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = dict(row)
            for column in TASK_COLUMNS:
                parsed[column] = float(row[column])
            rows.append(parsed)
    return rows


def enrich_scores(rows: list[dict[str, object]]) -> None:
    base = get_model(rows, "base")
    for row in rows:
        row["core_score"] = float(row["gsm8k"]) / float(base["gsm8k"]) + float(row["humaneval"]) / float(base["humaneval"])
        row["score_3task"] = float(row["core_score"]) + float(row["mmlu"]) / float(base["mmlu"])


def get_model(rows: list[dict[str, object]], model: str) -> dict[str, object]:
    for row in rows:
        if row["model"] == model:
            return row
    raise KeyError(f"Model not found in summary CSV: {model}")


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")


def plot_baseline(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["base", "math", "coder"]
    selected = [get_model(rows, label) for label in labels]
    metrics = [("gsm8k", "GSM8K"), ("mmlu", "MMLU"), ("humaneval", "HumanEval")]
    x_positions = list(range(len(labels)))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8, 4.6))
    for index, (metric, label) in enumerate(metrics):
        offsets = [x + (index - 1) * width for x in x_positions]
        values = [float(row[metric]) for row in selected]
        bars = axis.bar(offsets, values, width=width, label=label, color=COLORS[metric])
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=8)
    axis.set_xticks(x_positions, labels)
    axis.set_ylim(0, 0.82)
    axis.set_ylabel("Score")
    axis.set_title("Baseline Models on Three Evaluation Tasks")
    axis.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    save(figure, output)
    plt.close(figure)


def plot_dual_sweep(
    rows: list[dict[str, object]],
    points: list[tuple[str, float, str]],
    output: Path,
    title: str,
    xlabel: str,
) -> None:
    import matplotlib.pyplot as plt

    base = get_model(rows, "base")
    selected = [(label, value, get_model(rows, model)) for label, value, model in points]
    x_values = [value for _, value, _ in selected]
    x_labels = [label for label, _, _ in selected]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    axes[0].plot(x_values, [float(row["gsm8k"]) for _, _, row in selected], marker="o", label="GSM8K", color=COLORS["gsm8k"])
    axes[0].plot(x_values, [float(row["humaneval"]) for _, _, row in selected], marker="o", label="HumanEval", color=COLORS["humaneval"])
    axes[0].axhline(float(base["gsm8k"]), color=COLORS["gsm8k"], linestyle=":", alpha=0.75, label="base GSM8K")
    axes[0].axhline(float(base["humaneval"]), color=COLORS["humaneval"], linestyle=":", alpha=0.75, label="base HumanEval")
    axes[0].set_title("Raw Scores")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(x_values, x_labels)
    axes[0].legend(frameon=False)

    axes[1].plot(x_values, [float(row["core_score"]) for _, _, row in selected], marker="o", color=COLORS["core_score"], label="core_score")
    axes[1].axhline(2.0, color="#999999", linestyle=":", label="base core_score")
    axes[1].set_title("Normalized Core Score")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("GSM8K/base + HumanEval/base")
    axes[1].set_xticks(x_values, x_labels)
    axes[1].legend(frameon=False)

    figure.suptitle(title, y=1.04, fontsize=13)
    save(figure, output)
    plt.close(figure)


def plot_task_arithmetic_lambda(rows: list[dict[str, object]], output: Path) -> None:
    points = [
        ("0.01", 0.01, "ta_math_plus_coder_mlp_lam_0p01"),
        ("0.02", 0.02, "ta_math_plus_coder_mlp_lam_0p02"),
        ("0.05", 0.05, "ta_math_plus_coder_mlp_lam_0p05"),
        ("0.075", 0.075, "ta_math_plus_coder_mlp_lam_0p075"),
        ("0.10", 0.10, "ta_math_plus_coder_mlp_lam_0p1"),
    ]
    plot_dual_sweep(rows, points, output, "Task Arithmetic: lambda Sweep", "lambda")


def plot_layer_ablation(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    base = get_model(rows, "base")
    points = [
        ("full MLP", "ta_math_plus_coder_mlp_lam_0p05"),
        ("exclude 0-8", "ta_math_plus_coder_mlp_exclude_layers_0_8_lam_0p05"),
        ("exclude 9-18", "ta_math_plus_coder_mlp_exclude_layers_9_18_lam_0p05"),
        ("exclude 19-27", "ta_math_plus_coder_mlp_exclude_layers_19_27_lam_0p05"),
    ]
    labels = [label for label, _ in points]
    selected = [get_model(rows, model) for _, model in points]
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))

    humaneval_values = [float(row["humaneval"]) for row in selected]
    bars = axes[0].bar(labels, humaneval_values, color=COLORS["humaneval"])
    axes[0].axhline(float(base["humaneval"]), color="#999999", linestyle=":", label="base HumanEval")
    axes[0].bar_label(bars, labels=[f"{value:.3f}" for value in humaneval_values], padding=2, fontsize=8)
    axes[0].set_title("HumanEval")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 0.405)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(frameon=False, loc="lower left")

    core_values = [float(row["core_score"]) for row in selected]
    bars = axes[1].bar(labels, core_values, color=COLORS["core_score"])
    axes[1].axhline(2.0, color="#999999", linestyle=":", label="base core_score")
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in core_values], padding=2, fontsize=8)
    axes[1].set_title("core_score")
    axes[1].set_ylabel("Normalized score")
    axes[1].set_ylim(0, 2.28)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False, loc="lower left")

    figure.suptitle("Layer Ablation at lambda=0.05", y=1.04, fontsize=13)
    save(figure, output)
    plt.close(figure)


def plot_ties_density(rows: list[dict[str, object]], output: Path) -> None:
    points = [
        ("0.2", 0.2, "ties_math_plus_coder_mlp_exclude_layers_0_8_lam_0p05_dens_0p2"),
        ("0.4", 0.4, "ties_math_plus_coder_mlp_exclude_layers_0_8_lam_0p05_dens_0p4"),
        ("0.5", 0.5, "ties_math_plus_coder_mlp_exclude_layers_0_8_lam_0p05_dens_0p5"),
        ("0.6", 0.6, "ties_math_plus_coder_mlp_exclude_layers_0_8_lam_0p05_dens_0p6"),
        ("0.7", 0.7, "ties_math_plus_coder_mlp_exclude_layers_0_8_lam_0p05_dens_0p7"),
        ("0.8", 0.8, "ties_math_plus_coder_mlp_exclude_layers_0_8_lam_0p05_dens_0p8"),
        ("0.9", 0.9, "ties_math_plus_coder_mlp_exclude_layers_0_8_lam_0p05_dens_0p9"),
    ]
    plot_dual_sweep(rows, points, output, "TIES: density Sweep (exclude layers 0-8)", "density")


def plot_slerp_t(rows: list[dict[str, object]], output: Path) -> None:
    points = [
        ("0.025", 0.025, "slerp_math_to_coder_mlp_layers_9_18_t_0p025"),
        ("0.05", 0.05, "slerp_math_to_coder_mlp_layers_9_18_t_0p05"),
        ("0.075", 0.075, "slerp_math_to_coder_mlp_layers_9_18_t_0p075"),
        ("0.10", 0.10, "slerp_math_to_coder_mlp_layers_9_18_t_0p1"),
        ("0.125", 0.125, "slerp_math_to_coder_mlp_layers_9_18_t_0p125"),
        ("0.15", 0.15, "slerp_math_to_coder_mlp_layers_9_18_t_0p15"),
    ]
    plot_dual_sweep(rows, points, output, "SLERP: t Sweep (MLP layers 9-18)", "SLERP t")


def plot_tradeoff_scatter(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    base = get_model(rows, "base")
    figure, axis = plt.subplots(figsize=(7.9, 5.8))
    methods = ["Baseline", "Task Arithmetic", "TIES", "SLERP"]
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        axis.scatter(
            [float(row["gsm8k"]) for row in selected],
            [float(row["humaneval"]) for row in selected],
            s=58 if method == "Baseline" else 42,
            label=method,
            alpha=0.82,
            color=COLORS.get(method, "#999999"),
            edgecolor="white",
            linewidth=0.6,
        )

    labels = {
        "base": ("base", -0.022, -0.014),
        "math": ("math", 0.004, -0.013),
        "coder": ("coder", -0.026, 0.004),
        "ta_math_plus_coder_mlp_exclude_layers_0_8_lam_0p05": ("TA best", 0.005, 0.004),
        "ties_math_plus_coder_mlp_layers_9_18_lam_0p05_dens_0p85": ("TIES best", 0.006, 0.006),
        "slerp_math_to_coder_mlp_layers_9_18_t_0p125": ("SLERP best", -0.043, 0.005),
    }
    for model, (label, dx, dy) in labels.items():
        row = get_model(rows, model)
        x_value = float(row["gsm8k"])
        y_value = float(row["humaneval"])
        axis.scatter([x_value], [y_value], s=120, facecolors="none", edgecolors="black", linewidth=1.3, zorder=5)
        axis.annotate(label, (x_value, y_value), xytext=(x_value + dx, y_value + dy), fontsize=9)

    axis.axvline(float(base["gsm8k"]), color=COLORS["gsm8k"], linestyle=":", alpha=0.75)
    axis.axhline(float(base["humaneval"]), color=COLORS["humaneval"], linestyle=":", alpha=0.75)
    axis.set_xlim(0.447, 0.755)
    axis.set_ylim(0.273, 0.428)
    axis.set_xlabel("GSM8K score")
    axis.set_ylabel("HumanEval score")
    axis.set_title("Trade-off between Math Reasoning and Code Generation")
    axis.legend(frameon=False, loc="lower left")
    save(figure, output)
    plt.close(figure)


def main() -> int:
    """Regenerate the committed experiment figures from the summary CSV."""
    parser = argparse.ArgumentParser(description="Generate report figures from the committed result summary.")
    parser.add_argument("--input", type=Path, default=Path("outputs/merge_results_summary.csv"), help="Summary CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"), help="Figure output directory")
    args = parser.parse_args()

    rows = read_rows(args.input)
    required = {"method", "model", *TASK_COLUMNS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Summary CSV must contain columns: {sorted(required)}")
    enrich_scores(rows)
    configure_matplotlib()

    plot_baseline(rows, args.output_dir / "baseline_performance.png")
    plot_task_arithmetic_lambda(rows, args.output_dir / "task_arithmetic_lambda_sweep.png")
    plot_layer_ablation(rows, args.output_dir / "layer_ablation.png")
    plot_ties_density(rows, args.output_dir / "ties_density_sweep.png")
    plot_slerp_t(rows, args.output_dir / "slerp_t_sweep.png")
    plot_tradeoff_scatter(rows, args.output_dir / "method_tradeoff_scatter.png")
    print(f"Generated report figures in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
