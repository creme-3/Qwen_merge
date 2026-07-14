from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

TASKS = ["gsm8k", "mmlu", "humaneval"]

BASE_GEN_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
}

# 全局统一的最大生成 token（覆盖模型的默认 generation_config）
MAX_GEN_TOKS = 256

TASK_EVAL_CONFIG = {
    "gsm8k": {
        "num_fewshot": 5,
        "gen_kwargs": {"max_new_tokens": 256},
    },
    "mmlu": {
        "num_fewshot": 5,
        "gen_kwargs": {"max_new_tokens": 32},
    },
    "humaneval": {
        "num_fewshot": 0,
        "gen_kwargs": {"max_new_tokens": 256},
    },
}

CSV_BASE_COLUMNS = [
    "model",
    "task",
    "model_path",
    "num_fewshot",
    "max_new_toks",
    "do_sample",
    "temperature",
    "top_p",
]

CSV_TASK_METRIC_PRIORITY = {
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract"],
    "mmlu": ["acc,none", "acc_norm,none"],
    "humaneval": ["pass@1,none", "pass@1"],
}


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("qwen_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def default_models(root: Path) -> dict[str, Path]:
    return {
        "base": root / "models" / "Qwen2.5-1.5B-base",
        "coder": root / "models" / "Qwen2.5-1.5B-coder",
        "math": root / "models" / "Qwen2.5-1.5B-math",
    }


def build_model_args(model_path: Path, use_cuda: bool) -> str:
    dtype = "float16" if use_cuda else "float32"
    return f"pretrained={model_path.as_posix()},dtype={dtype},parallelize=False"


def select_device(torch_module: Any) -> tuple[str, bool]:
    if torch_module.cuda.is_available():
        return "cuda:0", True
    return "cpu", False


def enable_hf_offline_mode() -> None:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def normalize_results(raw_result: dict[str, Any]) -> dict[str, Any]:
    task_results = raw_result.get("results", raw_result)
    normalized: dict[str, Any] = {}
    for task_name, metrics in task_results.items():
        if isinstance(metrics, dict):
            normalized[task_name] = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        else:
            normalized[task_name] = metrics
    return normalized


def build_task_run(task_name: str) -> dict[str, Any]:
    task_config = TASK_EVAL_CONFIG[task_name]
    gen_kwargs = dict(BASE_GEN_KWARGS)
    gen_kwargs.update(task_config["gen_kwargs"])
    gen_kwargs["max_gen_toks"] = MAX_GEN_TOKS
    return {
        "num_fewshot": task_config["num_fewshot"],
        "gen_kwargs": gen_kwargs,
    }


def flatten_metrics(model_name: str, task_name: str, model_path: str, task_run: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "task": task_name,
        "model_path": model_path,
        "num_fewshot": task_run["num_fewshot"],
        # Report the effective generation limit used. We enforce `max_gen_toks`
        # globally; expose it in CSV under the `max_new_toks` column.
        "max_new_toks": task_run["gen_kwargs"].get("max_gen_toks", task_run["gen_kwargs"].get("max_new_tokens")),
        "do_sample": task_run["gen_kwargs"]["do_sample"],
        "temperature": task_run["gen_kwargs"]["temperature"],
        "top_p": task_run["gen_kwargs"]["top_p"],
    }
    row.update(metrics)
    return row


def main() -> int:
    # 统一的评测协议：同一任务下所有模型必须使用同样的 few-shot、解码参数和随机种子。
    parser = argparse.ArgumentParser(description="Evaluate the three local Qwen2.5-1.5B checkpoints on GSM8K, MMLU, and HumanEval.")
    parser.add_argument("--base", type=Path, default=None, help="Base model directory")
    parser.add_argument("--coder", type=Path, default=None, help="Coder model directory")
    parser.add_argument("--math", dest="math_model", type=Path, default=None, help="Math model directory")
    parser.add_argument("--models-dir", type=Path, default=None, help="Directory to scan for arbitrarily named models (default: repo/models)")
    parser.add_argument("--scan", action="store_true", help="Scan --models-dir for subdirectories that look like HF model snapshots and include them as candidates")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Which model names (directory basenames) to evaluate. If omitted, evaluates all discovered models.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=TASKS,
        help="Which benchmark tasks to run.",
    )
    parser.add_argument("--output", type=Path, default=Path("eval_results") / "qwen25_1p5b_local_eval.json", help="Output JSON file")
    parser.add_argument("--csv-output", type=Path, default=None, help="CSV output file. Default: same stem as JSON output.")
    parser.add_argument("--log-file", type=Path, default=None, help="Log file. Default: same stem as JSON output.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit per task for smoke tests. Omit for a full run.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation and scoring.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by lm_eval.")
    parser.add_argument("--offline", action="store_true", help="Use only local Hugging Face caches and avoid Hub network checks.")
    args = parser.parse_args()
    if args.offline:
        enable_hf_offline_mode()

    import torch
    from lm_eval import evaluator

    csv_output = args.csv_output or args.output.with_suffix(".csv")
    log_file = args.log_file or args.output.with_suffix(".log")
    logger = setup_logger(log_file)

    repo_root = Path(__file__).resolve().parents[1]
    defaults = default_models(repo_root)
    model_paths: dict[str, Path] = {
        "base": args.base or defaults["base"],
        "coder": args.coder or defaults["coder"],
        "math": args.math_model or defaults["math"],
    }

    # 添加任意命名的子目录为可评估模型（以子目录名作为模型标识）。
    models_dir = args.models_dir or (repo_root / "models")
    if args.scan:
        if not models_dir.exists():
            raise FileNotFoundError(f"Models dir to scan not found: {models_dir}")
        for child in sorted(models_dir.iterdir()):
            if not child.is_dir():
                continue
            # 简单的可识别标志：存在常见的权重文件
            if any(child.joinpath(fn).exists() for fn in ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json")):
                # 如果名字与已有的 base/coder/math 重名，会被显式的 args.base/args.coder/args.math 覆盖
                model_paths[child.name] = child

    for path in model_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing model directory: {path}")

    # 选择要评测的模型集合：如果用户通过 --models 指定名称，则只用这些名称；否则评测所有在 model_paths 中发现的模型。
    if args.models is None:
        selected_model_names = list(model_paths.keys())
    else:
        selected_model_names = args.models
        missing = [n for n in selected_model_names if n not in model_paths]
        if missing:
            raise ValueError(f"Requested model names not found: {missing}. Available: {list(model_paths.keys())}")
    selected_model_paths = {name: model_paths[name] for name in selected_model_names}
    selected_tasks = list(args.tasks)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Allow executing model-generated code for HumanEval (requires user consent).
    # This enables the underlying `code_eval` metric used by lm_eval/humaneval.
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    device, use_cuda = select_device(torch)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    logger.info("开始评测：device=%s, use_cuda=%s, batch_size=%s, limit=%s, seed=%s", device, use_cuda, args.batch_size, args.limit, args.seed)

    output: dict[str, Any] = {
        "device": device,
        "use_cuda": use_cuda,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "offline": args.offline,
        "tasks": selected_tasks,
        "models_requested": selected_model_names,
        "models": {},
    }

    csv_rows: list[dict[str, Any]] = []

    for model_name, model_path in selected_model_paths.items():
        logger.info("开始评测模型：%s | %s", model_name, model_path)
        model_args = build_model_args(model_path, use_cuda)
        task_runs = {task_name: build_task_run(task_name) for task_name in selected_tasks}
        normalized: dict[str, Any] = {}
        for task_name in selected_tasks:
            run_config = task_runs[task_name]
            logger.info("任务配置：model=%s | task=%s | fewshot=%s | gen_kwargs=%s", model_name, task_name, run_config["num_fewshot"], run_config["gen_kwargs"])
            try:
                result = evaluator.simple_evaluate(
                    model="hf",
                    model_args=model_args,
                    tasks=[task_name],
                    num_fewshot=run_config["num_fewshot"],
                    batch_size=args.batch_size,
                    device=device,
                    limit=args.limit,
                    bootstrap_iters=1000,
                    log_samples=False,
                    apply_chat_template=False,
                    fewshot_as_multiturn=False,
                    confirm_run_unsafe_code=True,
                    gen_kwargs=run_config["gen_kwargs"],
                    random_seed=args.seed,
                    numpy_random_seed=args.seed,
                    torch_random_seed=args.seed,
                    fewshot_random_seed=args.seed,
                )
            except NotImplementedError as e:
                # Some metrics (e.g. the HuggingFace `code_eval` used by HumanEval)
                # are not supported on Windows. Catch the error, log a warning,
                # and record a placeholder result so the overall run can continue.
                logger.warning("任务 %s 无法在当前平台计算：%s", task_name, str(e))
                result = {"results": {task_name: {"error": "NotImplementedError", "message": str(e)}}}
            except FileNotFoundError as e:
                # HumanEval needs the `code_eval` metric module. If it has not been
                # downloaded yet, or the machine cannot reach Hugging Face Hub,
                # lm_eval fails during task loading. Keep the run alive and surface
                # a clear message in the output files.
                if task_name == "humaneval":
                    logger.warning("任务 %s 无法加载 code_eval 指标：%s", task_name, str(e))
                    result = {
                        "results": {
                            task_name: {
                                "error": "FileNotFoundError",
                                "message": str(e),
                            }
                        }
                    }
                else:
                    raise
            task_metrics = normalize_results(result)
            normalized.update(task_metrics)
            csv_rows.append(flatten_metrics(model_name, task_name, str(model_path), run_config, task_metrics.get(task_name, task_metrics)))
        output["models"][model_name] = {
            "path": str(model_path),
            "model_args": model_args,
            "task_runs": task_runs,
            "results": normalized,
        }
        logger.info("模型结果：%s", json.dumps(normalized, ensure_ascii=False, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_keys: list[str] = []
    seen_metrics: set[str] = set()
    for task_name in selected_tasks:
        for metric_name in CSV_TASK_METRIC_PRIORITY.get(task_name, []):
            if any(metric_name in row for row in csv_rows) and metric_name not in seen_metrics:
                metric_keys.append(metric_name)
                seen_metrics.add(metric_name)
    remaining_metrics = sorted({key for row in csv_rows for key in row.keys()} - set(CSV_BASE_COLUMNS) - seen_metrics)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_BASE_COLUMNS + metric_keys + remaining_metrics)
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info("已保存 JSON：%s", args.output)
    logger.info("已保存 CSV：%s", csv_output)
    logger.info("已保存日志：%s", log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
