from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


MODEL_DIRS = {"math": "Qwen2.5-1.5B-math", "coder": "Qwen2.5-1.5B-coder"}


def select_mergekit_command() -> str:
    command = shutil.which("mergekit-yaml")
    if command:
        return command
    if os.name == "nt":
        command = shutil.which("mergekit-yaml.exe")
        if command:
            return command
    raise FileNotFoundError("mergekit-yaml not found in PATH")


def parse_layers(layer_ranges: list[str]) -> list[int]:
    layers: set[int] = set()
    for layer_range in layer_ranges:
        for part in layer_range.split(","):
            part = part.strip()
            if "-" in part:
                start_text, end_text = part.split("-", maxsplit=1)
                layers.update(range(int(start_text), int(end_text) + 1))
            elif part:
                layers.add(int(part))
    invalid = [layer for layer in layers if layer < 0 or layer > 27]
    if invalid:
        raise ValueError(f"Layer ids must be in [0, 27], got: {invalid}")
    return sorted(layers)


def format_value(value: str) -> str:
    return value.replace(".", "p").replace("-", "m")


def build_recipe(start_model: Path, end_model: Path, layers: list[int], t_value: str) -> str:
    t_lines = "\n".join(
        f'    - filter: "model.layers.{layer}.mlp."\n      value: {t_value}'
        for layer in layers
    )
    return f"""slices:
  - sources:
      - model: {start_model.as_posix()}
        layer_range: [0, 28]
      - model: {end_model.as_posix()}
        layer_range: [0, 28]

merge_method: slerp
base_model: {start_model.as_posix()}

dtype: float32
out_dtype: float16
tokenizer_source: base

parameters:
  t:
{t_lines}
    - value: 0.0
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run layer-selective SLERP experiments.")
    parser.add_argument("--start", choices=sorted(MODEL_DIRS), default="math")
    parser.add_argument("--end", choices=sorted(MODEL_DIRS), default="coder")
    parser.add_argument("--layers", action="append", default=None)
    parser.add_argument("--t", dest="t_value", default="0.05")
    parser.add_argument("--t-values", nargs="+", default=None)
    parser.add_argument("--write-only", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=None)
    args = parser.parse_args()

    if args.start == args.end:
        raise ValueError("--start and --end must be different")

    root = Path(__file__).resolve().parents[1]
    start_model = root / "models" / MODEL_DIRS[args.start]
    end_model = root / "models" / MODEL_DIRS[args.end]
    recipes_dir = root / "merge" / "experiments"
    outputs_dir = root / "merge_outputs"
    layers = parse_layers(args.layers or ["9-18"])
    t_values = args.t_values or [args.t_value]

    for t_value in t_values:
        recipe_text = build_recipe(start_model, end_model, layers, t_value)
        name = f"slerp_{args.start}_to_{args.end}_mlp_layers_{layers[0]}_{layers[-1]}_t_{format_value(t_value)}"
        recipe_path = recipes_dir / f"{name}.yml"
        output_dir = outputs_dir / name

        if args.print_only:
            print(recipe_text)
            continue

        recipes_dir.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(recipe_text, encoding="utf-8")
        print("Recipe:", recipe_path)
        print("Output directory:", output_dir)
        if args.write_only:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            select_mergekit_command(), str(recipe_path), str(output_dir),
            "--cuda", "--device", "cuda:0", "--low-cpu-memory", "--read-to-gpu",
            "--copy-tokenizer", "--safe-serialization", "--write-model-card",
            "--trust-remote-code", "--num-threads", str(args.cpu_threads or 4), "--quiet",
        ]
        print("Command:")
        print(" ".join(command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
