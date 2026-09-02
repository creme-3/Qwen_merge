from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from common import build_filter_weight_lines, parse_layer_ranges as parse_common_layer_ranges


DEFAULT_LAMBDAS = ["0.005", "0.01", "0.02", "0.03", "0.05", "0.075"]
DEFAULT_DENSITIES = ["0.2", "0.4", "0.6", "0.8"]
MODEL_DIRS = {
    "base": "Qwen2.5-1.5B-base",
    "math": "Qwen2.5-1.5B-math",
    "coder": "Qwen2.5-1.5B-coder",
}
FILTER_PRESETS = {
    "mlp": ["mlp."],
    "mlp_vo": ["mlp.", "self_attn.v_proj", "self_attn.o_proj"],
    "mlp_vo_norm": ["mlp.", "self_attn.v_proj", "self_attn.o_proj", "input_layernorm", "post_attention_layernorm"],
    "all_attn_mlp": ["mlp.", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
}


def select_mergekit_command() -> str:
    command = shutil.which("mergekit-yaml")
    if command:
        return command
    if os.name == "nt":
        command = shutil.which("mergekit-yaml.exe")
        if command:
            return command
    raise FileNotFoundError("mergekit-yaml not found in PATH")


def detect_cpu_threads() -> int:
    cpu_count = os.cpu_count() or 8
    return max(4, min(8, cpu_count))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def posix_repo_path(path: Path) -> str:
    try:
        return path.resolve().as_posix()
    except OSError:
        return path.as_posix()


def format_lambda(lambda_value: str) -> str:
    return lambda_value.replace(".", "p").replace("-", "m")


def build_recipe(
    method: str,
    base_model: Path,
    expert_model: Path,
    lambda_value: str,
    density_value: str | None,
    int8_mask: bool,
    weighted_filters: list[tuple[str, float]],
) -> str:
    weight_lines = build_filter_weight_lines(weighted_filters, indent=8)
    density_line = f"      density: {density_value}\n" if method == "ties" else ""
    int8_mask_line = f"  int8_mask: {str(int8_mask).lower()}\n" if method == "ties" else ""
    return f"""merge_method: {method}

base_model: {posix_repo_path(base_model)}

dtype: float32
out_dtype: float16
tokenizer_source: base

parameters:
  normalize: true
{int8_mask_line}  lambda: {lambda_value}

models:
  - model: {posix_repo_path(expert_model)}
    parameters:
{density_line}      weight:
{weight_lines}
        - value: 0.0
"""


def parse_layer_ranges(layer_ranges: list[str] | None) -> list[int] | None:
    return parse_common_layer_ranges(layer_ranges)


def resolve_layers(include_ranges: list[str] | None, exclude_ranges: list[str] | None) -> list[int] | None:
    include_layers = parse_layer_ranges(include_ranges)
    exclude_layers = parse_layer_ranges(exclude_ranges)

    if include_layers is not None and exclude_layers is not None:
        raise ValueError("--layers and --exclude-layers cannot be used together")
    if exclude_layers is None:
        return include_layers

    return [layer for layer in range(28) if layer not in set(exclude_layers)]


def layer_scoped_filters(filter_patterns: list[str], layers: list[int] | None, weight: float) -> list[tuple[str, float]]:
    if layers is None:
        return [(filter_pattern, weight) for filter_pattern in filter_patterns]

    scoped: list[tuple[str, float]] = []
    for layer in layers:
        for filter_pattern in filter_patterns:
            scoped.append((f"model.layers.{layer}.{filter_pattern}", weight))
    return scoped


def parse_layer_weight_specs(layer_weight_specs: list[str] | None) -> list[tuple[str, list[int], float]]:
    if not layer_weight_specs:
        return []

    parsed: list[tuple[str, list[int], float]] = []
    for spec in layer_weight_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --layer-weight spec: {spec}. Expected RANGE:WEIGHT")
        range_text, weight_text = spec.split(":", maxsplit=1)
        layers = parse_layer_ranges([range_text])
        if layers is None:
            raise ValueError(f"Invalid empty layer range in --layer-weight spec: {spec}")
        parsed.append((range_text, layers, float(weight_text)))
    return parsed


def layer_weighted_filters(filter_patterns: list[str], layer_weight_specs: list[tuple[str, list[int], float]]) -> list[tuple[str, float]]:
    weighted_filters: list[tuple[str, float]] = []
    seen_layers: set[int] = set()
    for _, layers, weight in layer_weight_specs:
        overlap = seen_layers.intersection(layers)
        if overlap:
            raise ValueError(f"Overlapping --layer-weight layer ids: {sorted(overlap)}")
        seen_layers.update(layers)
        for layer in layers:
            for filter_pattern in filter_patterns:
                weighted_filters.append((f"model.layers.{layer}.{filter_pattern}", weight))
    return weighted_filters


def parse_weight_anchor_specs(weight_anchor_specs: list[str] | None) -> list[tuple[int, float]]:
    if not weight_anchor_specs:
        return []

    anchors: list[tuple[int, float]] = []
    seen_layers: set[int] = set()
    for spec in weight_anchor_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --weight-anchor spec: {spec}. Expected LAYER:WEIGHT")
        layer_text, weight_text = spec.split(":", maxsplit=1)
        layer = int(layer_text)
        if layer < 0 or layer > 27:
            raise ValueError(f"Anchor layer must be in [0, 27], got: {layer}")
        if layer in seen_layers:
            raise ValueError(f"Duplicate --weight-anchor layer: {layer}")
        seen_layers.add(layer)
        anchors.append((layer, float(weight_text)))

    if len(anchors) < 2:
        raise ValueError("--weight-anchor requires at least two anchors")
    return sorted(anchors)


def interpolated_weight_for_layer(layer: int, anchors: list[tuple[int, float]]) -> float:
    if layer <= anchors[0][0]:
        return anchors[0][1]
    if layer >= anchors[-1][0]:
        return anchors[-1][1]

    for (left_layer, left_weight), (right_layer, right_weight) in zip(anchors, anchors[1:]):
        if left_layer <= layer <= right_layer:
            if right_layer == left_layer:
                return right_weight
            ratio = (layer - left_layer) / (right_layer - left_layer)
            return left_weight + ratio * (right_weight - left_weight)

    raise ValueError(f"Could not interpolate layer weight for layer {layer}")


def anchor_weighted_filters(filter_patterns: list[str], anchors: list[tuple[int, float]]) -> list[tuple[str, float]]:
    weighted_filters: list[tuple[str, float]] = []
    for layer in range(28):
        weight = round(interpolated_weight_for_layer(layer, anchors), 6)
        for filter_pattern in filter_patterns:
            weighted_filters.append((f"model.layers.{layer}.{filter_pattern}", weight))
    return weighted_filters


def write_recipe(
    recipes_dir: Path,
    method: str,
    base_name: str,
    expert: str,
    preset: str,
    layer_suffix: str,
    lambda_value: str,
    density_value: str | None,
    recipe_text: str,
) -> Path:
    recipes_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = recipes_dir / f"{build_experiment_name(method, base_name, expert, preset, layer_suffix, lambda_value, density_value)}.yml"
    recipe_path.write_text(recipe_text, encoding="utf-8")
    return recipe_path


def build_experiment_name(
    method: str,
    base_name: str,
    expert: str,
    preset: str,
    layer_suffix: str,
    lambda_value: str,
    density_value: str | None,
) -> str:
    prefix = "ta" if method == "task_arithmetic" else method
    density_suffix = f"_dens_{format_lambda(density_value)}" if density_value is not None else ""
    return f"{prefix}_{base_name}_plus_{expert}_{preset}{layer_suffix}_lam_{format_lambda(lambda_value)}{density_suffix}"


def build_merge_command(
    recipe_path: Path,
    output_dir: Path,
    cpu_threads: int | None,
    use_cuda: bool,
    device: str,
    read_to_gpu: bool,
) -> list[str]:
    command = [
        select_mergekit_command(),
        str(recipe_path),
        str(output_dir),
        "--low-cpu-memory",
        "--copy-tokenizer",
        "--safe-serialization",
        "--write-model-card",
        "--trust-remote-code",
        "--num-threads",
        str(cpu_threads or detect_cpu_threads()),
        "--quiet",
    ]
    if use_cuda:
        command[3:3] = ["--cuda", "--device", device]
        if read_to_gpu:
            command.insert(6, "--read-to-gpu")
    return command


def iter_lambda_values(args: argparse.Namespace) -> list[str]:
    if args.lambda_values:
        return args.lambda_values
    if args.all_lambdas:
        return DEFAULT_LAMBDAS
    return [args.lambda_value]


def iter_density_values(args: argparse.Namespace) -> list[str | None]:
    if args.method != "ties":
        return [None]
    if args.density_values:
        return args.density_values
    if args.all_densities:
        return DEFAULT_DENSITIES
    return [args.density_value]


def format_layer_suffix(layers: list[int] | None, excluded: bool) -> str:
    if layers is None:
        return ""
    if len(layers) == 1:
        prefix = "exclude_layers" if excluded else "layers"
        return f"_{prefix}_{layers[0]}"
    prefix = "exclude_layers" if excluded else "layers"
    return f"_{prefix}_{layers[0]}_{layers[-1]}"


def format_layer_weight_suffix(layer_weight_specs: list[tuple[str, list[int], float]]) -> str:
    if not layer_weight_specs:
        return ""

    parts = []
    for range_text, _, weight in layer_weight_specs:
        safe_range = range_text.replace(",", "_").replace("-", "_")
        safe_weight = str(weight).replace(".", "p").replace("-", "m")
        parts.append(f"{safe_range}_{safe_weight}")
    return "_weights_" + "_".join(parts)


def format_weight_anchor_suffix(anchors: list[tuple[int, float]]) -> str:
    if not anchors:
        return ""

    parts = []
    for layer, weight in anchors:
        safe_weight = str(weight).replace(".", "p").replace("-", "m")
        parts.append(f"{layer}_{safe_weight}")
    return "_anchors_" + "_".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-expert MLP task_arithmetic experiments.")
    parser.add_argument("--method", choices=["task_arithmetic", "ties"], default="task_arithmetic", help="Merge method to use")
    parser.add_argument("--base-name", choices=sorted(MODEL_DIRS), default="base", help="Named base model directory")
    parser.add_argument("--expert", choices=sorted(MODEL_DIRS), required=True, help="Expert model to add to the selected base model")
    parser.add_argument("--lambda", dest="lambda_value", default="0.02", help="Task arithmetic lambda for a single run")
    parser.add_argument("--lambdas", dest="lambda_values", nargs="+", default=None, help="Custom lambda sweep values, for example: --lambdas 0.05 0.075 0.1")
    parser.add_argument("--all-lambdas", action="store_true", help="Generate or run the recommended lambda sweep")
    parser.add_argument("--density", dest="density_value", default="0.5", help="TIES density for a single run")
    parser.add_argument("--densities", dest="density_values", nargs="+", default=None, help="Custom TIES density sweep values, for example: --densities 0.2 0.4 0.6")
    parser.add_argument("--all-densities", action="store_true", help="Generate or run the recommended TIES density sweep")
    parser.add_argument("--int8-mask", action=argparse.BooleanOptionalAction, default=True, help="Use int8 mask for TIES")
    parser.add_argument("--base", type=Path, default=None, help="Base model directory")
    parser.add_argument("--expert-path", type=Path, default=None, help="Override expert model directory")
    parser.add_argument("--recipes-dir", type=Path, default=None, help="Directory for generated YAML recipes")
    parser.add_argument("--outputs-dir", type=Path, default=None, help="Parent directory for merged model outputs")
    parser.add_argument("--filter", dest="filter_patterns", action="append", default=None, help="Mergekit tensor name filter. May be repeated.")
    parser.add_argument("--preset", choices=sorted(FILTER_PRESETS), default="mlp", help="Layer filter preset used when --filter is omitted")
    parser.add_argument("--layers", action="append", default=None, help="Restrict filters to layer ids/ranges, for example: --layers 0-13 or --layers 0-7,20-27")
    parser.add_argument("--exclude-layers", action="append", default=None, help="Use all layers except these ids/ranges, for example: --exclude-layers 0-8")
    parser.add_argument("--layer-weight", action="append", default=None, help="Assign per-layer-range weights, for example: --layer-weight 0-8:0 --layer-weight 9-18:1 --layer-weight 19-27:0.75")
    parser.add_argument("--weight-anchor", action="append", default=None, help="Interpolate layer weights from anchors, for example: --weight-anchor 0:0 --weight-anchor 12:1 --weight-anchor 27:0.5")
    parser.add_argument("--weight", type=float, default=1.0, help="Expert weight inside the selected tensor filter")
    parser.add_argument("--print-only", action="store_true", help="Print generated YAML and command plan without writing files")
    parser.add_argument("--write-only", action="store_true", help="Only write YAML recipes; do not call mergekit")
    parser.add_argument("--dry-run", action="store_true", help="Write recipes and print commands without running mergekit")
    parser.add_argument("--cpu-threads", type=int, default=None, help="CPU threads for mergekit")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True, help="Run mergekit tensor operations on CUDA")
    parser.add_argument("--device", default="cuda:0", help="CUDA device passed to mergekit when --cuda is enabled")
    parser.add_argument("--read-to-gpu", action=argparse.BooleanOptionalAction, default=True, help="Read tensors directly to GPU when --cuda is enabled")
    args = parser.parse_args()

    root = repo_root()
    if args.base is None and args.base_name == args.expert:
        raise ValueError("--base-name and --expert must be different unless --base or --expert-path overrides one side")

    base_model = args.base or (root / "models" / MODEL_DIRS[args.base_name])
    expert_model = args.expert_path or (root / "models" / MODEL_DIRS[args.expert])
    recipes_dir = args.recipes_dir or (root / "merge" / "experiments")
    outputs_dir = args.outputs_dir or (root / "merge_outputs")
    layer_weight_specs = parse_layer_weight_specs(args.layer_weight)
    weight_anchors = parse_weight_anchor_specs(args.weight_anchor)
    if layer_weight_specs and weight_anchors:
        raise ValueError("--layer-weight and --weight-anchor cannot be used together")
    if (layer_weight_specs or weight_anchors) and (args.layers or args.exclude_layers):
        raise ValueError("--layer-weight/--weight-anchor cannot be combined with --layers or --exclude-layers")

    layers = resolve_layers(args.layers, args.exclude_layers)
    excluded_layers = parse_layer_ranges(args.exclude_layers)
    filter_patterns = args.filter_patterns or FILTER_PRESETS[args.preset]
    layer_suffix = (
        format_weight_anchor_suffix(weight_anchors)
        or format_layer_weight_suffix(layer_weight_specs)
        or format_layer_suffix(excluded_layers if excluded_layers is not None else layers, excluded_layers is not None)
    )
    if weight_anchors:
        weighted_filters = anchor_weighted_filters(filter_patterns, weight_anchors)
    elif layer_weight_specs:
        weighted_filters = layer_weighted_filters(filter_patterns, layer_weight_specs)
    else:
        weighted_filters = layer_scoped_filters(filter_patterns, layers, args.weight)

    if not base_model.exists():
        raise FileNotFoundError(f"Missing base model directory: {base_model}")
    if not expert_model.exists():
        raise FileNotFoundError(f"Missing expert model directory: {expert_model}")

    for lambda_value in iter_lambda_values(args):
        for density_value in iter_density_values(args):
            recipe_text = build_recipe(
                method=args.method,
                base_model=base_model,
                expert_model=expert_model,
                lambda_value=lambda_value,
                density_value=density_value,
                int8_mask=args.int8_mask,
                weighted_filters=weighted_filters,
            )
            experiment_name = build_experiment_name(
                args.method,
                args.base_name,
                args.expert,
                args.preset,
                layer_suffix,
                lambda_value,
                density_value,
            )
            output_dir = outputs_dir / experiment_name

            if args.print_only:
                print("---")
                print(f"# output: {output_dir}")
                print(recipe_text)
                continue

            recipe_path = write_recipe(
                recipes_dir,
                args.method,
                args.base_name,
                args.expert,
                args.preset,
                layer_suffix,
                lambda_value,
                density_value,
                recipe_text,
            )

            print("Recipe:", recipe_path)
            print("Output directory:", output_dir)

            if args.write_only:
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            command = build_merge_command(
                recipe_path,
                output_dir,
                args.cpu_threads,
                args.cuda,
                args.device,
                args.read_to_gpu,
            )
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
