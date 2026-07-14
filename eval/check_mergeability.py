from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open


@dataclass
class ModelSnapshot:
    name: str
    root: Path
    config: dict[str, Any]
    generation_config: dict[str, Any]
    tokenizer_hashes: dict[str, str]
    tensor_keys: list[str]
    tensor_shapes: dict[str, tuple[int, ...]]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_tokenizer_hashes(model_root: Path) -> dict[str, str]:
    file_names = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    hashes: dict[str, str] = {}
    for file_name in file_names:
        file_path = model_root / file_name
        if file_path.exists():
            hashes[file_name] = file_sha256(file_path)
    return hashes


def tokenizer_core_match(reference_hashes: dict[str, str], candidate_hashes: dict[str, str]) -> bool:
    core_files = ["tokenizer.json", "vocab.json", "merges.txt"]
    return all(reference_hashes.get(file_name) == candidate_hashes.get(file_name) for file_name in core_files)


def collect_tensor_signature(model_root: Path) -> tuple[list[str], dict[str, tuple[int, ...]]]:
    tensor_path = model_root / "model.safetensors"
    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing weight file: {tensor_path}")

    tensor_shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        tensor_keys = list(handle.keys())
        for key in tensor_keys:
            tensor_shapes[key] = tuple(handle.get_slice(key).get_shape())
    return tensor_keys, tensor_shapes


def load_snapshot(name: str, model_root: Path) -> ModelSnapshot:
    return ModelSnapshot(
        name=name,
        root=model_root,
        config=read_json(model_root / "config.json"),
        generation_config=read_json(model_root / "generation_config.json"),
        tokenizer_hashes=collect_tokenizer_hashes(model_root),
        tensor_keys=[],
        tensor_shapes={},
    )


def enrich_snapshot(snapshot: ModelSnapshot) -> ModelSnapshot:
    tensor_keys, tensor_shapes = collect_tensor_signature(snapshot.root)
    snapshot.tensor_keys = tensor_keys
    snapshot.tensor_shapes = tensor_shapes
    return snapshot


def compare_dict_values(reference: dict[str, Any], candidate: dict[str, Any], keys: list[str]) -> dict[str, tuple[Any, Any]]:
    diff: dict[str, tuple[Any, Any]] = {}
    for key in keys:
        if reference.get(key) != candidate.get(key):
            diff[key] = (reference.get(key), candidate.get(key))
    return diff


def build_report(reference: ModelSnapshot, candidate: ModelSnapshot) -> dict[str, Any]:
    config_keys = [
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "tie_word_embeddings",
        "rope_theta",
        "max_position_embeddings",
        "sliding_window",
        "max_window_layers",
        "torch_dtype",
    ]
    generation_keys = ["bos_token_id", "eos_token_id", "do_sample", "max_new_tokens"]

    config_diff = compare_dict_values(reference.config, candidate.config, config_keys)
    generation_diff = compare_dict_values(reference.generation_config, candidate.generation_config, generation_keys)

    tokenizers_match = tokenizer_core_match(reference.tokenizer_hashes, candidate.tokenizer_hashes)
    same_keys = reference.tensor_keys == candidate.tensor_keys
    same_shapes = same_keys and all(
        reference.tensor_shapes[key] == candidate.tensor_shapes[key] for key in reference.tensor_keys
    )

    hard_blockers = []
    if reference.config.get("architectures") != candidate.config.get("architectures"):
        hard_blockers.append("architectures differ")
    if reference.config.get("model_type") != candidate.config.get("model_type"):
        hard_blockers.append("model_type differs")
    if not same_keys:
        hard_blockers.append("state_dict keys differ")
    if not same_shapes:
        hard_blockers.append("state_dict shapes differ")
    if not tokenizers_match:
        hard_blockers.append("core tokenizer assets differ")

    soft_warnings = []
    if config_diff:
        soft_warnings.append("config fields differ: " + ", ".join(sorted(config_diff.keys())))
    if generation_diff:
        soft_warnings.append("generation config differs: " + ", ".join(sorted(generation_diff.keys())))
    if reference.tokenizer_hashes.get("tokenizer_config.json") != candidate.tokenizer_hashes.get("tokenizer_config.json"):
        soft_warnings.append("tokenizer_config.json differs")

    return {
        "model": candidate.name,
        "path": str(candidate.root),
        "same_architecture": reference.config.get("architectures") == candidate.config.get("architectures"),
        "same_model_type": reference.config.get("model_type") == candidate.config.get("model_type"),
        "same_weight_keys": same_keys,
        "same_weight_shapes": same_shapes,
        "same_tokenizer_assets": tokenizers_match,
        "config_diff": config_diff,
        "generation_diff": generation_diff,
        "hard_blockers": hard_blockers,
        "soft_warnings": soft_warnings,
        "mergekit_methods_supported": len(hard_blockers) == 0,
        "methods": ["task_arithmetic", "ties", "slerp"] if len(hard_blockers) == 0 else [],
    }


def build_pair_report(left: ModelSnapshot, right: ModelSnapshot) -> dict[str, Any]:
    config_keys = [
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "tie_word_embeddings",
        "rope_theta",
        "max_position_embeddings",
        "sliding_window",
        "max_window_layers",
        "torch_dtype",
    ]
    generation_keys = ["bos_token_id", "eos_token_id", "do_sample", "max_new_tokens"]

    config_diff = compare_dict_values(left.config, right.config, config_keys)
    generation_diff = compare_dict_values(left.generation_config, right.generation_config, generation_keys)

    tokenizers_match = tokenizer_core_match(left.tokenizer_hashes, right.tokenizer_hashes)
    same_keys = left.tensor_keys == right.tensor_keys
    same_shapes = same_keys and all(left.tensor_shapes[key] == right.tensor_shapes[key] for key in left.tensor_keys)

    hard_blockers = []
    if left.config.get("architectures") != right.config.get("architectures"):
        hard_blockers.append("architectures differ")
    if left.config.get("model_type") != right.config.get("model_type"):
        hard_blockers.append("model_type differs")
    if not same_keys:
        hard_blockers.append("state_dict keys differ")
    if not same_shapes:
        hard_blockers.append("state_dict shapes differ")
    if not tokenizers_match:
        hard_blockers.append("core tokenizer assets differ")

    soft_warnings = []
    if config_diff:
        soft_warnings.append("config fields differ: " + ", ".join(sorted(config_diff.keys())))
    if generation_diff:
        soft_warnings.append("generation config differs: " + ", ".join(sorted(generation_diff.keys())))
    if left.tokenizer_hashes.get("tokenizer_config.json") != right.tokenizer_hashes.get("tokenizer_config.json"):
        soft_warnings.append("tokenizer_config.json differs")

    return {
        "left": left.name,
        "right": right.name,
        "same_architecture": left.config.get("architectures") == right.config.get("architectures"),
        "same_model_type": left.config.get("model_type") == right.config.get("model_type"),
        "same_weight_keys": same_keys,
        "same_weight_shapes": same_shapes,
        "same_tokenizer_assets": tokenizers_match,
        "config_diff": config_diff,
        "generation_diff": generation_diff,
        "hard_blockers": hard_blockers,
        "soft_warnings": soft_warnings,
        "mergekit_methods_supported": len(hard_blockers) == 0,
        "methods": ["task_arithmetic", "ties", "slerp"] if len(hard_blockers) == 0 else [],
    }


def default_models(root: Path) -> dict[str, Path]:
    return {
        "base": root / "models" / "Qwen2.5-1.5B-base",
        "coder": root / "models" / "Qwen2.5-1.5B-coder",
        "math": root / "models" / "Qwen2.5-1.5B-math",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether the three Qwen2.5-1.5B checkpoints can be merged with mergekit.")
    parser.add_argument("--base", type=Path, default=None, help="Base model directory")
    parser.add_argument("--coder", type=Path, default=None, help="Coder model directory")
    parser.add_argument("--math", dest="math_model", type=Path, default=None, help="Math model directory")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    defaults = default_models(repo_root)
    model_paths = {
        "base": args.base or defaults["base"],
        "coder": args.coder or defaults["coder"],
        "math": args.math_model or defaults["math"],
    }

    snapshots = {}
    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing model directory: {path}")
        snapshots[name] = enrich_snapshot(load_snapshot(name, path))

    base_snapshot = snapshots["base"]
    coder_snapshot = snapshots["coder"]
    math_snapshot = snapshots["math"]

    pair_report = build_pair_report(coder_snapshot, math_snapshot)
    base_reference_reports = {
        name: build_report(base_snapshot, snapshot)
        for name, snapshot in snapshots.items()
        if name != "base"
    }

    summary = {
        "base": str(base_snapshot.root),
        "models": {
            name: {
                "path": str(snapshot.root),
                "config": {
                    key: snapshot.config.get(key)
                    for key in [
                        "architectures",
                        "model_type",
                        "hidden_size",
                        "intermediate_size",
                        "num_hidden_layers",
                        "num_attention_heads",
                        "num_key_value_heads",
                        "vocab_size",
                        "max_position_embeddings",
                        "sliding_window",
                        "rope_theta",
                    ]
                },
                "weight_tensors": len(snapshot.tensor_keys),
            }
            for name, snapshot in snapshots.items()
        },
        "pair": pair_report,
        "base_reference_reports": base_reference_reports,
        "mergeable": pair_report["mergekit_methods_supported"],
        "recommended_methods": ["task_arithmetic", "ties", "slerp"] if pair_report["mergekit_methods_supported"] else [],
        "recommended_notes": [
            "最终合并对象是 math 和 coder，所以核心判断基于这两个模型是否同架构、同权重 key/shape、同 tokenizer 核心文件。",
            "base 仅作为额外参考，帮助你确认两个专家模型是否仍然保持在 Qwen2.5-1.5B 的同一族系里。",
        ],
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("Merge check summary")
        print(f"Base: {summary['base']}")
        for name, info in summary["models"].items():
            cfg = info["config"]
            print(
                f"- {name}: arch={cfg['architectures']}, hidden={cfg['hidden_size']}, layers={cfg['num_hidden_layers']}, "
                f"heads={cfg['num_attention_heads']}/{cfg['num_key_value_heads']}, vocab={cfg['vocab_size']}, "
                f"ctx={cfg['max_position_embeddings']}, rope_theta={cfg['rope_theta']}"
            )
        print("\nmath vs coder")
        print(f"  same_weight_keys: {pair_report['same_weight_keys']}")
        print(f"  same_weight_shapes: {pair_report['same_weight_shapes']}")
        print(f"  same_tokenizer_assets: {pair_report['same_tokenizer_assets']}")
        if pair_report["soft_warnings"]:
            for warning in pair_report["soft_warnings"]:
                print(f"  warning: {warning}")
        if pair_report["hard_blockers"]:
            for blocker in pair_report["hard_blockers"]:
                print(f"  blocker: {blocker}")
        else:
            print("  mergekit methods: task_arithmetic, ties, slerp")

        print("\nbase reference checks")
        for name, report in base_reference_reports.items():
            print(f"  {name} vs base: same_weight_keys={report['same_weight_keys']}, same_weight_shapes={report['same_weight_shapes']}, same_tokenizer_assets={report['same_tokenizer_assets']}")
            if report["soft_warnings"]:
                print(f"    warnings: {', '.join(report['soft_warnings'])}")
        print(f"\nmergeable(math+coder): {summary['mergeable']}")

    return 0 if summary["mergeable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())