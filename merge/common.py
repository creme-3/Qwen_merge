from __future__ import annotations

from pathlib import Path


MAX_MODEL_LAYERS = 28


def parse_layer_ranges(layer_ranges: list[str] | None) -> list[int] | None:
    """Expand comma-separated layer ids/ranges and validate Qwen2.5 layer bounds."""
    if not layer_ranges:
        return None
    layers: set[int] = set()
    for layer_range in layer_ranges:
        for part in layer_range.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", maxsplit=1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"Invalid layer range: {part}")
                layers.update(range(start, end + 1))
            else:
                layers.add(int(part))
    invalid = [layer for layer in layers if layer < 0 or layer >= MAX_MODEL_LAYERS]
    if invalid:
        raise ValueError(f"Layer ids must be in [0, {MAX_MODEL_LAYERS - 1}], got: {invalid}")
    return sorted(layers)


def model_path(root: Path, model_name: str) -> Path:
    """Return a named model path below the repository's models directory."""
    return root / "models" / model_name


def build_filter_weight_lines(filters: list[tuple[str, str | float]], indent: int = 8) -> str:
    """Format mergekit filter/value pairs for a YAML recipe."""
    prefix = " " * indent
    return "\n".join(
        f'{prefix}- filter: "{filter_pattern}"\n{prefix}  value: {value}'
        for filter_pattern, value in filters
    )
