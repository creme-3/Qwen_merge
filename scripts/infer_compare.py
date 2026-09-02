from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PROMPTS = [
    "Solve 17 * 23 and explain the calculation.",
    "Write a Python function that returns the factorial of n.",
]


def generate(model_path: Path, prompts: list[str], max_new_tokens: int, device: str) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device if use_cuda else "cpu")
    model.eval()
    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs.append(tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    del model
    if use_cuda:
        torch.cuda.empty_cache()
    return outputs


def main() -> int:
    """Generate deterministic outputs for a list of local checkpoints."""
    parser = argparse.ArgumentParser(description="Compare deterministic generations from local Qwen checkpoints.")
    parser.add_argument("--model", action="append", required=True, help="Model name=path, may be repeated")
    parser.add_argument("--prompt", action="append", default=None, help="Prompt text, may be repeated")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens")
    parser.add_argument("--device", default="cuda:0", help="Generation device; falls back to CPU when CUDA is unavailable")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output file")
    args = parser.parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS
    models: list[tuple[str, Path]] = []
    for item in args.model:
        if "=" not in item:
            parser.error("Each --model must use NAME=PATH")
        name, path_text = item.split("=", maxsplit=1)
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        models.append((name, path))

    sections = ["# Local inference comparison", ""]
    for name, path in models:
        sections.append(f"## {name}")
        sections.append(f"Model path: `{path}`")
        outputs = generate(path, prompts, args.max_new_tokens, args.device)
        for index, (prompt, output) in enumerate(zip(prompts, outputs), start=1):
            sections.extend([f"### Prompt {index}", f"**Input:** {prompt}", "", "```text", output, "```", ""])
    text = "\n".join(sections)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
