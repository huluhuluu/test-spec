#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.prompt_templates import build_prompt_messages
from run_eval import ARTIFACTS, MODEL_REGISTRY, SAMPLES_DIR, ModelSpec, render_chat_prompt


DEFAULT_MODELS = [
    "taobao_qwen3_4b_eagle3",
    "qwen3_1p7b_sw256_sglang",
    "qwen3_1p7b_eagle3_sglang_train",
]

torch: Any = None
AutoModelForCausalLM: Any = None
AutoTokenizer: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Math500 target-model entropy grouped by context length."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODEL_REGISTRY))
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument(
        "--max-continuation-tokens",
        type=int,
        default=256,
        help="Maximum gold-solution tokens analyzed per sample.",
    )
    parser.add_argument(
        "--context-bin-size",
        type=int,
        default=16,
        help="Group context lengths into bins of this size; use 1 for exact lengths.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for model execution, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ARTIFACTS / "entropy_math500",
    )
    parser.add_argument(
        "--base-model-overrides",
        nargs="*",
        default=[],
        metavar="MODEL=PATH",
        help="Override target/base model path for specific model keys.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and paths without loading model weights.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Write JSON outputs but do not import matplotlib or write the PNG.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read math500_entropy_summary.json from --out-dir and write the PNG only.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    if torch is None:
        raise RuntimeError("torch is not loaded")
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_model_spec(model_key: str) -> ModelSpec:
    return ModelSpec(key=model_key, **MODEL_REGISTRY[model_key])


def parse_base_model_overrides(values: list[str]) -> dict[str, Path]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected MODEL=PATH override, got {value!r}")
        model_key, path = value.split("=", 1)
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model in --base-model-overrides: {model_key}")
        overrides[model_key] = Path(path)
    return overrides


def base_model_dir(model_spec: ModelSpec, overrides: dict[str, Path]) -> Path:
    return overrides.get(model_spec.key, model_spec.base_local_dir)


def render_math500_prompt(tokenizer: Any, sample: dict[str, Any]) -> str:
    messages = build_prompt_messages("math500", sample)
    return render_chat_prompt(tokenizer, messages)


def continuation_text(sample: dict[str, Any]) -> str:
    solution = str(sample.get("solution") or "").strip()
    if solution:
        return "\n" + solution
    return "\nFinal answer: " + str(sample["gold_answer"])


def entropy_rows_for_sample(
    *,
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    model_key: str,
    device: str,
    max_continuation_tokens: int,
) -> list[dict[str, Any]]:
    prompt = render_math500_prompt(tokenizer, sample)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(continuation_text(sample), add_special_tokens=False)["input_ids"]
    continuation_ids = continuation_ids[:max_continuation_tokens]
    if not prompt_ids or not continuation_ids:
        return []

    input_ids = torch.tensor([prompt_ids + continuation_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]

    rows = []
    prompt_len = len(prompt_ids)
    for offset in range(len(continuation_ids)):
        logit_index = prompt_len + offset - 1
        if logit_index < 0:
            continue
        log_probs = torch.log_softmax(logits[logit_index].float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum().item()
        top2 = logits[logit_index].float().topk(2).values
        rows.append(
            {
                "model": model_key,
                "sample_id": sample["sample_id"],
                "sample_index": sample.get("sample_index"),
                "context_len": prompt_len + offset,
                "continuation_offset": offset,
                "entropy": entropy,
                "effective_vocab_size": float(torch.exp(torch.tensor(entropy)).item()),
                "top1_prob": float(probs.max().item()),
                "top1_top2_logit_margin": float((top2[0] - top2[1]).item()),
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]], context_bin_size: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        context_len = int(row["context_len"])
        bin_start = (context_len // context_bin_size) * context_bin_size
        grouped[bin_start].append(row)

    summary = []
    for bin_start in sorted(grouped):
        bucket = grouped[bin_start]
        summary.append(
            {
                "context_len": bin_start,
                "context_len_end": bin_start + context_bin_size - 1,
                "count": len(bucket),
                "mean_entropy": sum(float(row["entropy"]) for row in bucket) / len(bucket),
                "mean_effective_vocab_size": sum(
                    float(row["effective_vocab_size"]) for row in bucket
                )
                / len(bucket),
                "mean_top1_prob": sum(float(row["top1_prob"]) for row in bucket) / len(bucket),
                "mean_top1_top2_logit_margin": sum(
                    float(row["top1_top2_logit_margin"]) for row in bucket
                )
                / len(bucket),
            }
        )
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_summary(out_path: Path, summaries: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for model_key, rows in summaries.items():
        x = [int(row["context_len"]) for row in rows]
        y = [float(row["mean_entropy"]) for row in rows]
        ax.plot(x, y, linewidth=1.6, label=model_key)
    ax.set_title("Math500 Context Length vs Mean Target Entropy")
    ax.set_xlabel("Context length")
    ax.set_ylabel("Mean entropy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)


def main() -> None:
    global AutoModelForCausalLM, AutoTokenizer, torch

    args = parse_args()
    combined_path = args.out_dir / "math500_entropy_summary.json"
    plot_path = args.out_dir / "math500_entropy_by_context_length.png"
    if args.plot_only:
        if not combined_path.exists():
            raise FileNotFoundError(combined_path)
        summaries = json.loads(combined_path.read_text(encoding="utf-8"))
        plot_summary(plot_path, summaries)
        print(plot_path)
        return

    if args.context_bin_size < 1:
        raise ValueError("--context-bin-size must be >= 1")
    if args.max_continuation_tokens < 1:
        raise ValueError("--max-continuation-tokens must be >= 1")

    sample_path = SAMPLES_DIR / "math500.jsonl"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing {sample_path}. Run `rtk python -m eval.prepare_data` first.")
    samples = read_jsonl(sample_path)[: args.sample_size]
    if not samples:
        raise ValueError(f"No samples found in {sample_path}")

    model_specs = [make_model_spec(model_key) for model_key in args.models]
    base_overrides = parse_base_model_overrides(args.base_model_overrides)
    missing_paths = {
        model_spec.key: str(base_model_dir(model_spec, base_overrides))
        for model_spec in model_specs
        if not base_model_dir(model_spec, base_overrides).exists()
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "models": {
                        model_spec.key: str(base_model_dir(model_spec, base_overrides))
                        for model_spec in model_specs
                    },
                    "missing_model_paths": missing_paths,
                    "sample_path": str(sample_path),
                    "sample_count": len(samples),
                    "out_dir": str(args.out_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if missing_paths:
        raise FileNotFoundError(
            "Missing base model paths: "
            + json.dumps(missing_paths, ensure_ascii=False)
            + ". Use --base-model-overrides MODEL=PATH."
        )

    import torch as torch_module
    from transformers import AutoModelForCausalLM as model_cls
    from transformers import AutoTokenizer as tokenizer_cls

    torch = torch_module
    AutoModelForCausalLM = model_cls
    AutoTokenizer = tokenizer_cls

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[str, list[dict[str, Any]]] = {}
    for model_spec in model_specs:
        model_dir = base_model_dir(model_spec, base_overrides)
        print(f"Loading {model_spec.key} from {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype(args.dtype),
            trust_remote_code=True,
            device_map=None,
        ).to(args.device)
        model.eval()

        rows: list[dict[str, Any]] = []
        for index, sample in enumerate(samples, start=1):
            rows.extend(
                entropy_rows_for_sample(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    model_key=model_spec.key,
                    device=args.device,
                    max_continuation_tokens=args.max_continuation_tokens,
                )
            )
            if index % 10 == 0:
                print(f"  {model_spec.key}: {index}/{len(samples)} samples")

        summary = aggregate_rows(rows, args.context_bin_size)
        write_jsonl(args.out_dir / f"{model_spec.key}_math500_entropy_rows.jsonl", rows)
        json_path = args.out_dir / f"{model_spec.key}_math500_entropy_summary.json"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_summaries[model_spec.key] = summary

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.skip_plot:
        plot_summary(plot_path, all_summaries)
    print(combined_path)
    if not args.skip_plot:
        print(plot_path)


if __name__ == "__main__":
    main()
