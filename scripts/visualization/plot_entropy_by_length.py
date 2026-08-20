#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
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


DEFAULT_DATASETS = ["gsm8k", "math500", "mtbench", "humaneval", "ceval", "cmmlu"]
DEFAULT_MODELS = [
    "taobao_qwen3_4b_eagle3",
    "qwen3_1p7b_sw256_sglang",
    "qwen3_1p7b_eagle3_sglang_train",
    "qwen3_1p7b_eagle3-angelslim",
]

torch: Any = None
AutoModelForCausalLM: Any = None
AutoTokenizer: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot target-model entropy by context length for sampled datasets."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODEL_REGISTRY))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument(
        "--max-continuation-tokens",
        type=int,
        default=256,
        help="Maximum reference/generated continuation tokens analyzed per sample.",
    )
    parser.add_argument(
        "--context-bin-size",
        type=int,
        default=16,
        help="Group context lengths into bins of this size; use 1 for exact lengths.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS / "entropy_all_datasets")
    parser.add_argument(
        "--base-model-overrides",
        nargs="*",
        default=[],
        metavar="MODEL=PATH",
        help="Override target/base model path for specific model keys.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read existing summaries from --out-dir and write one PNG per dataset.",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def render_prompt(tokenizer: Any, dataset_name: str, sample: dict[str, Any]) -> str:
    messages = build_prompt_messages(dataset_name, sample)
    return render_chat_prompt(tokenizer, messages)


def reference_continuation(dataset_name: str, sample: dict[str, Any]) -> str | None:
    if dataset_name == "gsm8k":
        return "\n" + str(sample["gold_answer"])
    if dataset_name == "math500":
        solution = str(sample.get("solution") or "").strip()
        return "\n" + (solution or f"Final answer: {sample['gold_answer']}")
    if dataset_name == "humaneval":
        return str(sample["canonical_solution"])
    if dataset_name in {"ceval", "cmmlu"}:
        return str(sample["gold_answer"])
    if dataset_name == "mtbench":
        return None
    raise ValueError(dataset_name)


def generated_continuation_ids(
    *,
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    device: str,
    max_new_tokens: int,
) -> list[int]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )[0].tolist()
    return output_ids[len(prompt_ids) :]


def continuation_ids_for_sample(
    *,
    model: Any,
    tokenizer: Any,
    dataset_name: str,
    sample: dict[str, Any],
    prompt_ids: list[int],
    device: str,
    max_continuation_tokens: int,
) -> tuple[list[int], str]:
    reference = reference_continuation(dataset_name, sample)
    if reference is None:
        return (
            generated_continuation_ids(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                device=device,
                max_new_tokens=max_continuation_tokens,
            ),
            "greedy_generated",
        )
    return (
        tokenizer(reference, add_special_tokens=False)["input_ids"][:max_continuation_tokens],
        "reference",
    )


def entropy_rows_for_sample(
    *,
    model: Any,
    tokenizer: Any,
    dataset_name: str,
    sample: dict[str, Any],
    model_key: str,
    device: str,
    max_continuation_tokens: int,
) -> list[dict[str, Any]]:
    prompt = render_prompt(tokenizer, dataset_name, sample)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        return []
    continuation_ids, continuation_source = continuation_ids_for_sample(
        model=model,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        sample=sample,
        prompt_ids=prompt_ids,
        device=device,
        max_continuation_tokens=max_continuation_tokens,
    )
    if not continuation_ids:
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
                "dataset": dataset_name,
                "sample_id": sample["sample_id"],
                "sample_index": sample.get("sample_index"),
                "context_len": prompt_len + offset,
                "continuation_offset": offset,
                "continuation_source": continuation_source,
                "entropy": entropy,
                "effective_vocab_size": float(torch.exp(torch.tensor(entropy)).item()),
                "top1_prob": float(probs.max().item()),
                "top1_top2_logit_margin": float((top2[0] - top2[1]).item()),
            }
        )
    return rows


def clone_rows_for_model(rows: list[dict[str, Any]], model_key: str) -> list[dict[str, Any]]:
    cloned = []
    for row in rows:
        item = copy.copy(row)
        item["model"] = model_key
        cloned.append(item)
    return cloned


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


def plot_dataset_summary(
    *,
    dataset_name: str,
    out_path: Path,
    summaries: dict[str, list[dict[str, Any]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for model_key, rows in summaries.items():
        x = [int(row["context_len"]) for row in rows]
        y = [float(row["mean_entropy"]) for row in rows]
        ax.plot(x, y, linewidth=1.6, label=model_key)
    ax.set_title(f"{dataset_name} Context Length vs Mean Target Entropy")
    ax.set_xlabel("Context length")
    ax.set_ylabel("Mean entropy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_dataset_outputs(
    *,
    out_dir: Path,
    dataset_name: str,
    model_rows: dict[str, list[dict[str, Any]]],
    context_bin_size: int,
    skip_plot: bool,
) -> dict[str, list[dict[str, Any]]]:
    dataset_dir = out_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for model_key, rows in model_rows.items():
        summary = aggregate_rows(rows, context_bin_size)
        summaries[model_key] = summary
        write_jsonl(dataset_dir / f"{model_key}_entropy_rows.jsonl", rows)
        (dataset_dir / f"{model_key}_entropy_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    (dataset_dir / "entropy_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not skip_plot:
        plot_dataset_summary(
            dataset_name=dataset_name,
            out_path=dataset_dir / "entropy_by_context_length.png",
            summaries=summaries,
        )
    return summaries


def plot_existing_outputs(out_dir: Path, datasets: list[str]) -> None:
    for dataset_name in datasets:
        summary_path = out_dir / dataset_name / "entropy_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        plot_dataset_summary(
            dataset_name=dataset_name,
            out_path=out_dir / dataset_name / "entropy_by_context_length.png",
            summaries=summaries,
        )
        print(out_dir / dataset_name / "entropy_by_context_length.png")


def main() -> None:
    global AutoModelForCausalLM, AutoTokenizer, torch

    args = parse_args()
    if args.plot_only:
        plot_existing_outputs(args.out_dir, args.datasets)
        return
    if args.context_bin_size < 1:
        raise ValueError("--context-bin-size must be >= 1")
    if args.max_continuation_tokens < 1:
        raise ValueError("--max-continuation-tokens must be >= 1")

    dataset_samples = {}
    for dataset_name in args.datasets:
        sample_path = SAMPLES_DIR / f"{dataset_name}.jsonl"
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing {sample_path}. Run `rtk python -m eval.prepare_data` first.")
        samples = read_jsonl(sample_path)[: args.sample_size]
        if not samples:
            raise ValueError(f"No samples found in {sample_path}")
        dataset_samples[dataset_name] = samples

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
                    "datasets": {name: len(samples) for name, samples in dataset_samples.items()},
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

    specs_by_base: dict[str, list[ModelSpec]] = defaultdict(list)
    for model_spec in model_specs:
        specs_by_base[str(base_model_dir(model_spec, base_overrides))].append(model_spec)

    all_dataset_summaries = {}
    for dataset_name, samples in dataset_samples.items():
        print(f"Dataset {dataset_name}: {len(samples)} samples")
        model_rows: dict[str, list[dict[str, Any]]] = {}
        for base_path, specs in specs_by_base.items():
            representative = specs[0]
            print(f"Loading {representative.key} base from {base_path}")
            tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_path,
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
                        dataset_name=dataset_name,
                        sample=sample,
                        model_key=representative.key,
                        device=args.device,
                        max_continuation_tokens=args.max_continuation_tokens,
                    )
                )
                if index % 10 == 0:
                    print(f"  {representative.key}/{dataset_name}: {index}/{len(samples)} samples")

            for spec in specs:
                model_rows[spec.key] = rows if spec.key == representative.key else clone_rows_for_model(rows, spec.key)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_dataset_summaries[dataset_name] = write_dataset_outputs(
            out_dir=args.out_dir,
            dataset_name=dataset_name,
            model_rows=model_rows,
            context_bin_size=args.context_bin_size,
            skip_plot=args.skip_plot,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "entropy_summary.json").write_text(
        json.dumps(all_dataset_summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.out_dir / "entropy_summary.json")


if __name__ == "__main__":
    main()
