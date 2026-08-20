#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.config_loader import HF_ENDPOINT, MODEL_PATH, RUN_EVAL_CONFIG, project_path
from eval.prompt_templates import build_prompt_messages
from eval.prepare_data import DATASET_NAMES, DEFAULT_CMMLU_REPO as PREPARE_DEFAULT_CMMLU_REPO

ROOT = Path(__file__).resolve().parent
RUN_SPEC_CONFIG = RUN_EVAL_CONFIG.get("speculative", {})
RUN_CONDA_ENVS = RUN_EVAL_CONFIG.get("conda_envs", {})
RUN_PYTHON_BINS = RUN_EVAL_CONFIG.get("python_bins", {})
ARTIFACTS = project_path(RUN_EVAL_CONFIG.get("artifacts_dir", "artifacts"))
SAMPLES_DIR = ARTIFACTS / "samples"
LOGS_DIR = ARTIFACTS / "logs"
REPORTS_DIR = ARTIFACTS / "reports"
DEFAULT_HF_HOME = MODEL_PATH.expanduser()
TRACE_CONTEXT_WINDOW = int(RUN_EVAL_CONFIG.get("trace_context_window", 16))

SPEC_NUM_STEPS = int(RUN_SPEC_CONFIG.get("num_steps", 7))
SPEC_EAGLE_TOPK = int(RUN_SPEC_CONFIG.get("eagle_topk", 10))
SPEC_NUM_DRAFT_TOKENS = int(RUN_SPEC_CONFIG.get("num_draft_tokens", 32))
EVAL_CONTEXT_LENGTH = int(RUN_EVAL_CONFIG.get("context_length", 1024))
DEFAULT_MAX_NEW_TOKENS = int(RUN_EVAL_CONFIG.get("default_max_new_tokens", 2048))
LONG_MAX_NEW_TOKENS = int(RUN_EVAL_CONFIG.get("long_max_new_tokens", 2048))
PROMPT_TOKEN_SAFETY_MARGIN = int(RUN_EVAL_CONFIG.get("prompt_token_safety_margin", 8))
SGLANG_TOKEN_SAFETY_MARGIN = int(RUN_EVAL_CONFIG.get("sglang_token_safety_margin", 64))
MTBENCH_TURN1_HISTORY_RESERVE = int(RUN_EVAL_CONFIG.get("mtbench_turn1_history_reserve", 256))
SGLANG_INPUT_TOKEN_FUDGE = int(RUN_EVAL_CONFIG.get("sglang_input_token_fudge", 32))
DEFAULT_SAMPLE_SIZE = int(RUN_EVAL_CONFIG.get("default_sample_size", 80))
DEFAULT_GPUS = [int(gpu) for gpu in RUN_EVAL_CONFIG.get("default_gpus", [0, 1, 2, 3])]
DEFAULT_SEED = int(RUN_EVAL_CONFIG.get("default_seed", 20260429))
DEFAULT_DATASETS = list(RUN_EVAL_CONFIG.get("default_datasets", DATASET_NAMES))
DEFAULT_CMMLU_REPO = project_path(
    RUN_EVAL_CONFIG.get("cmmlu_repo", PREPARE_DEFAULT_CMMLU_REPO)
)
BACKEND_SGLANG = "specforge_sglang"
BACKEND_VLLM = "angelslim_vllm"
BACKEND_ANGELSLIM_EAGLE3 = "angelslim_eagle3"
SGLANG_PYTHON_BIN = os.environ.get("SGLANG_PYTHON_BIN") or RUN_PYTHON_BINS.get("sglang")
SGLANG_SW_PYTHON_BIN = (
    os.environ.get("SGLANG_SW_PYTHON_BIN") or RUN_PYTHON_BINS.get("sglang_sliding_window")
)
VLLM_PYTHON_BIN = os.environ.get("VLLM_PYTHON_BIN") or RUN_PYTHON_BINS.get("vllm")
ANGELSLIM_PYTHON_BIN = (
    os.environ.get("ANGELSLIM_PYTHON_BIN") or RUN_PYTHON_BINS.get("angelslim_eagle3")
)
SGLANG_CONDA_ENV = os.environ.get(
    "SGLANG_CONDA_ENV",
    RUN_CONDA_ENVS.get("sglang", "eagle3-sglang-bench"),
)
SGLANG_SW_CONDA_ENV = os.environ.get(
    "SGLANG_SW_CONDA_ENV",
    RUN_CONDA_ENVS.get("sglang_sliding_window", "eagle3-sglang-sw-bench"),
)
SGLANG_SW_DRAFT_ATTENTION_BACKEND = os.environ.get(
    "SGLANG_SW_DRAFT_ATTENTION_BACKEND",
    "triton",
)
VLLM_CONDA_ENV = os.environ.get(
    "VLLM_CONDA_ENV",
    RUN_CONDA_ENVS.get("vllm", "eagle3-vllm-bench"),
)
ANGELSLIM_CONDA_ENV = os.environ.get(
    "ANGELSLIM_CONDA_ENV",
    RUN_CONDA_ENVS.get("angelslim_eagle3", "eagle3-angelslim-bench"),
)

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
BOXED_RE = re.compile(r"\\boxed\{(.+?)\}")
LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
CODE_LINE_RE = re.compile(
    r"^\s*(def |class |from |import |@|return\b|if\b|elif\b|else:|for\b|while\b|try:|except\b|with\b|assert\b|pass\b|break\b|continue\b|raise\b|[A-Za-z_][A-Za-z0-9_]*\s*=)"
)

MODEL_REGISTRY = {
    "qwen3_1p7b_eagle3-angelslim": {
        "display_name": "AngelSlim/Qwen3-1.7B_eagle3",
        "draft_repo_id": "AngelSlim/Qwen3-1.7B_eagle3",
        "base_repo_id": "Qwen/Qwen3-1.7B",
        "backend": BACKEND_ANGELSLIM_EAGLE3,
    },
    "qwen3_1p7b_eagle3_sglang_train": {
        "display_name": "Qwen3-1.7B_eagle3_sharegpt_sglang_train",
        "draft_repo_id": "local/Qwen3-1.7B_eagle3",
        "base_repo_id": "Qwen/Qwen3-1.7B",
        "backend": BACKEND_SGLANG,
    },
    "qwen3_1p7b_eagle3_vb128k_step23w": {
        "display_name": "Qwen3-1.7B_eagle3_sharegpt_vb128k_step23w",
        "draft_repo_id": "local/Qwen3-1.7B_eagle3",
        "base_repo_id": "Qwen/Qwen3-1.7B",
        "backend": BACKEND_SGLANG,
    },
    "qwen3_4b_eagle3-angelslim": {
        "display_name": "AngelSlim/Qwen3-4B_eagle3",
        "draft_repo_id": "AngelSlim/Qwen3-4B_eagle3",
        "base_repo_id": "Qwen/Qwen3-4B",
        "backend": BACKEND_ANGELSLIM_EAGLE3,
    },
    "qwen3_4b_dflash_b16_sglang": {
        "display_name": "local/Qwen3-4B-DFlash-b16",
        "draft_repo_id": "local/Qwen3-4B-DFlash-b16",
        "base_repo_id": "Qwen/Qwen3-4B",
        "backend": BACKEND_SGLANG,
    },
    "taobao_qwen3_4b_eagle3": {
        "display_name": "taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3",
        "draft_repo_id": "taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3",
        "base_repo_id": "Qwen/Qwen3-4B-Instruct-2507",
        "backend": BACKEND_SGLANG,
    },
    "zjcxy_qwen3_4b_eagle3_zh": {
        "display_name": "Zjcxy-SmartAI/Eagle3-Qwen3-4B-Instruct-2507-zh",
        "draft_repo_id": "Zjcxy-SmartAI/Eagle3-Qwen3-4B-Instruct-2507-zh",
        "base_repo_id": "Qwen/Qwen3-4B-Instruct-2507",
        "backend": BACKEND_SGLANG,
    },
    "hunyuan_1p8b_eagle3": {
        "display_name": "AngelSlim/Hunyuan-1.8B-Instruct_eagle3",
        "draft_repo_id": "AngelSlim/Hunyuan-1.8B-Instruct_eagle3",
        "base_repo_id": "tencent/Hunyuan-1.8B-Instruct",
        "backend": BACKEND_ANGELSLIM_EAGLE3,
    },
    "hunyuan_4b_eagle3": {
        "display_name": "AngelSlim/Hunyuan-4B-Instruct_eagle3",
        "draft_repo_id": "AngelSlim/Hunyuan-4B-Instruct_eagle3",
        "base_repo_id": "tencent/Hunyuan-4B-Instruct",
        "backend": BACKEND_ANGELSLIM_EAGLE3,
    },
    "qwen3_1p7b_sw64_sglang": {
        "display_name": "local/qwen3-1.7b-eagle3-sharegpt-sw64",
        "draft_repo_id": "local/qwen3-1.7b-eagle3-sharegpt-sw64",
        "base_repo_id": "Qwen/Qwen3-1.7B",
        "backend": BACKEND_SGLANG,
        "requires_sliding_window_specforge": True,
    },
    "qwen3_1p7b_sw128_sglang": {
        "display_name": "local/qwen3-1.7b-eagle3-sharegpt-sw128",
        "draft_repo_id": "local/qwen3-1.7b-eagle3-sharegpt-sw128",
        "base_repo_id": "Qwen/Qwen3-1.7B",
        "backend": BACKEND_SGLANG,
        "requires_sliding_window_specforge": True,
    },
    "qwen3_1p7b_k2_sw128_sglang": {
        "display_name": "local/qwen3-1.7b-eagle3-k2-sw128-sharegpt",
        "draft_repo_id": "local/qwen3-1.7b-eagle3-k2-sw128-sharegpt",
        "base_repo_id": "Qwen/Qwen3-1.7B",
        "backend": BACKEND_SGLANG,
        "requires_sliding_window_specforge": True,
    },
    "qwen3_1p7b_k2_sw256_sglang": {
        "display_name": "local/qwen3-1.7b-eagle3-k2-sw256-sharegpt",
        "draft_repo_id": "local/qwen3-1.7b-eagle3-k2-sw256-sharegpt",
        "base_repo_id": "Qwen/Qwen3-1.7B",
        "backend": BACKEND_SGLANG,
        "requires_sliding_window_specforge": True,
    },
    "qwen3_1p7b_k2_sw256_mha_sglang": {
        "display_name": "local/qwen3-1.7b-eagle3-k2-sw256-mha-sharegpt",
        "draft_repo_id": "local/qwen3-1.7b-eagle3-k2-sw256-mha-sharegpt",
        "base_repo_id": "Qwen/Qwen3-1.7B",
        "backend": BACKEND_SGLANG,
        "requires_sliding_window_specforge": True,
    },
    "qwen3_1p7b_k4_sw256_sglang": {
        "display_name": "local/qwen3-1.7b-eagle3-k4-sw256-sharegpt",
        "draft_repo_id": "local/qwen3-1.7b-eagle3-k4-sw256-sharegpt",
        "base_repo_id": "Qwen/Qwen3-1.7B",
        "backend": BACKEND_SGLANG,
        "requires_sliding_window_specforge": True,
    },
    "qwen3_1p7b_sw256_sglang": {
        "display_name": "local/qwen3-1.7b-eagle3-sharegpt-sw256",
        "draft_repo_id": "local/qwen3-1.7b-eagle3-sharegpt-sw256",
        "base_repo_id": "Qwen/Qwen3-1.7B",
        "backend": BACKEND_SGLANG,
        "requires_sliding_window_specforge": True,
    },
}

for model_key, model_config in RUN_EVAL_CONFIG.get("model_overrides", {}).items():
    MODEL_REGISTRY[model_key] = {**MODEL_REGISTRY.get(model_key, {}), **model_config}

DEFAULT_MODEL_KEYS = list(RUN_EVAL_CONFIG.get("default_models", list(MODEL_REGISTRY)))

def conda_python_bin(env_name: str) -> Optional[str]:
    if not shutil.which("conda"):
        return None
    try:
        output = subprocess.check_output(
            ["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None
    path = output.strip().splitlines()[-1] if output.strip() else ""
    return path or None


def backend_python_bin(backend: str) -> Optional[str]:
    if backend == BACKEND_VLLM:
        return VLLM_PYTHON_BIN or conda_python_bin(VLLM_CONDA_ENV)
    if backend == BACKEND_ANGELSLIM_EAGLE3:
        return ANGELSLIM_PYTHON_BIN or conda_python_bin(ANGELSLIM_CONDA_ENV) or sys.executable
    return SGLANG_PYTHON_BIN or conda_python_bin(SGLANG_CONDA_ENV) or sys.executable


def sglang_sliding_window_python_bin() -> Optional[str]:
    return SGLANG_SW_PYTHON_BIN or conda_python_bin(SGLANG_SW_CONDA_ENV)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    draft_repo_id: str
    base_repo_id: str
    backend: str
    draft_model_path: Optional[str] = None
    base_model_path: Optional[str] = None
    requires_sliding_window_specforge: bool = False

    @property
    def draft_local_dir(self) -> Path:
        if self.draft_model_path is not None:
            return Path(self.draft_model_path)
        return DEFAULT_HF_HOME / repo_leaf(self.draft_repo_id)

    @property
    def base_local_dir(self) -> Path:
        if self.base_model_path is not None:
            return Path(self.base_model_path)
        return DEFAULT_HF_HOME / repo_leaf(self.base_repo_id)

    @property
    def python_bin(self) -> str:
        if self.backend == BACKEND_SGLANG and self.requires_sliding_window_specforge:
            python_bin = sglang_sliding_window_python_bin()
        else:
            python_bin = backend_python_bin(self.backend)
        if not python_bin:
            if self.backend == BACKEND_VLLM:
                env_name = VLLM_CONDA_ENV
                env_var = "VLLM_PYTHON_BIN"
            elif self.backend == BACKEND_ANGELSLIM_EAGLE3:
                env_name = ANGELSLIM_CONDA_ENV
                env_var = "ANGELSLIM_PYTHON_BIN"
            elif self.requires_sliding_window_specforge:
                env_name = SGLANG_SW_CONDA_ENV
                env_var = "SGLANG_SW_PYTHON_BIN"
            else:
                env_name = SGLANG_CONDA_ENV
                env_var = "SGLANG_PYTHON_BIN"
            raise FileNotFoundError(
                f"Cannot resolve Python for backend {self.backend}. "
                f"Set {env_var} "
                f"or create conda env {env_name}."
            )
        return python_bin


def repo_leaf(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def model_specs(model_keys: Optional[list[str]] = None) -> list[ModelSpec]:
    keys = model_keys or list(MODEL_REGISTRY)
    return [ModelSpec(key=key, **MODEL_REGISTRY[key]) for key in keys]


def ensure_sliding_window_specforge_available(model_spec: ModelSpec) -> None:
    if not model_spec.requires_sliding_window_specforge:
        return
    try:
        module = importlib.import_module("specforge.modeling.draft.llama3_eagle")
    except ImportError as exc:
        raise ImportError(
            "Sliding-window SGLang models require the modified SpecForge package. "
            "Install it in the selected runtime environment with "
            "`pip install -e third_party/SpecForge-sliding-window` from the `feat/sliding-window` branch, "
            "or set SGLANG_SW_PYTHON_BIN / SGLANG_SW_CONDA_ENV to an environment that has it."
        ) from exc
    if not hasattr(module, "LlamaForCausalLMEagle3"):
        raise ImportError(
            "The installed SpecForge package does not expose LlamaForCausalLMEagle3. "
            "Use the modified sliding-window SpecForge branch."
        )


def ensure_dirs() -> None:
    for path in (ARTIFACTS, SAMPLES_DIR, LOGS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_")


def format_gpu_ids(gpu_ids: list[int]) -> str:
    return ",".join(str(gpu_id) for gpu_id in gpu_ids)


def prepend_pythonpath(path: Path) -> None:
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    root = str(path)
    if root in parts:
        return
    os.environ["PYTHONPATH"] = root if not current else f"{root}{os.pathsep}{current}"


def prepend_import_path(path: Path) -> None:
    root = str(path)
    if root not in sys.path:
        sys.path.insert(0, root)
    prepend_pythonpath(path)


def clear_imported_specforge() -> None:
    for module_name in list(sys.modules):
        if module_name == "specforge" or module_name.startswith("specforge."):
            del sys.modules[module_name]


def find_specforge_source_from_draft_path(draft_model_path: Path) -> Optional[Path]:
    for path in (draft_model_path, *draft_model_path.parents):
        if (path / "specforge" / "modeling" / "draft" / "llama3_eagle.py").exists():
            return path
    return None


def sliding_window_specforge_source_path(draft_model_path: Path) -> Optional[Path]:
    env_path = os.environ.get("SGLANG_SW_SPECFORGE_PATH")
    if env_path:
        return Path(env_path)
    configured = RUN_EVAL_CONFIG.get("specforge_paths", {}).get("sliding_window")
    if configured:
        return project_path(configured)
    inferred = find_specforge_source_from_draft_path(draft_model_path)
    if inferred is not None:
        return inferred
    fallback = ROOT / "third_party" / "SpecForge-sliding-window"
    return fallback if fallback.exists() else None


def draft_model_config(draft_model_path: Path) -> dict[str, Any]:
    config_path = draft_model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing draft model config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def draft_training_args(draft_model_path: Path) -> dict[str, Any]:
    state_path = draft_model_path / "training_state.pt"
    if not state_path.exists():
        return {}
    try:
        import torch

        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(state, dict):
        return {}
    args = state.get("args")
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    return vars(args) if hasattr(args, "__dict__") else {}


def sliding_window_speculative_config(draft_model_path: Path) -> dict[str, int]:
    training_args = draft_training_args(draft_model_path)
    num_steps = SPEC_NUM_STEPS
    if training_args.get("ttt_length") is not None:
        num_steps = int(training_args["ttt_length"])
    return {
        "speculative_num_steps": num_steps,
        "speculative_eagle_topk": SPEC_EAGLE_TOPK,
        "speculative_num_draft_tokens": SPEC_NUM_DRAFT_TOKENS,
    }


def decode_token_pieces(tokenizer: Any, token_ids: list[int]) -> list[str]:
    pieces = []
    for token_id in token_ids:
        try:
            pieces.append(
                tokenizer.decode(
                    [int(token_id)],
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
            )
        except Exception:
            pieces.append(f"<decode_error:{token_id}>")
    return pieces


def decode_token_text(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    try:
        return tokenizer.decode(
            token_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
    except Exception:
        return ""


def mean_accept_length(accept_lengths: list[int]) -> float:
    return (sum(accept_lengths) / len(accept_lengths)) if accept_lengths else 0.0


def accept_length_histogram(accept_lengths: list[int]) -> dict[str, int]:
    histogram = Counter(int(length) for length in accept_lengths)
    return {str(k): histogram[k] for k in sorted(histogram)}


def histogram_from_sglang_meta(meta_info: dict[str, Any]) -> dict[str, int]:
    raw = meta_info.get("spec_accept_histogram")
    if not isinstance(raw, list):
        return {}
    histogram = {str(idx): int(count) for idx, count in enumerate(raw) if int(count) > 0}
    return histogram


def normalize_sglang_meta_info(meta_info: dict[str, Any]) -> dict[str, Any]:
    meta = dict(meta_info)
    histogram = histogram_from_sglang_meta(meta)
    if histogram:
        total_steps = sum(histogram.values())
        accepted_tokens = sum(int(length) * count for length, count in ((int(k), v) for k, v in histogram.items()))
        meta["accept_length_histogram"] = histogram
        meta["spec_verify_ct"] = total_steps
        meta["spec_accept_length"] = (accepted_tokens / total_steps) if total_steps > 0 else 0.0
    return meta


def build_trace_stats(trace_events: list[dict[str, Any]]) -> dict[str, Any]:
    accept_lengths = [int(ev["accept_len"]) for ev in trace_events if ev.get("accept_len") is not None]
    total_draft_tokens = 0
    for ev in trace_events:
        if ev.get("num_draft_tokens") is not None:
            total_draft_tokens += int(ev["num_draft_tokens"])
        elif isinstance(ev.get("tree"), dict) and isinstance(ev["tree"].get("draft_token_ids"), list):
            total_draft_tokens += max(0, len(ev["tree"]["draft_token_ids"]) - 1)
        else:
            total_draft_tokens += len(ev.get("draft_chunk_token_ids") or [])
    return {
        "spec_accept_length": mean_accept_length(accept_lengths),
        "spec_accept_rate": (
            (sum(accept_lengths) / total_draft_tokens) if total_draft_tokens > 0 else None
        ),
        "spec_verify_ct": len(accept_lengths),
        "accept_length_histogram": accept_length_histogram(accept_lengths),
    }


def context_accept_length_stats(trace_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[int]] = {}
    for ev in trace_events:
        if ev.get("context_len") is None or ev.get("accept_len") is None:
            continue
        context_len = int(ev["context_len"])
        grouped.setdefault(context_len, []).append(int(ev["accept_len"]))

    rows = []
    for context_len in sorted(grouped):
        values = grouped[context_len]
        rows.append(
            {
                "context_len": context_len,
                "count": len(values),
                "mean_accept_len": mean_accept_length(values),
            }
        )
    return rows


def merge_context_accept_length(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, float]] = {}
    for summary in summaries:
        for row in summary.get("context_accept_length", []):
            context_len = int(row["context_len"])
            count = int(row["count"])
            grouped.setdefault(context_len, {"count": 0.0, "sum": 0.0})
            grouped[context_len]["count"] += count
            grouped[context_len]["sum"] += float(row["mean_accept_len"]) * count

    rows = []
    for context_len in sorted(grouped):
        count = int(grouped[context_len]["count"])
        rows.append(
            {
                "context_len": context_len,
                "count": count,
                "mean_accept_len": grouped[context_len]["sum"] / count if count else 0.0,
            }
        )
    return rows


def merge_accept_length_histogram(summaries: list[dict[str, Any]]) -> dict[str, int]:
    histogram = Counter()
    for summary in summaries:
        for key, value in summary.get("accept_length_histogram", {}).items():
            histogram[int(key)] += int(value)
    return {str(k): histogram[k] for k in sorted(histogram)}


def sum_spec_trace_event_count(summaries: list[dict[str, Any]]) -> int:
    return sum(int(summary.get("spec_trace_event_count", 0)) for summary in summaries)


def build_trace_event(
    *,
    rid: str,
    backend: str,
    tokenizer: Any,
    prefix_token_ids: list[int],
    draft_chunk_token_ids: list[int],
    accept_len: int,
    committed_token_ids: list[int],
    step_index: int,
    turn_index: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    prefix_tail = prefix_token_ids[-TRACE_CONTEXT_WINDOW:]
    accepted_token_ids = draft_chunk_token_ids[:accept_len]
    rejected_candidate_token_ids = (
        draft_chunk_token_ids[accept_len:] if accept_len < len(draft_chunk_token_ids) else []
    )
    replacement_token_id = (
        committed_token_ids[accept_len] if len(committed_token_ids) > accept_len else None
    )
    event = {
        "rid": rid,
        "backend": backend,
        "step_index": step_index,
        "turn_index": turn_index,
        "accept_len": accept_len,
        "context_len": len(prefix_token_ids),
        "prefix_token_ids": prefix_tail,
        "prefix_tokens": decode_token_pieces(tokenizer, prefix_tail),
        "prefix_text": decode_token_text(tokenizer, prefix_tail),
        "draft_chunk_token_ids": draft_chunk_token_ids,
        "draft_chunk_tokens": decode_token_pieces(tokenizer, draft_chunk_token_ids),
        "accepted_token_ids": accepted_token_ids,
        "accepted_tokens": decode_token_pieces(tokenizer, accepted_token_ids),
        "rejected_candidate_token_ids": rejected_candidate_token_ids,
        "rejected_candidate_tokens": decode_token_pieces(tokenizer, rejected_candidate_token_ids),
        "committed_token_ids": committed_token_ids,
        "committed_tokens": decode_token_pieces(tokenizer, committed_token_ids),
        "committed_text": decode_token_text(tokenizer, committed_token_ids),
        "replacement_token_id": replacement_token_id,
        "replacement_token": (
            decode_token_pieces(tokenizer, [replacement_token_id])[0]
            if replacement_token_id is not None
            else None
        ),
    }
    if extra:
        event.update(extra)
    return event


def read_jsonl_new_entries(path: Path, start_offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    with path.open("r", encoding="utf-8") as f:
        f.seek(start_offset)
        rows = [json.loads(line) for line in f if line.strip()]
        end_offset = f.tell()
    return rows, end_offset


def load_angelslim_eagle3_model_class() -> Any:
    from eval.angelslim_hunyuan_kv import HunYuanDenseV1ForCausalLMKV

    try:
        eagle3_module = importlib.import_module(
            "angelslim.compressor.speculative.inference.models.eagle3.eagle3_model"
        )
        draft_base_module = importlib.import_module(
            "angelslim.compressor.speculative.inference.models.eagle3.draft.base_model"
        )
        util_module = importlib.import_module("angelslim.compressor.speculative.utils.util")
    except ImportError as exc:
        raise ImportError(
            "Failed to import AngelSlim Eagle3 modules. "
            "Install the benchmark package dependencies in the AngelSlim environment, for example:\n"
            "  uv pip install \"angelslim==0.3.0\" \"transformers==4.57.1\" \"huggingface_hub<1\" "
            "sympy antlr4-python3-runtime pyarrow datasets accelerate threadpoolctl shortuuid safetensors"
        ) from exc

    supported_architectures = eagle3_module.ModelLoader.SUPPORTED_ARCHITECTURES
    supported_architectures["HunYuanDenseV1ForCausalLM"] = HunYuanDenseV1ForCausalLMKV

    drafter_cls = draft_base_module.BaseEagle3Drafter
    if not getattr(drafter_cls, "_local_embed_patch_applied", False):
        original_load_safetensors = drafter_cls._load_safetensors_embed
        original_load_pytorch = drafter_cls._load_pytorch_embed

        def _load_safetensors_embed_local_first(self: Any, path: str) -> None:
            local_emb_path = Path(path) / "model.safetensors"
            if local_emb_path.exists():
                from safetensors import safe_open

                with safe_open(str(local_emb_path), framework="pt", device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    _, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
                self.embed_tokens.weight.data = tensor
                return
            return original_load_safetensors(self, path)

        def _load_pytorch_embed_local_first(self: Any, path: str) -> None:
            local_emb_path = Path(path) / "pytorch_model.bin"
            if local_emb_path.exists():
                import torch

                weights = torch.load(str(local_emb_path), map_location="cpu")
                tensor = weights["model.embed_tokens.weight"].float()
                self.embed_tokens.weight.data = tensor
                return
            return original_load_pytorch(self, path)

        drafter_cls._load_safetensors_embed = _load_safetensors_embed_local_first
        drafter_cls._load_pytorch_embed = _load_pytorch_embed_local_first
        drafter_cls._local_embed_patch_applied = True

    if not getattr(util_module, "_initialize_tree_patch_applied", False):
        def normalize_hidden_states_for_eagle(model: Any, hidden_states: Any) -> list[Any]:
            eagle_device = next(model.eagle_layer.parameters()).device
            return [state.to(eagle_device) for state in hidden_states]

        def initialize_tree_patched(input_ids: Any, inputs_embeds: Any, model: Any, past_key_values: Any, logits_processor: Any) -> Any:
            import torch

            outputs, orig, hidden_states = model(
                input_ids, inputs_embeds, past_key_values=past_key_values, output_orig=True
            )

            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

            add_inputs_embeds = None
            if inputs_embeds is not None:
                add_inputs_embeds = torch.cat(
                    [inputs_embeds, model.eagle_layer.embed_tokens(token)], dim=1
                )

            outputs["hidden_states"] = normalize_hidden_states_for_eagle(
                model, outputs["hidden_states"]
            )
            hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = (
                model.eagle_layer.topK_genrate(
                    hidden_states, input_ids, add_inputs_embeds, logits_processor
                )
            )
            return (
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                orig,
                hidden_states,
                token,
            )

        util_module.initialize_tree = initialize_tree_patched
        eagle3_module.initialize_tree = initialize_tree_patched
        util_module._initialize_tree_patch_applied = True

        def tree_decoding_patched(
            model: Any,
            tree_candidates: Any,
            past_key_values: Any,
            tree_position_ids: Any,
            input_ids: Any,
            retrieve_indices: Any,
        ) -> Any:
            import torch

            position_ids = tree_position_ids + input_ids.shape[1]
            if position_ids is not None and position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            outputs, tree_logits, hidden_state = model(
                input_ids=tree_candidates,
                output_orig=True,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            outputs["hidden_states"] = normalize_hidden_states_for_eagle(
                model, outputs["hidden_states"]
            )
            hidden_state = torch.cat(outputs["hidden_states"], dim=-1)
            logits = tree_logits[0, retrieve_indices]
            return logits, hidden_state, outputs

        util_module.tree_decoding = tree_decoding_patched
        eagle3_module.tree_decoding = tree_decoding_patched

    if not getattr(util_module, "_update_inputs_patch_applied", False):
        def update_inference_inputs_patched(
            input_ids: Any,
            inputs_embeds: Any,
            candidates: Any,
            best_candidate: Any,
            accept_length: Any,
            retrieve_indices: Any,
            logits_processor: Any,
            new_token: Any,
            past_key_values_data_list: Any,
            current_length_data: Any,
            model: Any,
            hidden_state_new: Any,
            sample_token: Any,
        ) -> Any:
            import torch

            if inputs_embeds is not None:
                assert input_ids.shape[1] == inputs_embeds.shape[1]
            prev_input_len = input_ids.shape[1]
            select_indices = retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
            input_ids = torch.cat(
                [
                    input_ids,
                    candidates[None, best_candidate, : accept_length + 1].to(input_ids.device),
                ],
                dim=-1,
            )

            if inputs_embeds is not None:
                add_inputs_embeds = model.eagle_layer.embed_tokens.weight[
                    candidates[None, best_candidate, : accept_length + 1].squeeze(0).tolist()
                ].unsqueeze(0)
                inputs_embeds = torch.cat([inputs_embeds, add_inputs_embeds], dim=1)

            for past_key_values_data in past_key_values_data_list:
                tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
                dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
                dst.copy_(tgt, non_blocking=True)

            current_length_data.fill_(prev_input_len + tgt.shape[-2])

            retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
            accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]

            next_inputs_embeds = None
            if inputs_embeds is not None:
                add_inputs_embeds = model.eagle_layer.embed_tokens.weight[
                    sample_token.squeeze(0).tolist()
                ].unsqueeze(0)
                next_inputs_embeds = torch.cat([inputs_embeds, add_inputs_embeds], dim=1)

            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, early_stop_signal = (
                model.eagle_layer.topK_genrate(
                    accept_hidden_state_new,
                    input_ids=torch.cat((input_ids, sample_token.to(input_ids.device)), dim=1),
                    inputs_embeds=next_inputs_embeds,
                    logits_processor=logits_processor,
                )
            )

            new_token += accept_length + 1

            return (
                input_ids,
                inputs_embeds,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                new_token,
                early_stop_signal,
            )

        util_module.update_inference_inputs = update_inference_inputs_patched
        eagle3_module.update_inference_inputs = update_inference_inputs_patched
        util_module._update_inputs_patch_applied = True

    return eagle3_module.Eagle3Model


def normalized_angelslim_eagle3_dir(draft_dir: Path) -> Path:
    config_path = draft_dir / "config.json"
    if not config_path.exists():
        return draft_dir

    config = json.loads(config_path.read_text(encoding="utf-8"))
    rope_scaling = config.get("rope_scaling")
    normalized = None
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("type")
        factor = rope_scaling.get("factor")
        if rope_type in {"linear", "dynamic"} and isinstance(factor, (int, float)) and float(factor) > 1.0:
            normalized = {"type": rope_type, "factor": float(factor)}

    if rope_scaling == normalized:
        return draft_dir

    patched_dir = ARTIFACTS / "tmp" / f"{draft_dir.name}_angelslim"
    patched_dir.mkdir(parents=True, exist_ok=True)
    config["rope_scaling"] = normalized
    (patched_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for filename in ("model.safetensors", "pytorch_model.bin"):
        src = draft_dir / filename
        dst = patched_dir / filename
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
    return patched_dir


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def sampling_params_for(dataset_name: str) -> dict[str, Any]:
    params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    }
    if dataset_name == "math500":
        params["max_new_tokens"] = DEFAULT_MAX_NEW_TOKENS
    elif dataset_name == "gsm8k":
        params["max_new_tokens"] = DEFAULT_MAX_NEW_TOKENS
    elif dataset_name == "humaneval":
        params["max_new_tokens"] = LONG_MAX_NEW_TOKENS
        params["stop"] = ["```", "\nif __name__ == '__main__':"]
    elif dataset_name in {"ceval", "cmmlu"}:
        params["max_new_tokens"] = DEFAULT_MAX_NEW_TOKENS
    elif dataset_name == "mtbench":
        params["max_new_tokens"] = LONG_MAX_NEW_TOKENS
    return params


def count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def adjusted_sampling_params(
    tokenizer: Any,
    prompt: str,
    dataset_name: str,
    *,
    reserve_tokens: int = 0,
    extra_input_tokens: int = 0,
) -> tuple[dict[str, Any], int]:
    params = sampling_params_for(dataset_name).copy()
    requested_max_new_tokens = int(params["max_new_tokens"])
    prompt_tokens = count_prompt_tokens(tokenizer, prompt)
    available = (
        (EVAL_CONTEXT_LENGTH - 1)
        - prompt_tokens
        - extra_input_tokens
        - PROMPT_TOKEN_SAFETY_MARGIN
        - reserve_tokens
    )
    params["max_new_tokens"] = max(1, min(requested_max_new_tokens, available))
    return params, prompt_tokens


def extract_final_number(text: str) -> Optional[str]:
    if not text:
        return None
    boxed = BOXED_RE.findall(text)
    if boxed:
        nums = NUM_RE.findall(boxed[-1].replace("$", ""))
        if nums:
            return nums[-1].replace(",", "").rstrip(".")
    match = re.search(r"Final answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    source = match.group(1) if match else text
    nums = NUM_RE.findall(source.replace("$", ""))
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def normalize_math_text(text: str) -> str:
    text = text.strip()
    boxed = BOXED_RE.findall(text)
    if boxed:
        text = boxed[-1]
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("$", "").replace(" ", "")
    text = text.strip(".")
    return text


def parse_sympy_expr(text: str) -> Optional[Any]:
    from sympy import sympify
    from sympy.parsing.latex import parse_latex

    text = normalize_math_text(text)
    if not text:
        return None
    for parser in (parse_latex, sympify):
        try:
            return parser(text)
        except Exception:
            continue
    return None


def math_equiv(pred: str, gold: str) -> bool:
    pred_norm = normalize_math_text(pred)
    gold_norm = normalize_math_text(gold)
    if pred_norm == gold_norm:
        return True
    pred_expr = parse_sympy_expr(pred_norm)
    gold_expr = parse_sympy_expr(gold_norm)
    if pred_expr is None or gold_expr is None:
        return False
    try:
        return bool((pred_expr - gold_expr).equals(0))
    except Exception:
        return False


def extract_letter_answer(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"Final answer\s*:\s*([A-D])", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    letters = LETTER_RE.findall(text.upper())
    return letters[-1].upper() if letters else None


def extract_code_completion(text: str) -> str:
    if not text:
        return ""
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip("\n")
    sanitized = text.replace("<answer>", "").replace("</answer>", "").strip()
    lines = sanitized.splitlines()
    code_lines: list[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        is_code = (
            not stripped
            or line.startswith((" ", "\t"))
            or bool(CODE_LINE_RE.match(line))
        )
        if not started:
            if is_code and stripped:
                started = True
                code_lines.append(line)
            continue
        if is_code:
            code_lines.append(line)
            continue
        if code_lines and not code_lines[-1].strip():
            break
        if code_lines:
            break
    if code_lines:
        return "\n".join(code_lines).strip("\n")
    return sanitized


def worker_run_humaneval(program: str, entry_point: str, test: str, timeout: int, result_path: Path) -> None:
    try:
        namespace: dict[str, Any] = {}
        exec(program, namespace)
        exec(test, namespace)
        namespace["check"](namespace[entry_point])
        result_path.write_text("pass", encoding="utf-8")
    except Exception as exc:
        result_path.write_text(f"fail:{type(exc).__name__}:{exc}", encoding="utf-8")


def humaneval_passes(sample: dict[str, Any], completion: str, timeout: int = 10) -> tuple[bool, str]:
    import tempfile

    completion = extract_code_completion(completion)
    program = sample["prompt"] + completion + "\n"
    with tempfile.TemporaryDirectory(prefix="humaneval_") as tmpdir:
        result_path = Path(tmpdir) / "result.txt"
        ctx = get_context("spawn")
        proc = ctx.Process(
            target=worker_run_humaneval,
            args=(program, sample["entry_point"], sample["test"], timeout, result_path),
        )
        proc.start()
        proc.join(timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return False, "timeout"
        if result_path.exists():
            result = result_path.read_text(encoding="utf-8").strip() or "empty"
            if result == "pass":
                return True, "pass"
            return False, result
        return False, f"exitcode={proc.exitcode}"


def score_sample(dataset_name: str, sample: dict[str, Any], output: Any) -> dict[str, Any]:
    if dataset_name == "mtbench":
        return {"metric_name": "generation_only", "score": None}

    if dataset_name == "gsm8k":
        pred = extract_final_number(output["text"])
        gold = extract_final_number(sample["gold_answer"])
        correct = pred == gold and pred is not None
        return {
            "metric_name": "exact_match",
            "score": 1.0 if correct else 0.0,
            "predicted_answer": pred,
            "gold_answer": gold,
        }

    if dataset_name == "math500":
        pred_match = re.search(r"Final answer\s*:\s*(.+)", output["text"], flags=re.IGNORECASE)
        pred = pred_match.group(1).strip() if pred_match else output["text"].strip()
        gold = sample["gold_answer"]
        correct = math_equiv(pred, gold)
        return {
            "metric_name": "math_equiv",
            "score": 1.0 if correct else 0.0,
            "predicted_answer": pred,
            "gold_answer": gold,
        }

    if dataset_name in {"ceval", "cmmlu"}:
        pred = extract_letter_answer(output["text"])
        gold = sample["gold_answer"]
        correct = pred == gold
        return {
            "metric_name": "accuracy",
            "score": 1.0 if correct else 0.0,
            "predicted_answer": pred,
            "gold_answer": gold,
        }

    if dataset_name == "humaneval":
        passed, reason = humaneval_passes(sample, output["text"])
        return {
            "metric_name": "pass@1",
            "score": 1.0 if passed else 0.0,
            "judge": reason,
            "completion": extract_code_completion(output["text"]),
        }

    raise ValueError(dataset_name)


def summarise_dataset_results(
    model_key: str,
    dataset_name: str,
    results: list[dict[str, Any]],
    trace_events: list[dict[str, Any]],
) -> dict[str, Any]:
    scored = [r["score"] for r in results if r["score"] is not None]
    summary: dict[str, Any] = {
        "model_key": model_key,
        "dataset": dataset_name,
        "sample_count": len(results),
        "scored_count": len(scored),
        "mean_score": (sum(scored) / len(scored)) if scored else None,
        "metric_name": results[0]["metric_name"] if results else None,
    }

    meta_scores = [r.get("spec_accept_length") for r in results if r.get("spec_accept_length") is not None]
    if meta_scores:
        summary["mean_request_accept_length"] = sum(meta_scores) / len(meta_scores)

    histogram = Counter()
    for ev in trace_events:
        if ev.get("accept_len") is not None:
            histogram[int(ev["accept_len"])] += 1
        elif ev.get("accept_length_histogram"):
            for key, value in ev["accept_length_histogram"].items():
                histogram[int(key)] += int(value)
    if not histogram:
        for row in results:
            meta_hist = row.get("meta_info", {}).get("accept_length_histogram")
            if not meta_hist:
                continue
            for key, value in meta_hist.items():
                histogram[int(key)] += int(value)
    summary["accept_length_histogram"] = {str(k): histogram[k] for k in sorted(histogram)}
    summary["context_accept_length"] = context_accept_length_stats(trace_events)
    summary["spec_trace_event_count"] = len(trace_events)
    return summary


def build_result_record(
    model_spec: ModelSpec,
    dataset_name: str,
    sample: dict[str, Any],
    rid: str,
    response: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    meta = response.get("meta_info", {})
    return {
        "model_key": model_spec.key,
        "model_name": model_spec.display_name,
        "backend": model_spec.backend,
        "dataset": dataset_name,
        "sample_id": sample["sample_id"],
        "rid": rid,
        "text": response.get("text", ""),
        "meta_info": meta,
        "spec_accept_length": meta.get("spec_accept_length"),
        "spec_accept_rate": meta.get("spec_accept_rate"),
        "spec_verify_ct": meta.get("spec_verify_ct"),
        **score,
    }


def sampling_params_for_vllm(dataset_name: str) -> dict[str, Any]:
    params = sampling_params_for(dataset_name).copy()
    max_new_tokens = params.pop("max_new_tokens")
    params["max_tokens"] = max_new_tokens
    return params


def adjusted_sampling_params_for_vllm(
    tokenizer: Any,
    prompt: str,
    dataset_name: str,
    *,
    reserve_tokens: int = 0,
) -> tuple[dict[str, Any], int]:
    params, prompt_tokens = adjusted_sampling_params(
        tokenizer,
        prompt,
        dataset_name,
        reserve_tokens=reserve_tokens,
    )
    max_new_tokens = params.pop("max_new_tokens")
    params["max_tokens"] = max_new_tokens
    return params, prompt_tokens


def adjusted_sampling_params_for_sglang(
    tokenizer: Any,
    prompt: str,
    dataset_name: str,
    *,
    reserve_tokens: int = 0,
) -> tuple[dict[str, Any], int]:
    return adjusted_sampling_params(
        tokenizer,
        prompt,
        dataset_name,
        reserve_tokens=reserve_tokens,
        extra_input_tokens=SGLANG_INPUT_TOKEN_FUDGE + SGLANG_TOKEN_SAFETY_MARGIN,
    )


def snapshot_vllm_metrics(llm: Any) -> dict[str, Any]:
    try:
        metrics = llm.get_metrics()
    except AssertionError:
        return {}

    snapshot: dict[str, Any] = {}
    for metric in metrics:
        if hasattr(metric, "values"):
            snapshot[metric.name] = list(metric.values)
        elif hasattr(metric, "value"):
            snapshot[metric.name] = metric.value
        elif hasattr(metric, "count") and hasattr(metric, "buckets"):
            snapshot[metric.name] = {
                "count": int(metric.count),
                "sum": float(metric.sum),
                "buckets": {str(k): int(v) for k, v in metric.buckets.items()},
            }
    return snapshot


def diff_vllm_metric_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
    num_spec_tokens: int,
) -> dict[str, Any]:
    def get_counter(name: str) -> int:
        return int(after.get(name, 0)) - int(before.get(name, 0))

    def get_vector(name: str) -> list[int]:
        a_vals = list(after.get(name, [0] * num_spec_tokens))
        b_vals = list(before.get(name, [0] * num_spec_tokens))
        if len(a_vals) < num_spec_tokens:
            a_vals.extend([0] * (num_spec_tokens - len(a_vals)))
        if len(b_vals) < num_spec_tokens:
            b_vals.extend([0] * (num_spec_tokens - len(b_vals)))
        return [int(a_vals[i]) - int(b_vals[i]) for i in range(num_spec_tokens)]

    num_drafts = get_counter("vllm:spec_decode_num_drafts")
    num_draft_tokens = get_counter("vllm:spec_decode_num_draft_tokens")
    num_accepted_tokens = get_counter("vllm:spec_decode_num_accepted_tokens")
    pos_counts = get_vector("vllm:spec_decode_num_accepted_tokens_per_pos")
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "acceptance_counts": pos_counts,
        "mean_acceptance_length": (num_accepted_tokens / num_drafts) if num_drafts > 0 else 0.0,
    }


def acceptance_histogram_from_pos_counts(
    num_drafts: int,
    pos_counts: list[int],
) -> dict[str, int]:
    histogram: dict[str, int] = {}
    prev = num_drafts
    for pos, count in enumerate(pos_counts):
        exact = prev - count
        if exact > 0:
            histogram[str(pos)] = exact
        prev = count
    if prev > 0:
        histogram[str(len(pos_counts))] = prev
    if prev < 0:
        raise ValueError(f"Invalid acceptance histogram state: num_drafts={num_drafts}, pos_counts={pos_counts}")
    return histogram


def build_vllm_trace_record(
    rid: str,
    delta: dict[str, Any],
) -> dict[str, Any]:
    histogram = acceptance_histogram_from_pos_counts(
        num_drafts=delta["num_drafts"],
        pos_counts=delta["acceptance_counts"],
    )
    return {
        "rid": rid,
        "backend": BACKEND_VLLM,
        "num_drafts": delta["num_drafts"],
        "num_draft_tokens": delta["num_draft_tokens"],
        "num_accepted_tokens": delta["num_accepted_tokens"],
        "mean_acceptance_length": delta["mean_acceptance_length"],
        "acceptance_counts": delta["acceptance_counts"],
        "accept_length_histogram": histogram,
        "accepted_token_ids": None,
        "accepted_tokens": None,
        "rejected_candidate_token_ids": None,
        "rejected_candidate_tokens": None,
        "draft_chunk_token_ids": None,
        "draft_chunk_tokens": None,
    }


def build_vllm_meta_info(delta: dict[str, Any]) -> dict[str, Any]:
    num_drafts = delta["num_drafts"]
    num_draft_tokens = delta["num_draft_tokens"]
    num_accepted_tokens = delta["num_accepted_tokens"]
    spec_accept_rate = (num_accepted_tokens / num_draft_tokens) if num_draft_tokens > 0 else None
    return {
        "spec_accept_length": delta["mean_acceptance_length"],
        "spec_accept_rate": spec_accept_rate,
        "spec_verify_ct": num_drafts,
        "vllm_num_drafts": num_drafts,
        "vllm_num_draft_tokens": num_draft_tokens,
        "vllm_num_accepted_tokens": num_accepted_tokens,
        "vllm_acceptance_counts": delta["acceptance_counts"],
    }


def flatten_trace_for_rids(trace_path: Path, request_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(trace_path):
        if row.get("rid") in request_ids:
            rows.append(row)
    return rows


def run_single_sample(
    engine: Any,
    tokenizer: Any,
    model_spec: ModelSpec,
    dataset_name: str,
    sample: dict[str, Any],
) -> dict[str, Any]:
    rid = f"{dataset_name}::{safe_name(str(sample['sample_id']))}"

    if dataset_name == "mtbench":
        messages = build_prompt_messages(dataset_name, sample)
        turns = []
        for turn_index, user_turn in enumerate(sample["turns"]):
            if turn_index == 0:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_turn},
                ]
            else:
                messages.append({"role": "user", "content": user_turn})
            prompt = render_chat_prompt(tokenizer, messages)
            sampling_params, prompt_tokens = adjusted_sampling_params_for_sglang(
                tokenizer,
                prompt,
                "mtbench",
                reserve_tokens=MTBENCH_TURN1_HISTORY_RESERVE if turn_index == 0 else 0,
            )
            response = engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                rid=f"{rid}:turn{turn_index+1}",
            )
            assistant_text = response["text"]
            meta_info = dict(response.get("meta_info", {}))
            meta_info["prompt_tokens"] = prompt_tokens
            meta_info["max_new_tokens"] = sampling_params["max_new_tokens"]
            turns.append({"turn_index": turn_index + 1, "text": assistant_text, "meta_info": meta_info})
            messages.append({"role": "assistant", "content": assistant_text})
        return {
            "rid": rid,
            "text": turns[-1]["text"] if turns else "",
            "meta_info": turns[-1]["meta_info"] if turns else {},
            "turns": turns,
        }

    messages = build_prompt_messages(dataset_name, sample)
    prompt = render_chat_prompt(tokenizer, messages)
    sampling_params, prompt_tokens = adjusted_sampling_params_for_sglang(
        tokenizer,
        prompt,
        dataset_name,
    )
    response = engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        rid=rid,
    )
    response["rid"] = rid
    response["meta_info"] = dict(response.get("meta_info", {}))
    response["meta_info"]["prompt_tokens"] = prompt_tokens
    response["meta_info"]["max_new_tokens"] = sampling_params["max_new_tokens"]
    return response


def evaluate_model_sglang(
    model_key: str,
    gpu_ids: list[int],
    sample_size: int,
    seed: int,
    cmmlu_repo: str,
    dataset_names: list[str],
) -> dict[str, Any]:
    del seed, cmmlu_repo
    os.environ["CUDA_VISIBLE_DEVICES"] = format_gpu_ids(gpu_ids)
    os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    os.environ["EAGLE3_TRACE_BACKEND"] = "sglang"
    os.environ["EAGLE3_TRACE_CONTEXT_WINDOW"] = str(TRACE_CONTEXT_WINDOW)
    prepend_pythonpath(ROOT)

    from transformers import AutoTokenizer
    from sglang.srt.entrypoints.engine import Engine
    from eval import sitecustomize

    model_spec = model_specs([model_key])[0]
    ensure_sliding_window_specforge_available(model_spec)
    speculative_num_steps = SPEC_NUM_STEPS
    speculative_eagle_topk = SPEC_EAGLE_TOPK
    speculative_num_draft_tokens = SPEC_NUM_DRAFT_TOKENS
    speculative_draft_attention_backend = None
    if model_spec.requires_sliding_window_specforge:
        spec_config = sliding_window_speculative_config(model_spec.draft_local_dir)
        speculative_num_steps = spec_config["speculative_num_steps"]
        speculative_eagle_topk = spec_config["speculative_eagle_topk"]
        speculative_num_draft_tokens = spec_config["speculative_num_draft_tokens"]
        speculative_draft_attention_backend = SGLANG_SW_DRAFT_ATTENTION_BACKEND
        os.environ["EAGLE3_SLIDING_WINDOW_EAGLE_PATCH"] = "1"
        sitecustomize.install_sglang_sliding_window_eagle_patch()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_spec.base_local_dir),
        trust_remote_code=True,
    )

    model_log_dir = LOGS_DIR / model_spec.key
    if model_log_dir.exists():
        shutil.rmtree(model_log_dir)
    model_log_dir.mkdir(parents=True, exist_ok=True)
    trace_path = model_log_dir / "spec_trace_raw.jsonl"
    os.environ["EAGLE3_TRACE_TOKENIZER_PATH"] = str(model_spec.base_local_dir)
    os.environ["EAGLE3_TRACE_PATH"] = str(trace_path)
    sitecustomize.install_sglang_trace_patch()

    engine = Engine(
        model_path=str(model_spec.base_local_dir),
        tokenizer_path=str(model_spec.base_local_dir),
        speculative_draft_model_path=str(model_spec.draft_local_dir),
        speculative_algorithm="EAGLE3",
        speculative_num_steps=speculative_num_steps,
        speculative_eagle_topk=speculative_eagle_topk,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        speculative_draft_attention_backend=speculative_draft_attention_backend,
        enable_multi_layer_eagle=False,
        trust_remote_code=True,
        mem_fraction_static=0.72,
        page_size=1,
        context_length=EVAL_CONTEXT_LENGTH,
        tp_size=len(gpu_ids),
    )

    combined_path = model_log_dir / "combined_results.jsonl"
    summaries = []
    sample_paths = {name: SAMPLES_DIR / f"{name}.jsonl" for name in dataset_names}
    missing = [str(path) for path in sample_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared sample files. Run `python -m eval.prepare_data` first. Missing: "
            + ", ".join(missing)
        )
    try:
        for dataset_name in dataset_names:
            dataset_rows = read_jsonl(sample_paths[dataset_name])[:sample_size]
            request_ids: set[str] = set()
            result_rows: list[dict[str, Any]] = []
            dataset_dir = model_log_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            results_path = dataset_dir / "results.jsonl"
            if results_path.exists():
                results_path.unlink()

            for sample in dataset_rows:
                response = run_single_sample(engine, tokenizer, model_spec, dataset_name, sample)
                if dataset_name == "mtbench":
                    for turn in response["turns"]:
                        request_ids.add(f"{response['rid']}:turn{turn['turn_index']}")
                    score = {"metric_name": "generation_only", "score": None}
                else:
                    request_ids.add(response["rid"])
                    score = score_sample(dataset_name, sample, response)

                record = build_result_record(
                    model_spec=model_spec,
                    dataset_name=dataset_name,
                    sample=sample,
                    rid=response["rid"],
                    response=response,
                    score=score,
                )
                record["meta_info"] = normalize_sglang_meta_info(record["meta_info"])
                record["spec_accept_length"] = record["meta_info"].get("spec_accept_length")
                record["spec_accept_rate"] = record["meta_info"].get("spec_accept_rate")
                record["spec_verify_ct"] = record["meta_info"].get("spec_verify_ct")
                if dataset_name == "mtbench":
                    record["turns"] = response["turns"]
                append_jsonl(results_path, record)
                append_jsonl(combined_path, record)
                result_rows.append(record)

            trace_events = flatten_trace_for_rids(trace_path, request_ids)
            trace_out = dataset_dir / "spec_trace.jsonl"
            with trace_out.open("w", encoding="utf-8") as f:
                for event in trace_events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")

            summary = summarise_dataset_results(
                model_key=model_spec.key,
                dataset_name=dataset_name,
                results=result_rows,
                trace_events=trace_events,
            )
            json_dump(dataset_dir / "summary.json", summary)
            summaries.append(summary)
    finally:
        engine.shutdown()

    final_summary = {
        "model_key": model_spec.key,
        "model_name": model_spec.display_name,
        "backend": model_spec.backend,
        "gpu_ids": gpu_ids,
        "speculative_config": {
            "speculative_num_steps": speculative_num_steps,
            "speculative_eagle_topk": speculative_eagle_topk,
            "speculative_num_draft_tokens": speculative_num_draft_tokens,
            "speculative_draft_attention_backend": speculative_draft_attention_backend,
            "context_length": EVAL_CONTEXT_LENGTH,
            "tensor_parallel_size": len(gpu_ids),
        },
        "datasets": summaries,
        "accept_length_histogram": merge_accept_length_histogram(summaries),
        "context_accept_length": merge_context_accept_length(summaries),
        "spec_trace_event_count": sum_spec_trace_event_count(summaries),
    }
    json_dump(model_log_dir / "model_summary.json", final_summary)
    return final_summary


def evaluate_model_sglang_dflash(
    model_key: str,
    gpu_ids: list[int],
    sample_size: int,
    seed: int,
    cmmlu_repo: str,
    dataset_names: list[str],
) -> dict[str, Any]:
    del seed, cmmlu_repo
    os.environ["CUDA_VISIBLE_DEVICES"] = format_gpu_ids(gpu_ids)
    os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    os.environ["EAGLE3_TRACE_BACKEND"] = "sglang"
    os.environ["EAGLE3_TRACE_DFLASH"] = "1"
    os.environ["EAGLE3_TRACE_CONTEXT_WINDOW"] = str(TRACE_CONTEXT_WINDOW)
    prepend_pythonpath(ROOT)

    from transformers import AutoTokenizer
    from sglang.srt.entrypoints.engine import Engine
    from eval import sitecustomize

    model_spec = model_specs([model_key])[0]
    config_summary = validate_dflash_config(
        model_spec.base_local_dir,
        model_spec.draft_local_dir,
        expected_block_size=16,
    )
    block_size = int(config_summary["block_size"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_spec.base_local_dir),
        trust_remote_code=True,
    )

    model_log_dir = LOGS_DIR / model_spec.key
    if model_log_dir.exists():
        shutil.rmtree(model_log_dir)
    model_log_dir.mkdir(parents=True, exist_ok=True)
    trace_path = model_log_dir / "spec_trace_raw.jsonl"
    os.environ["EAGLE3_TRACE_TOKENIZER_PATH"] = str(model_spec.base_local_dir)
    os.environ["EAGLE3_TRACE_PATH"] = str(trace_path)
    sitecustomize.install_sglang_dflash_trace_patch()

    engine = Engine(
        model_path=str(model_spec.base_local_dir),
        tokenizer_path=str(model_spec.base_local_dir),
        speculative_draft_model_path=str(model_spec.draft_local_dir),
        speculative_algorithm="DFLASH",
        speculative_dflash_block_size=block_size,
        speculative_num_draft_tokens=block_size,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        dtype="bfloat16",
        attention_backend="fa3",
        trust_remote_code=True,
        mem_fraction_static=0.75,
        context_length=EVAL_CONTEXT_LENGTH,
        tp_size=len(gpu_ids),
    )

    combined_path = model_log_dir / "combined_results.jsonl"
    summaries = []
    sample_paths = {name: SAMPLES_DIR / f"{name}.jsonl" for name in dataset_names}
    missing = [str(path) for path in sample_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared sample files. Run `python -m eval.prepare_data` first. Missing: "
            + ", ".join(missing)
        )
    try:
        for dataset_name in dataset_names:
            dataset_rows = read_jsonl(sample_paths[dataset_name])[:sample_size]
            request_ids: set[str] = set()
            result_rows: list[dict[str, Any]] = []
            dataset_dir = model_log_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            results_path = dataset_dir / "results.jsonl"
            if results_path.exists():
                results_path.unlink()

            for sample in dataset_rows:
                response = run_single_sample(engine, tokenizer, model_spec, dataset_name, sample)
                if dataset_name == "mtbench":
                    for turn in response["turns"]:
                        request_ids.add(f"{response['rid']}:turn{turn['turn_index']}")
                    score = {"metric_name": "generation_only", "score": None}
                else:
                    request_ids.add(response["rid"])
                    score = score_sample(dataset_name, sample, response)

                record = build_result_record(
                    model_spec=model_spec,
                    dataset_name=dataset_name,
                    sample=sample,
                    rid=response["rid"],
                    response=response,
                    score=score,
                )
                record["meta_info"] = normalize_sglang_meta_info(record["meta_info"])
                record["spec_accept_length"] = record["meta_info"].get("spec_accept_length")
                record["spec_accept_rate"] = record["meta_info"].get("spec_accept_rate")
                record["spec_verify_ct"] = record["meta_info"].get("spec_verify_ct")
                if dataset_name == "mtbench":
                    record["turns"] = response["turns"]
                append_jsonl(results_path, record)
                append_jsonl(combined_path, record)
                result_rows.append(record)

            trace_events = flatten_trace_for_rids(trace_path, request_ids)
            trace_out = dataset_dir / "spec_trace.jsonl"
            with trace_out.open("w", encoding="utf-8") as f:
                for event in trace_events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")

            summary = summarise_dataset_results(
                model_key=model_spec.key,
                dataset_name=dataset_name,
                results=result_rows,
                trace_events=trace_events,
            )
            json_dump(dataset_dir / "summary.json", summary)
            summaries.append(summary)
    finally:
        engine.shutdown()

    final_summary = {
        "model_key": model_spec.key,
        "model_name": model_spec.display_name,
        "backend": model_spec.backend,
        "gpu_ids": gpu_ids,
        "speculative_config": {
            "speculative_algorithm": "DFLASH",
            "speculative_dflash_block_size": block_size,
            "speculative_num_draft_tokens": block_size,
            "speculative_num_steps": 1,
            "speculative_eagle_topk": 1,
            "context_length": EVAL_CONTEXT_LENGTH,
            "tensor_parallel_size": len(gpu_ids),
        },
        "config": config_summary,
        "datasets": summaries,
        "accept_length_histogram": merge_accept_length_histogram(summaries),
        "context_accept_length": merge_context_accept_length(summaries),
        "spec_trace_event_count": sum_spec_trace_event_count(summaries),
    }
    json_dump(model_log_dir / "model_summary.json", final_summary)
    return final_summary


def run_sliding_window_sglang_runtime_smoke(
    model_spec: ModelSpec,
    draft_model_path: Path,
    gpu_ids: list[int],
    model_log_dir: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from sglang.srt.entrypoints.engine import Engine
    from eval import sitecustomize

    os.environ["CUDA_VISIBLE_DEVICES"] = format_gpu_ids(gpu_ids)
    os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    os.environ["EAGLE3_SLIDING_WINDOW_EAGLE_PATCH"] = "1"
    os.environ["EAGLE3_TRACE_BACKEND"] = "sglang"
    os.environ["EAGLE3_TRACE_TOKENIZER_PATH"] = str(model_spec.base_local_dir)
    os.environ["EAGLE3_TRACE_PATH"] = str(model_log_dir / "sliding_window_runtime_trace.jsonl")
    os.environ["EAGLE3_TRACE_CONTEXT_WINDOW"] = str(TRACE_CONTEXT_WINDOW)

    sitecustomize.install_sglang_sliding_window_eagle_patch()
    sitecustomize.install_sglang_trace_patch()
    spec_config = sliding_window_speculative_config(draft_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_spec.base_local_dir),
        trust_remote_code=True,
    )
    prompt = render_chat_prompt(
        tokenizer,
        [{"role": "user", "content": "Compute 12 + 30. Answer with only the number."}],
    )
    engine = Engine(
        model_path=str(model_spec.base_local_dir),
        tokenizer_path=str(model_spec.base_local_dir),
        speculative_draft_model_path=str(draft_model_path),
        speculative_algorithm="EAGLE3",
        speculative_num_steps=spec_config["speculative_num_steps"],
        speculative_eagle_topk=spec_config["speculative_eagle_topk"],
        speculative_num_draft_tokens=spec_config["speculative_num_draft_tokens"],
        speculative_draft_attention_backend=SGLANG_SW_DRAFT_ATTENTION_BACKEND,
        enable_multi_layer_eagle=False,
        trust_remote_code=True,
        mem_fraction_static=0.72,
        page_size=1,
        context_length=EVAL_CONTEXT_LENGTH,
        tp_size=len(gpu_ids),
    )
    try:
        response = engine.generate(
            prompt=prompt,
            sampling_params={"temperature": 0.0, "max_new_tokens": 8},
            rid="sliding-window-runtime-smoke",
        )
    finally:
        engine.shutdown()

    meta_info = normalize_sglang_meta_info(response.get("meta_info", {}))
    return {
        "loaded": True,
        "output_text": response.get("text", ""),
        "meta_info": meta_info,
        **spec_config,
        "speculative_draft_attention_backend": SGLANG_SW_DRAFT_ATTENTION_BACKEND,
        "enable_multi_layer_eagle": False,
    }


def run_sliding_window_smoke_test(
    model_key: str,
    gpu_ids: list[int],
    seq_length: int,
    seed: int,
    draft_model_path: Optional[Path] = None,
    expected_num_layers: Optional[int] = None,
    expected_sliding_window: Optional[int] = None,
    run_runtime: bool = True,
) -> dict[str, Any]:
    if len(gpu_ids) != 1:
        raise ValueError("test-sliding-window currently expects exactly one GPU.")
    if seq_length < 2:
        raise ValueError("test-sliding-window requires seq_length >= 2.")

    model_spec = model_specs([model_key])[0]
    if not model_spec.requires_sliding_window_specforge:
        raise ValueError(f"{model_key} is not configured as a sliding-window model.")

    draft_model_path = (
        Path(draft_model_path) if draft_model_path else model_spec.draft_local_dir
    )
    config = draft_model_config(draft_model_path)
    num_layers = int(config.get("num_hidden_layers", 1))
    sliding_window = config.get("sliding_window")
    max_window_layers = config.get("max_window_layers")
    use_sliding_window = bool(config.get("use_sliding_window", False))
    if num_layers < 2:
        raise RuntimeError(f"Expected a multi-layer draft model, found {num_layers} layer.")
    if expected_num_layers is not None and num_layers != expected_num_layers:
        raise RuntimeError(
            f"Expected {expected_num_layers} draft layers, found {num_layers}."
        )
    if sliding_window is None:
        raise RuntimeError("Expected sliding_window in draft config.")
    sliding_window = int(sliding_window)
    if (
        expected_sliding_window is not None
        and sliding_window != expected_sliding_window
    ):
        raise RuntimeError(
            f"Expected sliding_window={expected_sliding_window}, found {sliding_window}."
        )
    if not use_sliding_window:
        raise RuntimeError("Expected use_sliding_window=true in draft config.")
    if max_window_layers is None:
        raise RuntimeError("Expected max_window_layers in draft config.")
    max_window_layers = int(max_window_layers)
    if max_window_layers != num_layers:
        raise RuntimeError(
            f"Expected max_window_layers={num_layers}, found {max_window_layers}."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = format_gpu_ids(gpu_ids)
    os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    specforge_path = sliding_window_specforge_source_path(draft_model_path)
    if specforge_path is not None:
        prepend_import_path(specforge_path)
        clear_imported_specforge()
    prepend_import_path(ROOT)
    os.environ["EAGLE3_SLIDING_WINDOW_EAGLE_PATCH"] = "1"

    from eval import sitecustomize

    sitecustomize.install_sglang_sliding_window_eagle_patch()
    sglang_eagle3_module = importlib.import_module("sglang.srt.models.llama_eagle3")
    sglang_patch_installed = bool(
        getattr(
            sglang_eagle3_module.LlamaForCausalLMEagle3,
            "_sliding_window_eagle3_patched",
            False,
        )
    )
    if not sglang_patch_installed:
        raise RuntimeError("SGLang Eagle3 patch for internal multi-layer draft models was not installed.")
    sglang_window_method = getattr(
        sglang_eagle3_module.LlamaForCausalLMEagle3,
        "get_attention_sliding_window_size",
        None,
    )
    if sglang_window_method is None:
        raise RuntimeError("SGLang Eagle3 patch did not expose get_attention_sliding_window_size().")

    class _WindowProbe:
        config = type(
            "Config",
            (),
            {"use_sliding_window": use_sliding_window, "sliding_window": sliding_window},
        )()

    sglang_attention_sliding_window_size = sglang_window_method(_WindowProbe())
    if sglang_attention_sliding_window_size != sliding_window - 1:
        raise RuntimeError(
            "Unexpected SGLang attention sliding-window size: "
            f"{sglang_attention_sliding_window_size}"
        )

    try:
        module = importlib.import_module("specforge.modeling.draft.llama3_eagle")
    except ImportError as exc:
        raise ImportError(
            "Sliding-window smoke requires the modified SpecForge package. "
            "Run this command in an environment with the feat/sliding-window checkout installed."
        ) from exc
    model_cls = getattr(module, "LlamaForCausalLMEagle3", None)
    if model_cls is None or not hasattr(model_cls, "_layer_caches"):
        raise ImportError(
            "The selected SpecForge package does not expose multi-layer sliding-window Eagle3. "
            f"Loaded module from {getattr(module, '__file__', 'unknown')}."
        )

    import torch

    from specforge.modeling.auto import AutoEagle3DraftModel

    if not torch.cuda.is_available():
        raise RuntimeError("test-sliding-window requires CUDA.")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    draft_model = AutoEagle3DraftModel.from_pretrained(
        str(draft_model_path),
        dtype=dtype,
        attention_backend="sdpa",
    ).to(device)
    draft_model.eval()

    if int(getattr(draft_model, "num_draft_layers", 1)) != num_layers:
        raise RuntimeError(
            "Loaded draft model did not instantiate the expected number of layers."
        )
    if (
        not hasattr(draft_model, "layers")
        or len(draft_model.layers) != num_layers
    ):
        raise RuntimeError("Loaded draft model does not expose the expected layers ModuleList.")
    layer_windows = [
        getattr(layer.self_attn, "sliding_window", None) for layer in draft_model.layers
    ]
    if layer_windows != [sliding_window] * num_layers:
        raise RuntimeError(f"Unexpected per-layer sliding windows: {layer_windows}")

    layer_forward_counts = [0 for _ in range(num_layers)]
    original_forwards = []
    for layer_idx, layer in enumerate(draft_model.layers):
        original_forward = layer.forward
        original_forwards.append(original_forward)

        def counted_forward(
            *args: Any,
            _idx: int = layer_idx,
            _forward: Any = original_forward,
            **kwargs: Any,
        ) -> Any:
            layer_forward_counts[_idx] += 1
            return _forward(*args, **kwargs)

        layer.forward = counted_forward

    vocab_size = int(getattr(draft_model.config, "vocab_size", 0))
    if vocab_size <= 0:
        raise RuntimeError(f"Invalid draft model vocab_size={vocab_size}.")
    input_high = min(vocab_size, 32000)
    input_ids = torch.randint(
        low=0,
        high=input_high,
        size=(1, seq_length),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones((1, seq_length), dtype=torch.long, device=device)
    hidden_width = int(draft_model.fc.in_features)
    hidden_states = torch.randn(
        (1, seq_length, hidden_width),
        dtype=dtype,
        device=device,
    )
    inputs_embeds = draft_model.embed_input_ids(input_ids).to(dtype)

    try:
        with torch.no_grad():
            draft_hidden = draft_model(
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                ttt_length=2,
            )
            draft_logits = draft_model.lm_head(draft_hidden)
    finally:
        for layer, original_forward in zip(draft_model.layers, original_forwards):
            layer.forward = original_forward

    if layer_forward_counts != [1] * num_layers:
        raise RuntimeError(
            f"Expected each draft layer to run once, got {layer_forward_counts}."
        )
    if not torch.isfinite(draft_hidden.float()).all():
        raise RuntimeError("Draft hidden states contain non-finite values.")
    if not torch.isfinite(draft_logits.float()).all():
        raise RuntimeError("Draft logits contain non-finite values.")

    model_num_draft_layers = int(draft_model.num_draft_layers)
    hidden_states_shape = list(hidden_states.shape)
    draft_hidden_shape = list(draft_hidden.shape)
    draft_logits_shape = list(draft_logits.shape)
    draft_hidden_abs_mean = float(draft_hidden.float().abs().mean().item())
    draft_logits_abs_mean = float(draft_logits.float().abs().mean().item())

    del draft_logits, draft_hidden, hidden_states, inputs_embeds, draft_model
    torch.cuda.empty_cache()

    model_log_dir = LOGS_DIR / model_spec.key
    model_log_dir.mkdir(parents=True, exist_ok=True)
    runtime_summary = None
    if run_runtime:
        runtime_summary = run_sliding_window_sglang_runtime_smoke(
            model_spec=model_spec,
            draft_model_path=draft_model_path,
            gpu_ids=gpu_ids,
            model_log_dir=model_log_dir,
        )

    summary = {
        "backend": "specforge_sliding_window_direct_and_sglang_runtime",
        "model": model_key,
        "specforge_module": getattr(module, "__file__", None),
        "specforge_source_path": str(specforge_path) if specforge_path is not None else None,
        "sglang_eagle3_module": getattr(sglang_eagle3_module, "__file__", None),
        "sglang_internal_layers_patch": sglang_patch_installed,
        "sglang_attention_sliding_window_size": sglang_attention_sliding_window_size,
        "draft_model_path": str(draft_model_path),
        "gpu_ids": gpu_ids,
        "seq_length": seq_length,
        "expected_num_layers": expected_num_layers,
        "expected_sliding_window": expected_sliding_window,
        "config_num_hidden_layers": num_layers,
        "model_num_draft_layers": model_num_draft_layers,
        "sliding_window": sliding_window,
        "use_sliding_window": use_sliding_window,
        "max_window_layers": max_window_layers,
        "layer_sliding_windows": layer_windows,
        "layer_forward_counts": layer_forward_counts,
        "hidden_states_shape": hidden_states_shape,
        "draft_hidden_shape": draft_hidden_shape,
        "draft_logits_shape": draft_logits_shape,
        "draft_hidden_abs_mean": draft_hidden_abs_mean,
        "draft_logits_abs_mean": draft_logits_abs_mean,
        "sglang_runtime": runtime_summary,
    }
    json_dump(model_log_dir / "sliding_window_smoke_summary.json", summary)
    return summary


def validate_dflash_config(
    base_model_path: Path,
    draft_model_path: Path,
    expected_block_size: Optional[int],
) -> dict[str, Any]:
    base_config_path = base_model_path / "config.json"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing base model config: {base_config_path}")
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    draft_config = draft_model_config(draft_model_path)

    architectures = draft_config.get("architectures") or []
    if "DFlashDraftModel" not in architectures:
        raise RuntimeError(f"Expected DFlashDraftModel architecture, found {architectures}.")
    auto_map = draft_config.get("auto_map") or {}
    if auto_map.get("AutoModel") != "dflash.DFlashDraftModel":
        raise RuntimeError(f"Unexpected DFlash AutoModel mapping: {auto_map}.")

    block_size = int(draft_config.get("block_size", 0))
    if block_size <= 0:
        raise RuntimeError(f"Invalid DFlash block_size={block_size}.")
    if expected_block_size is not None and block_size != expected_block_size:
        raise RuntimeError(
            f"Expected DFlash block_size={expected_block_size}, found {block_size}."
        )

    dflash_config = draft_config.get("dflash_config")
    if not isinstance(dflash_config, dict):
        raise RuntimeError("Missing DFlash dflash_config in draft config.")
    target_layer_ids = dflash_config.get("target_layer_ids")
    if not isinstance(target_layer_ids, list) or not target_layer_ids:
        raise RuntimeError(f"Invalid DFlash target_layer_ids={target_layer_ids}.")
    if any(not isinstance(layer_id, int) for layer_id in target_layer_ids):
        raise RuntimeError(f"DFlash target_layer_ids must be integers: {target_layer_ids}.")

    base_num_layers = int(base_config.get("num_hidden_layers", 0))
    num_target_layers = int(draft_config.get("num_target_layers", 0))
    if base_num_layers <= 0:
        raise RuntimeError(f"Invalid base num_hidden_layers={base_num_layers}.")
    if num_target_layers != base_num_layers:
        raise RuntimeError(
            f"DFlash num_target_layers={num_target_layers} does not match "
            f"base num_hidden_layers={base_num_layers}."
        )
    if max(target_layer_ids) >= base_num_layers:
        raise RuntimeError(
            f"DFlash target_layer_ids exceed base layer count: {target_layer_ids}."
        )

    for key in ("model_type", "hidden_size", "vocab_size"):
        if draft_config.get(key) != base_config.get(key):
            raise RuntimeError(
                f"DFlash draft/base config mismatch for {key}: "
                f"draft={draft_config.get(key)!r}, base={base_config.get(key)!r}."
            )

    return {
        "base_model_path": str(base_model_path),
        "draft_model_path": str(draft_model_path),
        "base_model_type": base_config.get("model_type"),
        "base_num_hidden_layers": base_num_layers,
        "draft_architectures": architectures,
        "draft_auto_map": auto_map,
        "draft_num_hidden_layers": int(draft_config.get("num_hidden_layers", 0)),
        "draft_num_target_layers": num_target_layers,
        "block_size": block_size,
        "target_layer_ids": target_layer_ids,
        "mask_token_id": dflash_config.get("mask_token_id"),
    }


def run_dflash_smoke_test(
    model_key: str,
    gpu_ids: list[int],
    draft_model_path: Optional[Path] = None,
    base_model_path: Optional[Path] = None,
    expected_block_size: Optional[int] = 16,
    max_new_tokens: int = 8,
    attention_backend: str = "fa3",
    mem_fraction_static: float = 0.75,
) -> dict[str, Any]:
    if len(gpu_ids) != 1:
        raise ValueError("test-dflash currently expects exactly one GPU.")
    if max_new_tokens <= 0:
        raise ValueError("test-dflash requires --max-new-tokens > 0.")

    model_spec = model_specs([model_key])[0]
    base_path = Path(base_model_path) if base_model_path else model_spec.base_local_dir
    draft_path = Path(draft_model_path) if draft_model_path else model_spec.draft_local_dir
    config_summary = validate_dflash_config(base_path, draft_path, expected_block_size)
    block_size = int(config_summary["block_size"])

    os.environ["CUDA_VISIBLE_DEVICES"] = format_gpu_ids(gpu_ids)
    os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    prepend_pythonpath(ROOT)

    from transformers import AutoTokenizer
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if SpeculativeAlgorithm.from_string("DFLASH").name != "DFLASH":
        raise RuntimeError("The active SGLang install does not support DFLASH.")

    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    prompt = render_chat_prompt(
        tokenizer,
        [{"role": "user", "content": "Compute 12 + 30. Answer with only the number."}],
    )

    model_log_dir = LOGS_DIR / model_spec.key
    model_log_dir.mkdir(parents=True, exist_ok=True)
    engine = Engine(
        model_path=str(base_path),
        tokenizer_path=str(base_path),
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=str(draft_path),
        speculative_dflash_block_size=block_size,
        speculative_num_draft_tokens=block_size,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        dtype="bfloat16",
        attention_backend=attention_backend,
        trust_remote_code=True,
        mem_fraction_static=mem_fraction_static,
        context_length=EVAL_CONTEXT_LENGTH,
        tp_size=len(gpu_ids),
    )
    try:
        response = engine.generate(
            prompt=prompt,
            sampling_params={"temperature": 0.0, "max_new_tokens": max_new_tokens},
            rid="dflash-smoke",
        )
    finally:
        engine.shutdown()

    output_text = response.get("text", "")
    if not isinstance(output_text, str) or not output_text.strip():
        raise RuntimeError(f"DFlash smoke generated empty output: {response!r}")
    meta_info = normalize_sglang_meta_info(response.get("meta_info", {}))
    summary = {
        "backend": "sglang_dflash",
        "model": model_key,
        "display_name": model_spec.display_name,
        "gpu_ids": gpu_ids,
        "attention_backend": attention_backend,
        "mem_fraction_static": mem_fraction_static,
        "max_new_tokens": max_new_tokens,
        "speculative_config": {
            "speculative_algorithm": "DFLASH",
            "speculative_dflash_block_size": block_size,
            "speculative_num_draft_tokens": block_size,
            "speculative_num_steps": 1,
            "speculative_eagle_topk": 1,
        },
        "config": config_summary,
        "output_text": output_text,
        "meta_info": meta_info,
    }
    json_dump(model_log_dir / "dflash_smoke_summary.json", summary)
    return summary


def generate_vllm_single(
    llm: Any,
    prompt: str,
    sampling_params_dict: dict[str, Any],
) -> dict[str, Any]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(**sampling_params_dict)
    output = llm.generate([prompt], sampling_params=sampling_params)[0]
    return {
        "text": output.outputs[0].text,
        "token_ids": list(output.outputs[0].token_ids),
    }


def generate_angelslim_eagle3_single(
    model: Any,
    tokenizer: Any,
    prompt: str,
    sampling_params_dict: dict[str, Any],
    *,
    rid: str,
    turn_index: Optional[int] = None,
) -> dict[str, Any]:
    eagle3_module = importlib.import_module(
        "angelslim.compressor.speculative.inference.models.eagle3.eagle3_model"
    )
    stop_strings = sampling_params_dict.get("stop", [])
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].cuda()
    prompt_tokens = int(input_ids.shape[1])
    kv_cache_reserve = SPEC_NUM_DRAFT_TOKENS + 16
    max_new_tokens = min(
        int(sampling_params_dict["max_tokens"]),
        max(1, EVAL_CONTEXT_LENGTH - prompt_tokens - kv_cache_reserve),
    )
    trace_events: list[dict[str, Any]] = []
    step_index = 0
    original_update = eagle3_module.update_inference_inputs

    def traced_update_inference_inputs(
        input_ids: Any,
        inputs_embeds: Any,
        candidates: Any,
        best_candidate: Any,
        accept_length: Any,
        retrieve_indices: Any,
        logits_processor: Any,
        new_token: Any,
        past_key_values_data_list: Any,
        current_length_data: Any,
        model: Any,
        hidden_state_new: Any,
        sample_token: Any,
    ) -> Any:
        nonlocal step_index
        best_idx = int(best_candidate.item() if hasattr(best_candidate, "item") else best_candidate)
        accept_len = int(accept_length.item() if hasattr(accept_length, "item") else accept_length)
        prefix_ids = input_ids[0].tolist()
        draft_chunk = [int(token_id) for token_id in candidates[best_idx].tolist()]
        while draft_chunk and draft_chunk[-1] < 0:
            draft_chunk.pop()
        committed = [int(token_id) for token_id in candidates[best_idx, : accept_len + 1].tolist()]
        step_index += 1
        trace_events.append(
            build_trace_event(
                rid=rid,
                backend=BACKEND_ANGELSLIM_EAGLE3,
                tokenizer=tokenizer,
                prefix_token_ids=prefix_ids,
                draft_chunk_token_ids=draft_chunk,
                accept_len=accept_len,
                committed_token_ids=committed,
                step_index=step_index,
                turn_index=turn_index,
            )
        )
        return original_update(
            input_ids,
            inputs_embeds,
            candidates,
            best_candidate,
            accept_length,
            retrieve_indices,
            logits_processor,
            new_token,
            past_key_values_data_list,
            current_length_data,
            model,
            hidden_state_new,
            sample_token,
        )

    eagle3_module.update_inference_inputs = traced_update_inference_inputs
    try:
        output_ids, _, _, accept_length_list = model.eagle_generate(
            input_ids,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            max_new_tokens=max_new_tokens,
            max_length=EVAL_CONTEXT_LENGTH,
            log=True,
        )
    finally:
        eagle3_module.update_inference_inputs = original_update
    generated_ids = output_ids[0][input_ids.shape[1] :]
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
    )
    for stop_text in stop_strings:
        if stop_text and stop_text in text:
            text = text.split(stop_text, 1)[0]
            break
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                text = text.replace(special_tok, "")
        else:
            text = text.replace(special_token, "")
    accept_length_list = [int(x) for x in accept_length_list]
    num_drafts = len(accept_length_list)
    num_accepted_tokens = sum(accept_length_list)
    num_draft_tokens = num_drafts * SPEC_NUM_DRAFT_TOKENS
    trace_meta = build_trace_stats(trace_events)
    if not trace_events:
        trace_meta = {
            "spec_accept_length": mean_accept_length(accept_length_list),
            "spec_accept_rate": (
                (num_accepted_tokens / num_draft_tokens) if num_draft_tokens > 0 else None
            ),
            "spec_verify_ct": num_drafts,
            "accept_length_histogram": accept_length_histogram(accept_length_list),
        }
    return {
        "text": text,
        "token_ids": list(map(int, generated_ids.tolist())),
        "meta_info": {
            **trace_meta,
            "angelslim_num_drafts": num_drafts,
            "angelslim_num_draft_tokens": num_draft_tokens,
            "angelslim_num_accepted_tokens": num_accepted_tokens,
        },
        "accept_length_list": accept_length_list,
        "trace_events": trace_events,
    }


def build_angelslim_trace_record(rid: str, accept_length_list: list[int]) -> dict[str, Any]:
    histogram = Counter(int(length) for length in accept_length_list)
    num_drafts = len(accept_length_list)
    num_accepted_tokens = sum(accept_length_list)
    num_draft_tokens = num_drafts * SPEC_NUM_DRAFT_TOKENS
    return {
        "rid": rid,
        "backend": BACKEND_ANGELSLIM_EAGLE3,
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "mean_acceptance_length": (num_accepted_tokens / num_drafts) if num_drafts else 0.0,
        "acceptance_counts": None,
        "accept_length_histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "accepted_token_ids": None,
        "accepted_tokens": None,
        "rejected_candidate_token_ids": None,
        "rejected_candidate_tokens": None,
        "draft_chunk_token_ids": None,
        "draft_chunk_tokens": None,
    }


def run_vllm_single_sample(
    llm: Any,
    tokenizer: Any,
    model_spec: ModelSpec,
    dataset_name: str,
    sample: dict[str, Any],
    num_spec_tokens: int,
    trace_path: Path,
    trace_offset: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], int]:
    rid = f"{dataset_name}::{safe_name(str(sample['sample_id']))}"
    before_metrics = snapshot_vllm_metrics(llm)
    trace_events: list[dict[str, Any]] = []

    if dataset_name == "mtbench":
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        turns = []
        for turn_index, user_turn in enumerate(sample["turns"]):
            messages.append({"role": "user", "content": user_turn})
            prompt = render_chat_prompt(tokenizer, messages)
            sampling_params, prompt_tokens = adjusted_sampling_params_for_vllm(
                tokenizer,
                prompt,
                "mtbench",
                reserve_tokens=MTBENCH_TURN1_HISTORY_RESERVE if turn_index == 0 else 0,
            )
            start_offset = trace_offset
            output = generate_vllm_single(llm, prompt, sampling_params)
            sample_trace_events, trace_offset = read_jsonl_new_entries(trace_path, start_offset)
            for event in sample_trace_events:
                event["rid"] = rid
                event["turn_index"] = turn_index + 1
            trace_events.extend(sample_trace_events)
            turn_trace_meta = build_trace_stats(sample_trace_events)
            turns.append(
                {
                    "turn_index": turn_index + 1,
                    "text": output["text"],
                    "token_ids": output["token_ids"],
                    "meta_info": {
                        "prompt_tokens": prompt_tokens,
                        "max_new_tokens": sampling_params["max_tokens"],
                        **turn_trace_meta,
                    },
                }
            )
            messages.append({"role": "assistant", "content": output["text"]})
        after_metrics = snapshot_vllm_metrics(llm)
        delta = diff_vllm_metric_snapshots(before_metrics, after_metrics, num_spec_tokens)
        meta_info = build_vllm_meta_info(delta)
        if trace_events:
            meta_info.update(build_trace_stats(trace_events))
        for turn in turns:
            turn["meta_info"] = {**meta_info, **turn["meta_info"]}
        response = {
            "rid": rid,
            "text": turns[-1]["text"] if turns else "",
            "token_ids": turns[-1]["token_ids"] if turns else [],
            "meta_info": {**meta_info, **(turns[-1]["meta_info"] if turns else {})},
            "turns": turns,
        }
        return response, delta, trace_events, trace_offset

    messages = build_prompt_messages(dataset_name, sample)
    prompt = render_chat_prompt(tokenizer, messages)
    sampling_params, prompt_tokens = adjusted_sampling_params_for_vllm(
        tokenizer,
        prompt,
        dataset_name,
    )
    start_offset = trace_offset
    output = generate_vllm_single(llm, prompt, sampling_params)
    trace_events, trace_offset = read_jsonl_new_entries(trace_path, start_offset)
    for event in trace_events:
        event["rid"] = rid
    after_metrics = snapshot_vllm_metrics(llm)
    delta = diff_vllm_metric_snapshots(before_metrics, after_metrics, num_spec_tokens)
    meta_info = build_vllm_meta_info(delta)
    if trace_events:
        meta_info.update(build_trace_stats(trace_events))
    response = {
        "rid": rid,
        "text": output["text"],
        "token_ids": output["token_ids"],
        "meta_info": {
            **meta_info,
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": sampling_params["max_tokens"],
        },
    }
    return response, delta, trace_events, trace_offset


def evaluate_model_vllm(
    model_key: str,
    gpu_ids: list[int],
    sample_size: int,
    seed: int,
    cmmlu_repo: str,
    dataset_names: list[str],
) -> dict[str, Any]:
    del seed, cmmlu_repo
    os.environ["CUDA_VISIBLE_DEVICES"] = format_gpu_ids(gpu_ids)
    os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["EAGLE3_TRACE_CONTEXT_WINDOW"] = str(TRACE_CONTEXT_WINDOW)
    prepend_pythonpath(ROOT)

    from transformers import AutoTokenizer
    from vllm import LLM

    model_spec = model_specs([model_key])[0]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_spec.base_local_dir),
        trust_remote_code=True,
    )

    model_log_dir = LOGS_DIR / model_spec.key
    if model_log_dir.exists():
        shutil.rmtree(model_log_dir)
    model_log_dir.mkdir(parents=True, exist_ok=True)
    trace_path = model_log_dir / "spec_trace_raw.jsonl"
    os.environ["EAGLE3_TRACE_BACKEND"] = "vllm"
    os.environ["EAGLE3_TRACE_PATH"] = str(trace_path)
    os.environ["EAGLE3_TRACE_TOKENIZER_PATH"] = str(model_spec.base_local_dir)
    from eval import sitecustomize

    sitecustomize.install_vllm_trace_patch()

    llm = LLM(
        model=str(model_spec.base_local_dir),
        trust_remote_code=True,
        tensor_parallel_size=len(gpu_ids),
        enable_chunked_prefill=False,
        enforce_eager=False,
        gpu_memory_utilization=0.8,
        speculative_config={
            "method": "eagle3",
            "model": str(model_spec.draft_local_dir),
            "num_speculative_tokens": SPEC_NUM_DRAFT_TOKENS,
        },
        disable_log_stats=False,
        max_model_len=EVAL_CONTEXT_LENGTH,
        max_num_seqs=1,
        max_cudagraph_capture_size=24,
    )

    combined_path = model_log_dir / "combined_results.jsonl"
    summaries = []
    trace_offset = 0
    sample_paths = {name: SAMPLES_DIR / f"{name}.jsonl" for name in dataset_names}
    missing = [str(path) for path in sample_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared sample files. Run `python -m eval.prepare_data` first. Missing: "
            + ", ".join(missing)
        )

    try:
        for dataset_name in dataset_names:
            dataset_rows = read_jsonl(sample_paths[dataset_name])[:sample_size]
            result_rows: list[dict[str, Any]] = []
            trace_events: list[dict[str, Any]] = []
            dataset_dir = model_log_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            results_path = dataset_dir / "results.jsonl"
            if results_path.exists():
                results_path.unlink()

            for sample in dataset_rows:
                response, delta, sample_trace_events, trace_offset = run_vllm_single_sample(
                    llm=llm,
                    tokenizer=tokenizer,
                    model_spec=model_spec,
                    dataset_name=dataset_name,
                    sample=sample,
                    num_spec_tokens=SPEC_NUM_DRAFT_TOKENS,
                    trace_path=trace_path,
                    trace_offset=trace_offset,
                )
                if dataset_name == "mtbench":
                    score = {"metric_name": "generation_only", "score": None}
                else:
                    score = score_sample(dataset_name, sample, response)

                record = build_result_record(
                    model_spec=model_spec,
                    dataset_name=dataset_name,
                    sample=sample,
                    rid=response["rid"],
                    response=response,
                    score=score,
                )
                record["output_token_ids"] = response.get("token_ids", [])
                if dataset_name == "mtbench":
                    record["turns"] = response["turns"]
                append_jsonl(results_path, record)
                append_jsonl(combined_path, record)
                result_rows.append(record)
                if sample_trace_events:
                    trace_events.extend(sample_trace_events)
                else:
                    trace_events.append(build_vllm_trace_record(response["rid"], delta))

            trace_out = dataset_dir / "spec_trace.jsonl"
            with trace_out.open("w", encoding="utf-8") as f:
                for event in trace_events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")

            summary = summarise_dataset_results(
                model_key=model_spec.key,
                dataset_name=dataset_name,
                results=result_rows,
                trace_events=trace_events,
            )
            json_dump(dataset_dir / "summary.json", summary)
            summaries.append(summary)
    finally:
        del llm

    final_summary = {
        "model_key": model_spec.key,
        "model_name": model_spec.display_name,
        "backend": model_spec.backend,
        "gpu_ids": gpu_ids,
        "speculative_config": {
            "method": "eagle3",
            "num_speculative_tokens": SPEC_NUM_DRAFT_TOKENS,
            "context_length": EVAL_CONTEXT_LENGTH,
            "tensor_parallel_size": len(gpu_ids),
        },
        "datasets": summaries,
        "accept_length_histogram": merge_accept_length_histogram(summaries),
        "context_accept_length": merge_context_accept_length(summaries),
        "spec_trace_event_count": sum_spec_trace_event_count(summaries),
    }
    json_dump(model_log_dir / "model_summary.json", final_summary)
    return final_summary


def evaluate_model_angelslim_eagle3(
    model_key: str,
    gpu_ids: list[int],
    sample_size: int,
    seed: int,
    cmmlu_repo: str,
    dataset_names: list[str],
) -> dict[str, Any]:
    del seed, cmmlu_repo
    os.environ["CUDA_VISIBLE_DEVICES"] = format_gpu_ids(gpu_ids)
    os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT

    import torch

    Eagle3Model = load_angelslim_eagle3_model_class()

    model_spec = model_specs([model_key])[0]
    draft_dir = normalized_angelslim_eagle3_dir(model_spec.draft_local_dir)
    model = Eagle3Model.from_pretrained(
        str(model_spec.base_local_dir),
        str(draft_dir),
        total_token=SPEC_NUM_DRAFT_TOKENS,
        depth=SPEC_NUM_STEPS,
        top_k=SPEC_EAGLE_TOPK,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=True,
    )
    model.eval()
    tokenizer = model.tokenizer

    model_log_dir = LOGS_DIR / model_spec.key
    if model_log_dir.exists():
        shutil.rmtree(model_log_dir)
    model_log_dir.mkdir(parents=True, exist_ok=True)

    combined_path = model_log_dir / "combined_results.jsonl"
    summaries = []
    sample_paths = {name: SAMPLES_DIR / f"{name}.jsonl" for name in dataset_names}
    missing = [str(path) for path in sample_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared sample files. Run `python -m eval.prepare_data` first. Missing: "
            + ", ".join(missing)
        )

    for dataset_name in dataset_names:
        dataset_rows = read_jsonl(sample_paths[dataset_name])[:sample_size]
        result_rows: list[dict[str, Any]] = []
        trace_events: list[dict[str, Any]] = []
        dataset_dir = model_log_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        results_path = dataset_dir / "results.jsonl"
        if results_path.exists():
            results_path.unlink()

        for sample in dataset_rows:
            rid = f"{dataset_name}::{safe_name(str(sample['sample_id']))}"
            if dataset_name == "mtbench":
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                turns = []
                for turn_index, user_turn in enumerate(sample["turns"]):
                    messages.append({"role": "user", "content": user_turn})
                    prompt = render_chat_prompt(tokenizer, messages)
                    sampling_params, prompt_tokens = adjusted_sampling_params_for_vllm(
                        tokenizer,
                        prompt,
                        "mtbench",
                        reserve_tokens=MTBENCH_TURN1_HISTORY_RESERVE if turn_index == 0 else 0,
                    )
                    output = generate_angelslim_eagle3_single(
                        model,
                        tokenizer,
                        prompt,
                        sampling_params,
                        rid=rid,
                        turn_index=turn_index + 1,
                    )
                    turns.append(
                        {
                            "turn_index": turn_index + 1,
                            "text": output["text"],
                            "token_ids": output["token_ids"],
                            "meta_info": {
                                "prompt_tokens": prompt_tokens,
                                "max_new_tokens": sampling_params["max_tokens"],
                                **output["meta_info"],
                            },
                        }
                    )
                    trace_events.extend(output["trace_events"])
                    messages.append({"role": "assistant", "content": output["text"]})
                response = {
                    "rid": rid,
                    "text": turns[-1]["text"] if turns else "",
                    "token_ids": turns[-1]["token_ids"] if turns else [],
                    "meta_info": turns[-1]["meta_info"] if turns else {},
                    "turns": turns,
                }
                score = {"metric_name": "generation_only", "score": None}
            else:
                messages = build_prompt_messages(dataset_name, sample)
                prompt = render_chat_prompt(tokenizer, messages)
                sampling_params, prompt_tokens = adjusted_sampling_params_for_vllm(
                    tokenizer,
                    prompt,
                    dataset_name,
                )
                output = generate_angelslim_eagle3_single(
                    model,
                    tokenizer,
                    prompt,
                    sampling_params,
                    rid=rid,
                )
                response = {
                    "rid": rid,
                    "text": output["text"],
                    "token_ids": output["token_ids"],
                    "meta_info": {
                        "prompt_tokens": prompt_tokens,
                        "max_new_tokens": sampling_params["max_tokens"],
                        **output["meta_info"],
                    },
                }
                score = score_sample(dataset_name, sample, response)
                trace_events.extend(output["trace_events"])

            record = build_result_record(
                model_spec=model_spec,
                dataset_name=dataset_name,
                sample=sample,
                rid=response["rid"],
                response=response,
                score=score,
            )
            record["output_token_ids"] = response.get("token_ids", [])
            if dataset_name == "mtbench":
                record["turns"] = response["turns"]
            append_jsonl(results_path, record)
            append_jsonl(combined_path, record)
            result_rows.append(record)
        trace_out = dataset_dir / "spec_trace.jsonl"
        with trace_out.open("w", encoding="utf-8") as f:
            for event in trace_events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        summary = summarise_dataset_results(
            model_key=model_spec.key,
            dataset_name=dataset_name,
            results=result_rows,
            trace_events=trace_events,
        )
        json_dump(dataset_dir / "summary.json", summary)
        summaries.append(summary)

    final_summary = {
        "model_key": model_spec.key,
        "model_name": model_spec.display_name,
        "backend": model_spec.backend,
        "gpu_ids": gpu_ids,
        "speculative_config": {
            "method": "eagle3",
            "speculative_num_steps": SPEC_NUM_STEPS,
            "speculative_eagle_topk": SPEC_EAGLE_TOPK,
            "speculative_num_draft_tokens": SPEC_NUM_DRAFT_TOKENS,
            "context_length": EVAL_CONTEXT_LENGTH,
            "tensor_parallel_size": len(gpu_ids),
        },
        "datasets": summaries,
        "accept_length_histogram": merge_accept_length_histogram(summaries),
        "context_accept_length": merge_context_accept_length(summaries),
        "spec_trace_event_count": sum_spec_trace_event_count(summaries),
    }
    json_dump(model_log_dir / "model_summary.json", final_summary)
    return final_summary


def evaluate_model_current_process(
    model_key: str,
    gpu_ids: list[int],
    sample_size: int,
    seed: int,
    cmmlu_repo: str,
    dataset_names: list[str],
) -> dict[str, Any]:
    model_spec = model_specs([model_key])[0]
    if model_spec.backend == BACKEND_VLLM:
        return evaluate_model_vllm(model_key, gpu_ids, sample_size, seed, cmmlu_repo, dataset_names)
    if model_spec.backend == BACKEND_ANGELSLIM_EAGLE3:
        return evaluate_model_angelslim_eagle3(
            model_key,
            gpu_ids,
            sample_size,
            seed,
            cmmlu_repo,
            dataset_names,
        )
    if model_key == "qwen3_4b_dflash_b16_sglang":
        return evaluate_model_sglang_dflash(
            model_key,
            gpu_ids,
            sample_size,
            seed,
            cmmlu_repo,
            dataset_names,
        )
    return evaluate_model_sglang(model_key, gpu_ids, sample_size, seed, cmmlu_repo, dataset_names)


def orchestrate_run(
    model_keys: list[str],
    gpus: list[int],
    sample_size: int,
    seed: int,
    cmmlu_repo: Path,
    dataset_names: list[str],
) -> list[dict[str, Any]]:
    ensure_dirs()

    def run_model_subprocess(model_key: str, gpu_ids: list[int]) -> dict[str, Any]:
        model_spec = model_specs([model_key])[0]
        if not Path(model_spec.python_bin).exists():
            raise FileNotFoundError(
                f"Missing python for backend {model_spec.backend}: {model_spec.python_bin}"
            )
        worker_log = REPORTS_DIR / f"{model_spec.key}.worker.log"
        cmd = [
            model_spec.python_bin,
            str(ROOT / "run_eval.py"),
            "run-model",
            "--model",
            model_key,
            "--gpus",
            *[str(gpu_id) for gpu_id in gpu_ids],
            "--sample-size",
            str(sample_size),
            "--seed",
            str(seed),
            "--cmmlu-repo",
            str(cmmlu_repo),
            "--datasets",
            *dataset_names,
        ]
        env = os.environ.copy()
        env["HF_HOME"] = str(DEFAULT_HF_HOME)
        env["HF_ENDPOINT"] = HF_ENDPOINT
        env["EAGLE3_TRACE_CONTEXT_WINDOW"] = str(TRACE_CONTEXT_WINDOW)
        with worker_log.open("w", encoding="utf-8") as f:
            subprocess.run(cmd, check=True, env=env, stdout=f, stderr=subprocess.STDOUT)
        summary_path = LOGS_DIR / model_spec.key / "model_summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    for model_key in model_keys:
        results.append(run_model_subprocess(model_key, gpus))

    json_dump(REPORTS_DIR / "run_summary.json", results)
    write_markdown_summary(results, REPORTS_DIR / "run_summary.md", sample_size, gpus)
    return results


def write_markdown_summary(
    results: list[dict[str, Any]],
    path: Path,
    sample_size: int,
    gpus: list[int],
) -> None:
    lines = [
        "# EAGLE-3 Eval Summary",
        "",
        f"- Sample size per dataset: {sample_size}",
        f"- GPUs: {', '.join(map(str, gpus))}",
        f"- Speculative config: steps={SPEC_NUM_STEPS}, topk={SPEC_EAGLE_TOPK}, draft_tokens={SPEC_NUM_DRAFT_TOKENS}",
        "",
    ]
    for result in results:
        lines.append(f"## {result['model_name']}")
        lines.append("")
        lines.append(f"- GPUs: {', '.join(map(str, result.get('gpu_ids', [])))}")
        lines.append(f"- Backend: {result.get('backend', 'unknown')}")
        for dataset in result["datasets"]:
            score = dataset["mean_score"]
            score_text = "n/a" if score is None else f"{score:.4f}"
            hist = ", ".join(f"{k}:{v}" for k, v in dataset["accept_length_histogram"].items()) or "n/a"
            lines.append(
                f"- {dataset['dataset']}: score={score_text}, "
                f"request_accept_len={dataset.get('mean_request_accept_length', 'n/a')}, "
                f"accept_hist={hist}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EAGLE-3 benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    common.add_argument("--seed", type=int, default=DEFAULT_SEED)
    common.add_argument("--cmmlu-repo", type=Path, default=DEFAULT_CMMLU_REPO)
    common.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, choices=DATASET_NAMES)

    p_run = sub.add_parser("run", parents=[common])
    p_run.add_argument("--models", nargs="*", default=DEFAULT_MODEL_KEYS)
    p_run.add_argument("--gpus", nargs="*", type=int, default=DEFAULT_GPUS)

    p_run_model = sub.add_parser("run-model", parents=[common])
    p_run_model.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p_run_model.add_argument("--gpus", nargs="+", required=True, type=int)

    p_sliding_window = sub.add_parser("test-sliding-window")
    p_sliding_window.add_argument(
        "--model",
        default="qwen3_1p7b_k2_sw128_sglang",
        choices=list(MODEL_REGISTRY),
    )
    p_sliding_window.add_argument("--draft-model-path", type=Path)
    p_sliding_window.add_argument("--gpus", nargs="+", required=True, type=int)
    p_sliding_window.add_argument("--expected-num-layers", type=int, default=None)
    p_sliding_window.add_argument("--expected-sliding-window", type=int, default=None)
    p_sliding_window.add_argument("--seq-length", type=int, default=160)
    p_sliding_window.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_sliding_window.add_argument("--skip-runtime", action="store_true")

    p_dflash = sub.add_parser("test-dflash")
    p_dflash.add_argument(
        "--model",
        default="qwen3_4b_dflash_b16_sglang",
        choices=list(MODEL_REGISTRY),
    )
    p_dflash.add_argument("--draft-model-path", type=Path)
    p_dflash.add_argument("--base-model-path", type=Path)
    p_dflash.add_argument("--gpus", nargs="+", required=True, type=int)
    p_dflash.add_argument("--expected-block-size", type=int, default=16)
    p_dflash.add_argument("--max-new-tokens", type=int, default=8)
    p_dflash.add_argument("--attention-backend", default="fa3")
    p_dflash.add_argument("--mem-fraction-static", type=float, default=0.75)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if args.command == "run":
        orchestrate_run(
            args.models,
            args.gpus,
            args.sample_size,
            args.seed,
            args.cmmlu_repo,
            args.datasets,
        )
        return

    if args.command == "run-model":
        evaluate_model_current_process(
            args.model,
            args.gpus,
            args.sample_size,
            args.seed,
            str(args.cmmlu_repo),
            args.datasets,
        )
        return

    if args.command == "test-sliding-window":
        summary = run_sliding_window_smoke_test(
            args.model,
            args.gpus,
            args.seq_length,
            args.seed,
            args.draft_model_path,
            args.expected_num_layers,
            args.expected_sliding_window,
            not args.skip_runtime,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.command == "test-dflash":
        summary = run_dflash_smoke_test(
            model_key=args.model,
            gpu_ids=args.gpus,
            draft_model_path=args.draft_model_path,
            base_model_path=args.base_model_path,
            expected_block_size=args.expected_block_size,
            max_new_tokens=args.max_new_tokens,
            attention_backend=args.attention_backend,
            mem_fraction_static=args.mem_fraction_static,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
