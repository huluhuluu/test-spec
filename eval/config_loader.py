from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


CONFIG = load_config()
DOWNLOAD_CONFIG = CONFIG.get("downloads", CONFIG)
MODEL_PATH = Path(DOWNLOAD_CONFIG["model_path"]).expanduser()
DATA_PATH = Path(DOWNLOAD_CONFIG["data_path"]).expanduser()
HFD_PATH = Path(DOWNLOAD_CONFIG["hfd_path"]).expanduser()
HF_ENDPOINT = DOWNLOAD_CONFIG.get("hf_endpoint", "https://hf-mirror.com")
RUN_EVAL_CONFIG = CONFIG.get("run_eval", {})


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


HF_DATASET_REPOS = {
    "gsm8k": "openai/gsm8k",
    "math500": "HuggingFaceH4/MATH-500",
    "mtbench": "HuggingFaceH4/mt_bench_prompts",
    "humaneval": "openai/openai_humaneval",
    "ceval": "ceval/ceval-exam",
}


def repo_leaf(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def dataset_local_dir(dataset_name: str) -> Path:
    return DATA_PATH / repo_leaf(HF_DATASET_REPOS[dataset_name])
