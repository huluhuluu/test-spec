# 1. EAGLE-3 Benchmark Harness

This repository evaluates EAGLE-3 draft models across reasoning, coding, chat, and Chinese exam benchmarks. The main entry point is [`run_eval.py`](run_eval.py), which handles dataset sampling, backend dispatch, speculative-decoding trace collection, scoring, and report generation.

The harness requires model weights, one `SGLang` environment, one `vLLM` environment, one `AngelSlim` Eagle3 environment, and the `CMMLU` submodule.

## 1.1 Quick Start

### 1.1.1 Clone the Repository

```bash
# Clone this benchmark harness.
git clone --recursive https://github.com/huluhuluu/test-spec.git
cd test-spec

# Initialize the required CMMLU submodule.
git submodule update --init --recursive --depth 1 third_party/CMMLU
```

### 1.1.2 Environment

This benchmark uses separate Python environments because `SGLang`, local sliding-window SGLang, `vLLM`, and `AngelSlim` Eagle3 pin different high-impact inference dependencies such as `torch`, CUDA-related wheels, attention kernels, and tokenizer/runtime packages. **Note**: `PyTorch` version must be selected based on the CUDA version, [reference](https://pytorch.org/get-started/previous-versions/).

- **SGLang environment**

  ```bash
  # Create environment.
  conda create -y -n eagle3-sglang-bench python=3.11
  conda activate eagle3-sglang-bench

  export UV_DEFAULT_INDEX=https://mirrors.ustc.edu.cn/pypi/simple
  # Install SGLang plus the packages used by data preparation and scoring.
  uv pip install "sglang[all]"
  uv pip install transformers sympy antlr4-python3-runtime pyarrow
  ```

- **SGLang sliding-window environment**

  ```bash
  # Create the SGLang environment used by local sliding-window draft checkpoints.
  conda create -y -n eagle3-sglang-sw-bench python=3.11
  conda activate eagle3-sglang-sw-bench

  export UV_DEFAULT_INDEX=https://mirrors.ustc.edu.cn/pypi/simple
  uv pip install "sglang[all]"
  uv pip install transformers sympy antlr4-python3-runtime pyarrow datasets accelerate

  # Install the modified SpecForge branch used to produce and load the
  # sliding-window draft checkpoints.
  git submodule update --init third_party/SpecForge
  git -C third_party/SpecForge checkout feat/sliding-window
  pip install -e third_party/SpecForge
  ```

- **vLLM environment**
  ```bash
  # Create and activate the isolated vLLM benchmark environment.
  conda create -y -n eagle3-vllm-bench python=3.11
  conda activate eagle3-vllm-bench

  # Install vLLM plus the packages used by data preparation, scoring, and Hunyuan Eagle3 loading.
  export UV_DEFAULT_INDEX=https://mirrors.ustc.edu.cn/pypi/simple
  uv pip install "vllm==0.11.2"
  uv pip install transformers sympy antlr4-python3-runtime pyarrow datasets accelerate threadpoolctl
  ```

- **AngelSlim Eagle3 environment**
  ```bash
  # Create and activate the isolated AngelSlim Eagle3 benchmark environment.
  conda create -y -n eagle3-angelslim-bench python=3.11
  conda activate eagle3-angelslim-bench

  # Install the published AngelSlim package plus the packages used by data preparation and scoring.
  export UV_DEFAULT_INDEX=https://mirrors.ustc.edu.cn/pypi/simple
  uv pip install \
    "angelslim==0.3.0" \
    "transformers==4.57.1" huggingface_hub sympy antlr4-python3-runtime pyarrow datasets accelerate threadpoolctl shortuuid safetensors torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
  ```

### 1.1.3 Models and Data

All model and Hugging Face dataset downloads go through `hfd.sh` scripts (`CMMLU data` is the exception). Model and dataset download paths are configured in `eval/config.json` under `downloads.model_path` and `downloads.data_path`.

```bash
# download `hfd.sh` if needed
# wget https://hf-mirror.com/hfd/hfd.sh; chmod a+x hfd.sh

# Download all configured models.
bash scripts/download_models.sh

# Download Hugging Face datasets with hfd.sh and initialize the CMMLU submodule.
bash scripts/download_datasets.sh

# Build sampled JSONL inputs under artifacts/samples/.
python -m eval.prepare_data --sample-size 80
```

### 1.1.4 Evaluation

Evaluation defaults and backend Python settings are configured in `eval/config.json` under `run_eval`.

```bash
# Run all configured models and datasets.
# Models are executed sequentially in the given order, and each model uses all listed GPUs.
python run_eval.py run --sample-size 80 --gpus 0 1 2 3
```

`run` picks backend Python from environment variables and `run_eval.python_bins` / `run_eval.conda_envs` in `eval/config.json`, then launches one subprocess per model. `run-model` uses the current Python interpreter directly, so activate the matching backend environment first.

```bash
# Run one local sliding-window model in its own environment.
conda activate eagle3-sglang-sw-bench
python run_eval.py run-model \
  --model qwen3_1p7b_sw64_sglang \
  --gpus 4 5 \
  --datasets gsm8k
```

```bash
# Run one Qwen EAGLE3 model through the AngelSlim backend.
conda activate eagle3-angelslim-bench
python run_eval.py run-model \
  --model qwen3_1p7b_eagle3-angelslim \
  --gpus 5 \
  --datasets gsm8k
```


## 1.2 Repository Layout

```text
.
├── README.md                    # project docs
├── run_eval.py                  # eval CLI, backend dispatch, scoring, reports
├── eval/
│   ├── __init__.py
│   ├── config.json              # downloads and run_eval defaults
│   ├── config_loader.py         # config reader
│   ├── sitecustomize.py         # SGLang/vLLM trace patches
│   ├── prompt_templates.json    # dataset prompt templates
│   ├── prompt_templates.py      # prompt renderer
│   └── prepare_data.py          # local dataset sampler
├── scripts/
│   ├── download_datasets.sh     # dataset downloader
│   ├── download_models.sh       # model downloader
│   ├── plot_context_accept_length.py # context/accept plot
│   └── viz_trace.py             # trace tree renderer
├── third_party/
│   └── CMMLU/                   # CMMLU submodule
└── artifacts/
    ├── samples/                 # sampled JSONL inputs
    ├── logs/                    # per-model results
    └── reports/                 # run summaries
```

## 1.3 Models and Backends

| Model key | Backend | Base model | Draft model |
| --- | --- | --- | --- |
| `qwen3_1p7b_eagle3-angelslim` | AngelSlim | `Qwen3-1.7B` | `AngelSlim/Qwen3-1.7B_eagle3` |
| `qwen3_4b_eagle3-angelslim` | AngelSlim | `Qwen3-4B` | `AngelSlim/Qwen3-4B_eagle3` |
| `taobao_qwen3_4b_eagle3` | SGLang | `Qwen3-4B-Instruct-2507` | `taobao-mnn/...Eagle3` |
| `zjcxy_qwen3_4b_eagle3_zh` | SGLang | `Qwen3-4B-Instruct-2507` | `Zjcxy-SmartAI/...zh` |
| `hunyuan_1p8b_eagle3` | AngelSlim | `Hunyuan-1.8B-Instruct` | `AngelSlim/...1.8B...eagle3` |
| `hunyuan_4b_eagle3` | AngelSlim | `Hunyuan-4B-Instruct` | `AngelSlim/...4B...eagle3` |
| `qwen3_1p7b_sw64_sglang` | SGLang SW | local `Qwen3-1.7B` | local SW64 checkpoint |
| `qwen3_1p7b_sw256_sglang` | SGLang SW | local `Qwen3-1.7B` | local SW256 checkpoint |

## 1.4 Datasets

| Dataset | Source | Prompt | Sample file |
| --- | --- | --- | --- |
| `gsm8k` | `openai/gsm8k` | zero-shot CoT | `artifacts/samples/gsm8k.jsonl` |
| `math500` | `HuggingFaceH4/MATH-500` | zero-shot math CoT | `artifacts/samples/math500.jsonl` |
| `mtbench` | `HuggingFaceH4/mt_bench_prompts` | MT-Bench turn prompt | `artifacts/samples/mtbench.jsonl` |
| `humaneval` | `openai/openai_humaneval` | code completion | `artifacts/samples/humaneval.jsonl` |
| `ceval` | `ceval/ceval-exam` | answer-only MCQ | `artifacts/samples/ceval.jsonl` |
| `cmmlu` | `third_party/CMMLU` | answer-only MCQ | `artifacts/samples/cmmlu.jsonl` |

Prompt templates are stored in [`eval/prompt_templates.json`](eval/prompt_templates.json). They are official or benchmark-standard where a canonical template exists; otherwise they are explicit community/common templates chosen for reproducible zero-shot evaluation.

## 1.5 Evaluation

Use `python run_eval.py` to evaluate selected models and datasets. Before starting an evaluation run, make sure model weights, datasets, and sampled JSONL files are already prepared.

### 1.5.1 Prerequisites

Run the following steps once before evaluation:

```bash
# Download model weights.
bash scripts/download_models.sh

# Download datasets and initialize the CMMLU submodule.
bash scripts/download_datasets.sh

# Build sampled evaluation inputs.
python -m eval.prepare_data --sample-size 80
```

Note: `run` launches one subprocess per model in `--models` order, and each model uses the full `--gpus` list.

### 1.5.2 Configuration

Download paths, run defaults, and local model overrides are configured in [`eval/config.json`](eval/config.json):

```jsonc
{
  "downloads": { // download paths
    "model_path": "/data/HUGGINGFACE/models", // model weights
    "data_path": "/data/HUGGINGFACE/datasets", // datasets
    "hfd_path": "/data/HUGGINGFACE/hfd.sh", // hfd script
    "hf_endpoint": "https://hf-mirror.com" // HF endpoint
  },
  "run_eval": { // run_eval defaults
    "artifacts_dir": "artifacts", // outputs
    "cmmlu_repo": "third_party/CMMLU", // CMMLU repo
    "default_sample_size": 80, // sample size
    "default_seed": 20260429, // seed
    "default_gpus": [0, 1, 2, 3], // GPUs
    "default_datasets": ["gsm8k", "math500", "mtbench", "humaneval", "ceval", "cmmlu"], // datasets
    "default_models": [
      "qwen3_1p7b_eagle3-angelslim",
      "qwen3_4b_eagle3-angelslim",
      "taobao_qwen3_4b_eagle3",
      "zjcxy_qwen3_4b_eagle3_zh",
      "hunyuan_1p8b_eagle3",
      "hunyuan_4b_eagle3"
    ],
    "trace_context_window": 16, // trace window
    "context_length": 1024, // context limit
    "default_max_new_tokens": 2048, // default output budget
    "long_max_new_tokens": 2048, // long-form budget
    "prompt_token_safety_margin": 8, // prompt margin
    "sglang_token_safety_margin": 64, // SGLang margin
    "mtbench_turn1_history_reserve": 256, // MT-Bench reserve
    "sglang_input_token_fudge": 32, // SGLang input fudge
    "speculative": {
      "num_steps": 7, // steps
      "eagle_topk": 10, // top-k
      "num_draft_tokens": 32 // draft tokens
    },
    "conda_envs": {
      "sglang": "eagle3-sglang-bench", // SGLang env
      "sglang_sliding_window": "eagle3-sglang-sw-bench", // sliding-window env
      "vllm": "eagle3-vllm-bench", // vLLM env
      "angelslim_eagle3": "eagle3-angelslim-bench" // AngelSlim env
    },
    "python_bins": {
      "sglang": null, // optional Python
      "sglang_sliding_window": null, // optional Python
      "vllm": null, // optional Python
      "angelslim_eagle3": null // optional Python
    },
    "model_overrides": {
      "qwen3_1p7b_sw64_sglang": {
        "draft_model_path": "/workspace/code/test-spec/SpecForge/outputs/qwen3-1.7b-eagle3-sharegpt-sw64/epoch_9_step_233900", // draft path
        "base_model_path": "/data/HUGGINGFACE/Qwen3-1.7B" // base path
      },
      "qwen3_1p7b_sw256_sglang": {
        "draft_model_path": "/workspace/code/test-spec/SpecForge/outputs/qwen3-1.7b-eagle3-sharegpt-sw256/epoch_9_step_233900", // draft path
        "base_model_path": "/data/HUGGINGFACE/Qwen3-1.7B" // base path
      }
    }
  }
}
```

To test a local checkpoint with an existing model key, only override its paths:

```jsonc
{
  "run_eval": {
    "model_overrides": {
      "qwen3_1p7b_sw64_sglang": {
        "draft_model_path": "/path/to/draft_checkpoint",
        "base_model_path": "/path/to/base_model"
      }
    }
  }
}
```

To add a new test model key, add the key to `MODEL_REGISTRY` in [`run_eval.py`](run_eval.py):

```python
"my_local_eagle3_sglang": {
    "display_name": "local/my-eagle3",
    "draft_repo_id": "local/my-eagle3-draft",
    "base_repo_id": "local/my-base-model",
    "backend": BACKEND_SGLANG,
}
```

Then add its local paths in `eval/config.json` under `run_eval.model_overrides`:

```jsonc
{
  "run_eval": {
    "model_overrides": {
      "my_local_eagle3_sglang": {
        "draft_model_path": "/path/to/draft_checkpoint",
        "base_model_path": "/path/to/base_model"
      }
    }
  }
}
```

Add the key to `eval/config.json` under `run_eval.default_models` if it should run by default, or pass it directly:

```bash
python run_eval.py run-model \
  --model my_local_eagle3_sglang \
  --gpus 0 \
  --datasets gsm8k
```

For sliding-window draft checkpoints, set `"requires_sliding_window_specforge": true` on the `MODEL_REGISTRY` entry and run from an environment with the modified `third_party/SpecForge` installed.

### 1.5.3 Common Arguments

| Argument | Description |
| --- | --- |
| `--sample-size` | Number of sampled examples per dataset. Defaults to `run_eval.default_sample_size` in `eval/config.json` and should match `python -m eval.prepare_data --sample-size ...`. |
| `--gpus` | Visible GPUs for the current model run. Defaults to `run_eval.default_gpus` in `eval/config.json`; every model uses the full list. |
| `--models` | One or more model keys from Section `1.3`. Defaults to `run_eval.default_models` in `eval/config.json`. |
| `--datasets` | One or more dataset names from Section `1.4`. Defaults to `run_eval.default_datasets` in `eval/config.json`. |
| `--seed` | Random seed used when sampling evaluation inputs. Defaults to `run_eval.default_seed` in `eval/config.json`. |
| `--cmmlu-repo` | Optional override for the local `CMMLU` repository path. Defaults to `run_eval.cmmlu_repo` in `eval/config.json`. |

## 1.6 Evaluation Settings

The defaults below live in [`eval/config.json`](eval/config.json) under the `run_eval` section. Edit that file to change repository defaults; pass CLI arguments to override them for one command.

| Config key in `eval/config.json` | Value |
| --- | --- |
| `run_eval.speculative.num_steps` | `7` |
| `run_eval.speculative.eagle_topk` | `10` |
| `run_eval.speculative.num_draft_tokens` | `32` |
| `run_eval.context_length` | `1024` |
| Decoding | Greedy decoding: `temperature=0.0`, `top_p=1.0` |
| `run_eval.default_max_new_tokens` | `2048` |
| `run_eval.long_max_new_tokens` for `mtbench` / `humaneval` | `2048` |
| Effective output budget | `max_new_tokens` does not include input length, but generation must still fit the context window. In this repo the runtime clamps it to the remaining budget after prompt tokens, reserve tokens, and safety margins. |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | `1` |

## 1.7 Outputs

All generated files are written under `artifacts/`. These files are runtime outputs and are not required before running the benchmark.

| Path | Description |
| --- | --- |
| `artifacts/samples/*.jsonl` | Sampled benchmark inputs |
| `artifacts/logs/<model>/combined_results.jsonl` | Combined per-model results |
| `artifacts/logs/<model>/<dataset>/results.jsonl` | Per-model, per-dataset results |
| `artifacts/logs/<model>/spec_trace_raw.jsonl` | Raw SGLang speculative trace |
| `artifacts/logs/<model>/<dataset>/spec_trace.jsonl` | Per-dataset trace |
| `artifacts/logs/<model>/<dataset>/summary.json` | Per-dataset summary |
| `artifacts/reports/run_summary.json` | Full run summary |
| `artifacts/reports/run_summary.md` | Markdown run summary |

Common result fields include `score`, `spec_accept_length`, `spec_accept_rate`, `spec_verify_ct`, `accepted_tokens`, and `rejected_candidate_tokens`.
