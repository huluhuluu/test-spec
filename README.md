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

This benchmark uses three Python environments because `SGLang`, `vLLM`, and `AngelSlim` Eagle3 pin different high-impact inference dependencies such as `torch`, CUDA-related wheels, attention kernels, and tokenizer/runtime packages. 

- **SGLang environment**

  ```bash
  # Create environment.
  conda create -y -n eagle3-sglang-bench python=3.11
  conda activate eagle3-sglang-bench

  export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
  # Install SGLang plus the packages used by data preparation and scoring.
  uv pip install "sglang[all]"
  uv pip install transformers sympy antlr4-python3-runtime pyarrow
  ```

- **vLLM environment**
  ```bash
  # Create and activate the isolated vLLM benchmark environment.
  conda create -y -n eagle3-vllm-bench python=3.11
  conda activate eagle3-vllm-bench

  # Install vLLM plus the packages used by data preparation, scoring, and Hunyuan Eagle3 loading.
  export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
  uv pip install "vllm==0.11.2"
  uv pip install transformers sympy antlr4-python3-runtime pyarrow datasets accelerate threadpoolctl
  ```

- **AngelSlim Eagle3 environment**
  ```bash
  # Create and activate the isolated AngelSlim Eagle3 benchmark environment.
  conda create -y -n eagle3-angelslim-bench python=3.11
  conda activate eagle3-angelslim-bench

  # Install the published AngelSlim package plus the packages used by data preparation and scoring.
  export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
  uv pip install \
    "angelslim==0.3.0" \
    "transformers==4.57.1" \
    "huggingface_hub<1" \
    sympy antlr4-python3-runtime pyarrow datasets accelerate threadpoolctl shortuuid safetensors
  ```

### 1.1.3 Configure Paths

Model and dataset paths are configured in [`eval/config.json`](eval/config.json). Before downloading or running anything, edit these values:

```json
{
  "model_path": "/data/HUGGINGFACE/models",
  "data_path": "/data/HUGGINGFACE/datasets",
  "hfd_path": "/data/HUGGINGFACE/hfd.sh",
  "hf_endpoint": "https://hf-mirror.com"
}
```

### 1.1.4 Models and Data

All model and Hugging Face dataset downloads go through `hfd.sh` scripts(`CMMLU data` is the exception). 

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

### 1.1.5 Evaluation

```bash
# Run all configured models and datasets.
# Models are executed sequentially in the given order, and each model uses all listed GPUs.
python run_eval.py run --sample-size 80 --gpus 0 1 2 3
```


## 1.2 Repository Layout

```text
.
├── README.md
├── eval/
│   ├── __init__.py
│   ├── config.json
│   ├── config_loader.py
│   ├── prompt_templates.json
│   ├── prompt_templates.py
│   └── prepare_data.py
├── run_eval.py
├── scripts/
│   ├── download_datasets.sh
│   ├── download_models.sh
├── third_party/
│   └── CMMLU/
└── artifacts/
    ├── samples/
    ├── logs/
    └── reports/
```

| Path | Purpose |
| --- | --- |
| [`eval/config.json`](eval/config.json) | User-editable paths for model weights, datasets, `hfd.sh`, and Hugging Face endpoint |
| [`eval/config_loader.py`](eval/config_loader.py) | Internal JSON config reader used by Python code and shell scripts |
| [`eval/prompt_templates.json`](eval/prompt_templates.json) | Dataset prompt templates used by `run_eval.py` |
| [`eval/prompt_templates.py`](eval/prompt_templates.py) | Prompt template loader and renderer |
| [`eval/prepare_data.py`](eval/prepare_data.py) | Reads local datasets and writes sampled `artifacts/samples/*.jsonl`; it does not download data |
| [`run_eval.py`](run_eval.py) | Evaluation scheduler, backend dispatch, scoring, and reporting |
| [`scripts/download_models.sh`](scripts/download_models.sh) | Model download entry point using `hfd.sh` and paths from [`eval/config.json`](eval/config.json) |
| [`scripts/download_datasets.sh`](scripts/download_datasets.sh) | Dataset download entry point using `hfd.sh --dataset` and paths from [`eval/config.json`](eval/config.json) |
| `third_party/CMMLU/` | CMMLU submodule used by the `cmmlu` dataset loader |
| `artifacts/samples/` | Sampled benchmark inputs |
| `artifacts/logs/` | Per-model and per-dataset result logs |
| `artifacts/reports/` | Run-level summaries |

## 1.3 Models and Backends

| Model key | Backend | Base model | Draft model |
| --- | --- | --- | --- |
| `qwen3_1p7b_eagle3` | [`vLLM`](https://github.com/vllm-project/vllm) | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) | [`AngelSlim/Qwen3-1.7B_eagle3`](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) |
| `qwen3_4b_eagle3` | [`vLLM`](https://github.com/vllm-project/vllm) | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | [`AngelSlim/Qwen3-4B_eagle3`](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3) |
| `taobao_qwen3_4b_eagle3` | [`SGLang`](https://github.com/sgl-project/sglang) | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | [`taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3`](https://huggingface.co/taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3) |
| `zjcxy_qwen3_4b_eagle3_zh` | [`SGLang`](https://github.com/sgl-project/sglang) | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | [`Zjcxy-SmartAI/Eagle3-Qwen3-4B-Instruct-2507-zh`](https://huggingface.co/Zjcxy-SmartAI/Eagle3-Qwen3-4B-Instruct-2507-zh) |
| `hunyuan_1p8b_eagle3` | [`AngelSlim`](https://github.com/tencent/AngelSlim) | [`tencent/Hunyuan-1.8B-Instruct`](https://huggingface.co/tencent/Hunyuan-1.8B-Instruct) | [`AngelSlim/Hunyuan-1.8B-Instruct_eagle3`](https://huggingface.co/AngelSlim/Hunyuan-1.8B-Instruct_eagle3) |
| `hunyuan_4b_eagle3` | [`AngelSlim`](https://github.com/tencent/AngelSlim) | [`tencent/Hunyuan-4B-Instruct`](https://huggingface.co/tencent/Hunyuan-4B-Instruct) | [`AngelSlim/Hunyuan-4B-Instruct_eagle3`](https://huggingface.co/AngelSlim/Hunyuan-4B-Instruct_eagle3) |
| `qwen3_1p7b_sw64_sglang` | [`SGLang`](https://github.com/sgl-project/sglang) + local sliding-window draft | Local `Qwen/Qwen3-1.7B` path | Local sliding-window checkpoint path |
| `qwen3_1p7b_sw256_sglang` | [`SGLang`](https://github.com/sgl-project/sglang) + local sliding-window draft | Local `Qwen/Qwen3-1.7B` path | Local sliding-window checkpoint path |



## 1.4 Datasets

| Dataset | Source | Prompt template | Preparation | Local output |
| --- | --- | --- | --- | --- |
| `gsm8k` | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | Community zero-shot CoT template, adapted from [`lm-evaluation-harness` GSM8K CoT task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml). Not an official dataset prompt. | `scripts/download_datasets.sh` | `artifacts/samples/gsm8k.jsonl` |
| `math500` | [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Common zero-shot math CoT template. The dataset card does not define a canonical official prompt, so this repo uses a simple community-style prompt. | `scripts/download_datasets.sh` | `artifacts/samples/math500.jsonl` |
| `mtbench` | [`HuggingFaceH4/mt_bench_prompts`](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts) | Standard MT-Bench single-turn prompt flow based on [`FastChat / llm_judge`](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge). | `scripts/download_datasets.sh` | `artifacts/samples/mtbench.jsonl` |
| `humaneval` | [`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) | Official HumanEval completion format: generate the `completion` for the provided problem prompt, following [`openai/human-eval`](https://github.com/openai/human-eval). | `scripts/download_datasets.sh` | `artifacts/samples/humaneval.jsonl` |
| `ceval` | [`ceval/ceval-exam`](https://huggingface.co/datasets/ceval/ceval-exam) | Adapted from the official C-Eval answer-only prompt in the [`ceval` repository](https://github.com/hkust-nlp/ceval). This repo uses the zero-shot single-question form instead of the original k-shot template. | `scripts/download_datasets.sh` | `artifacts/samples/ceval.jsonl` |
| `cmmlu` | [`haonan-li/CMMLU`](https://github.com/haonan-li/CMMLU) | Adapted from the CMMLU direct-answer prompt in [`third_party/CMMLU/src/mp_utils.py`](third_party/CMMLU/src/mp_utils.py). This repo uses a simplified zero-shot form instead of the original few-shot builder. | `git submodule` via `scripts/download_datasets.sh` | `artifacts/samples/cmmlu.jsonl` |

Prompt templates are stored in [`eval/prompt_templates.json`](eval/prompt_templates.json). They are official or benchmark-standard where a canonical template exists; otherwise they are explicit community/common templates chosen for reproducible zero-shot evaluation.

## 1.5 Evaluation

Use `run_eval.py` to evaluate selected models and datasets. Before starting an evaluation run, make sure model weights, datasets, and sampled JSONL files are already prepared.

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

### 1.5.2 Activate the Correct Environment

Activate the environment that matches the backend of the models you want to test.

- `SGLang` models:
  - `taobao_qwen3_4b_eagle3`
  - `zjcxy_qwen3_4b_eagle3_zh`

```bash
# Activate the SGLang environment before running SGLang-backed models.
conda activate eagle3-sglang-bench
```

- `Local sliding-window SGLang` models:
  - `qwen3_1p7b_sw64_sglang`
  - `qwen3_1p7b_sw256_sglang`

```bash
# Activate the environment that contains both `sglang` and the modified
# sliding-window `specforge` package.
conda activate test-spec
```

The local sliding-window checkpoints are loaded with `trust_remote_code=True`, so the
runtime environment must be able to import both:

- `sglang`
- `specforge`

Initialize the modified `SpecForge` submodule before installing it into the runtime
environment:

```bash
git submodule update --init third_party/SpecForge
```

In the current setup, `specforge` is installed from the vendored submodule path:

```bash
pip install -e third_party/SpecForge
```

The submodule is expected to track the sliding-window branch:

```bash
git -C third_party/SpecForge branch --show-current
# expected: feat/sliding-window
```

- `vLLM` models:
  - `qwen3_1p7b_eagle3`
  - `qwen3_4b_eagle3`

```bash
# Activate the vLLM environment before running vLLM-backed models.
conda activate eagle3-vllm-bench
```

- `AngelSlim Eagle3` models:
  - `hunyuan_1p8b_eagle3`
  - `hunyuan_4b_eagle3`

```bash
# Activate the AngelSlim Eagle3 environment before running AngelSlim-backed models.
conda activate eagle3-angelslim-bench
```

Run different model in separate commands under their corresponding environments.

### 1.5.3 Run SGLang Models

```bash
# Run two SGLang models sequentially on GPUs 2,3,4,5.
conda activate eagle3-sglang-bench
python run_eval.py run \
  --sample-size 80 \
  --gpus 2 3 4 5 \
  --models taobao_qwen3_4b_eagle3 zjcxy_qwen3_4b_eagle3_zh \
  --datasets gsm8k math500 ceval cmmlu
```

### 1.5.4 Run vLLM Models

```bash
# Run Qwen vLLM models sequentially on GPUs 0,1.
conda activate eagle3-vllm-bench
python run_eval.py run \
  --sample-size 80 \
  --gpus 0 1 \
  --models qwen3_1p7b_eagle3 qwen3_4b_eagle3 \
  --datasets gsm8k math500 humaneval mtbench
```

### 1.5.5 Run AngelSlim Eagle3 Models

```bash
# Run the Hunyuan 4B Eagle3 model on GPUs 0,1 through the AngelSlim Eagle3 backend.
conda activate eagle3-angelslim-bench
python run_eval.py run \
  --sample-size 80 \
  --gpus 0 1 \
  --models hunyuan_1p8b_eagle3 hunyuan_4b_eagle3 \
  --datasets gsm8k math500 humaneval mtbench
```

### 1.5.6 Run Local Sliding-Window Draft Models Through SGLang

```bash
# Run the local sliding-window checkpoints on GPUs 4,5.
conda activate test-spec
python run_eval.py run-model \
  --model qwen3_1p7b_sw64_sglang \
  --gpus 4 5 \
  --datasets gsm8k
```

The harness now uses the SGLang EAGLE3 path for local sliding-window checkpoints.
This repository no longer depends on the `third_party/SpecForge` submodule, but the
runtime environment must still provide the modified `specforge` package so
`LlamaForCausalLMEagle3` can be resolved when loading local draft checkpoints.

### 1.5.7 Sequential Scheduling Behavior

`run` mode executes models sequentially in the order given by `--models`.

- If you pass `--models a b c`, the harness runs `a`, then `b`, then `c`.
- Each model uses the full GPU list from `--gpus`.
- The next model starts only after the previous model finishes.

For example, this command:

```bash
python run_eval.py run \
  --sample-size 80 \
  --gpus 2 3 4 5 \
  --models taobao_qwen3_4b_eagle3 zjcxy_qwen3_4b_eagle3_zh
```

means:

1. Run `taobao_qwen3_4b_eagle3` on GPUs `2,3,4,5`.
2. After it finishes, run `zjcxy_qwen3_4b_eagle3_zh` on GPUs `2,3,4,5`.

### 1.5.8 Common Arguments

| Argument | Description |
| --- | --- |
| `--sample-size` | Number of sampled examples per dataset. This should match the size used by `python -m eval.prepare_data --sample-size ...`. |
| `--gpus` | Visible GPUs for the current model run. Every model in the command uses the full list. |
| `--models` | One or more model keys from the table in Section `1.3`. |
| `--datasets` | One or more dataset names from Section `1.4`. |
| `--seed` | Random seed used when sampling evaluation inputs. |
| `--cmmlu-repo` | Optional override for the local `CMMLU` repository path. |

### 1.5.9 Outputs

After a run finishes, check:

- [`artifacts/logs/`](artifacts/logs/) for per-model and per-dataset raw results
- [`artifacts/reports/run_summary.json`](artifacts/reports/run_summary.json) for the machine-readable run summary
- [`artifacts/reports/run_summary.md`](artifacts/reports/run_summary.md) for the Markdown summary

## 1.6 Evaluation Settings

| Setting | Value |
| --- | --- |
| `speculative_num_steps` | `7` |
| `speculative_eagle_topk` | `10` |
| `speculative_num_draft_tokens` | `32` |
| `context_length` | `1024` |
| Decoding | Greedy decoding: `temperature=0.0`, `top_p=1.0` |
| Default `max_new_tokens` | `2048` |
| `mtbench` / `humaneval` `max_new_tokens` | `2048` |
| `ceval` / `cmmlu` `max_new_tokens` | `2048` |
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
