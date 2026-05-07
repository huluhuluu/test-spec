# Repository Guidelines

## Project Structure & Module Organization
`run_eval.py` is the main CLI entry for benchmark orchestration, scoring, and report generation. Core helpers live in `eval/`: `config.json` stores local paths, `config_loader.py` reads them, `prepare_data.py` builds sampled JSONL inputs, and `prompt_templates.*` defines dataset prompts. Utility scripts live in `scripts/`, generated outputs go under `artifacts/`, and vendored dependencies stay in `third_party/` (`CMMLU` submodule and `SpecForge` reference code).

## Build, Test, and Development Commands
Prefix shell commands with `rtk` in this repo.

```bash
rtk python -m eval.prepare_data --sample-size 80
rtk python -m eval.prepare_data --sample-size 20 --smoke
rtk python run_eval.py run --sample-size 80 --gpus 0 1
rtk python run_eval.py run-model --model qwen3_4b_eagle3 --gpus 0 1 --datasets gsm8k math500
rtk bash scripts/download_models.sh
rtk bash scripts/download_datasets.sh
```

Use `prepare_data` before evaluation so `artifacts/samples/*.jsonl` exists. Edit `eval/config.json` before downloading models or datasets.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints on public helpers, `Path` over raw string paths where practical, and `snake_case` for functions, variables, and JSONL filenames. Keep constants uppercase, as in `DEFAULT_GPUS`. Match the current standard library-first style in `run_eval.py`; no formatter or linter is configured, so keep changes small and consistent with nearby code.

## Testing Guidelines
There is no dedicated `tests/` package yet. Validate changes with focused smoke runs:

```bash
rtk python -m eval.prepare_data --sample-size 20 --smoke
rtk python run_eval.py run-model --model qwen3_1p7b_eagle3 --gpus 0 --datasets gsm8k
```

When changing sampling, scoring, or backend dispatch, verify the relevant files under `artifacts/logs/<model>/` and `artifacts/reports/run_summary.json`. Keep dataset-specific fixtures and outputs out of git unless explicitly needed.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit-style prefixes such as `feat:`. Continue with concise, imperative subjects, for example `fix: handle missing draft config`. PRs should state which backend(s) were affected, list the exact validation command(s), note any required environment variables or conda envs, and include sample output paths when results changed.

## Configuration & Data Notes
Do not hardcode machine-specific model or dataset paths outside `eval/config.json`. Treat `third_party/` as vendored code: avoid local rewrites unless the task is explicitly about upstream integration or patching a submodule.
