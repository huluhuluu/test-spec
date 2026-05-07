from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Optional

from eval.config_loader import dataset_local_dir


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
THIRD_PARTY_DIR = ROOT / "third_party"
SAMPLES_DIR = ARTIFACTS / "samples"
REPORTS_DIR = ARTIFACTS / "reports"
DEFAULT_CMMLU_REPO = THIRD_PARTY_DIR / "CMMLU"
DATASET_NAMES = ["gsm8k", "math500", "mtbench", "humaneval", "ceval", "cmmlu"]


def ensure_prepare_dirs() -> None:
    for path in (ARTIFACTS, THIRD_PARTY_DIR, SAMPLES_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def prepare_datasets(
    sample_size: int,
    seed: int,
    cmmlu_repo: Path,
    dataset_names: Optional[list[str]] = None,
    force: bool = False,
) -> dict[str, Path]:
    ensure_prepare_dirs()
    paths = {}
    for dataset_name in (dataset_names or DATASET_NAMES):
        path = SAMPLES_DIR / f"{dataset_name}.jsonl"
        if path.exists() and not force:
            print(f"[reuse-sample] {dataset_name}: {path}")
            paths[dataset_name] = path
            continue

        rows = build_dataset_samples(dataset_name, sample_size, seed, cmmlu_repo)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[write-sample] {dataset_name}: {path} ({len(rows)} rows)")
        paths[dataset_name] = path
    return paths


def build_dataset_samples(dataset_name: str, sample_size: int, seed: int, cmmlu_repo: Path) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if dataset_name == "gsm8k":
        rows = load_gsm8k()
    elif dataset_name == "math500":
        rows = load_math500()
    elif dataset_name == "mtbench":
        rows = load_mtbench()
    elif dataset_name == "humaneval":
        rows = load_humaneval()
    elif dataset_name == "ceval":
        rows = load_ceval()
    elif dataset_name == "cmmlu":
        rows = load_cmmlu(cmmlu_repo)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name == "mtbench":
        sample = rows[:sample_size]
    else:
        if len(rows) < sample_size:
            raise ValueError(f"{dataset_name} only has {len(rows)} rows")
        sample = rng.sample(rows, sample_size)

    for idx, row in enumerate(sample):
        row["dataset"] = dataset_name
        row["sample_index"] = idx
    return sample


def local_dataset_dir(dataset_name: str) -> Path:
    path = dataset_local_dir(dataset_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing local dataset {dataset_name}: {path}. "
            "Run `bash scripts/download_datasets.sh` first."
        )
    return path


def require_files(dataset_name: str, *patterns: str) -> list[Path]:
    base_path = local_dataset_dir(dataset_name)
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(base_path.glob(pattern)))
    if not paths:
        expected = ", ".join(str(base_path / pattern) for pattern in patterns)
        raise FileNotFoundError(
            f"Missing local files for {dataset_name}: {expected}. "
            "Run `bash scripts/download_datasets.sh` first."
        )
    return paths


def read_parquet_rows(dataset_name: str, *patterns: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to prepare local parquet datasets.") from exc

    rows: list[dict[str, Any]] = []
    for path in require_files(dataset_name, *patterns):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def read_local_jsonl_rows(dataset_name: str, *patterns: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in require_files(dataset_name, *patterns):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_gsm8k() -> list[dict[str, Any]]:
    ds = read_parquet_rows("gsm8k", "main/test-*.parquet")
    rows = []
    for item in ds:
        rows.append(
            {
                "sample_id": item["question"][:48],
                "question": item["question"],
                "gold_answer": item["answer"],
            }
        )
    return rows


def load_math500() -> list[dict[str, Any]]:
    ds = read_local_jsonl_rows("math500", "test.jsonl")
    rows = []
    for item in ds:
        rows.append(
            {
                "sample_id": item["unique_id"],
                "question": item["problem"],
                "gold_answer": item["answer"],
                "subject": item["subject"],
                "level": item["level"],
                "solution": item["solution"],
            }
        )
    return rows


def load_mtbench() -> list[dict[str, Any]]:
    ds = read_parquet_rows("mtbench", "data/*.parquet")
    rows = []
    for item in ds:
        rows.append(
            {
                "sample_id": str(item["prompt_id"]),
                "category": item["category"],
                "turns": item["prompt"],
                "reference": item["reference"],
            }
        )
    return rows


def load_humaneval() -> list[dict[str, Any]]:
    ds = read_parquet_rows("humaneval", "openai_humaneval/test-*.parquet")
    rows = []
    for item in ds:
        rows.append(
            {
                "sample_id": item["task_id"],
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "entry_point": item["entry_point"],
                "test": item["test"],
                "canonical_solution": item["canonical_solution"],
            }
        )
    return rows


def load_ceval() -> list[dict[str, Any]]:
    rows = []
    for val_path in require_files("ceval", "*/val-*.parquet"):
        subject = val_path.parent.name
        ds = read_parquet_rows("ceval", f"{subject}/val-*.parquet")
        for item in ds:
            rows.append(
                {
                    "sample_id": f"{subject}:{item['id']}",
                    "subject": subject,
                    "question": item["question"],
                    "choices": {
                        "A": item["A"],
                        "B": item["B"],
                        "C": item["C"],
                        "D": item["D"],
                    },
                    "gold_answer": item["answer"],
                    "explanation": item.get("explanation", ""),
                }
            )
    return rows


def ensure_cmmlu_repo(cmmlu_repo: Path) -> Path:
    test_dir = cmmlu_repo / "data" / "test"
    if test_dir.exists():
        return cmmlu_repo

    if cmmlu_repo.exists():
        raise FileNotFoundError(f"CMMLU repo exists but dataset files are missing: {test_dir}")

    raise FileNotFoundError(
        "CMMLU repo not found. Run `bash scripts/download_datasets.sh` first, "
        "or run `git submodule update --init --recursive --depth 1 -- third_party/CMMLU`. "
        f"Missing path: {cmmlu_repo}"
    )


def load_cmmlu(cmmlu_repo: Path) -> list[dict[str, Any]]:
    cmmlu_repo = ensure_cmmlu_repo(cmmlu_repo)
    rows = []
    for csv_path in sorted((cmmlu_repo / "data" / "test").glob("*.csv")):
        subject = csv_path.stem
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, item in enumerate(reader):
                qid = (
                    item.get("Unnamed: 0")
                    or item.get("")
                    or item.get("id")
                    or item.get("ID")
                    or str(row_idx)
                )
                question = item.get("Question") or item.get("question")
                answer = item.get("Answer") or item.get("answer")
                rows.append(
                    {
                        "sample_id": f"{subject}:{qid}",
                        "subject": subject,
                        "question": question,
                        "choices": {
                            "A": item["A"],
                            "B": item["B"],
                            "C": item["C"],
                            "D": item["D"],
                        },
                        "gold_answer": answer,
                    }
                )
    return rows


def smoke_test(cmmlu_repo: Path, sample_size: int, seed: int, dataset_names: list[str]) -> None:
    paths = prepare_datasets(
        sample_size=sample_size,
        seed=seed,
        cmmlu_repo=cmmlu_repo,
        dataset_names=dataset_names,
    )
    summary = {}
    for name, path in paths.items():
        rows = read_jsonl(path)
        summary[name] = {"count": len(rows), "first_sample_id": rows[0]["sample_id"]}
    json_dump(REPORTS_DIR / "smoke_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EAGLE-3 benchmark samples")
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--cmmlu-repo", type=Path, default=DEFAULT_CMMLU_REPO)
    parser.add_argument("--datasets", nargs="*", default=DATASET_NAMES, choices=DATASET_NAMES)
    parser.add_argument("--force-resample", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke_test(args.cmmlu_repo, args.sample_size, args.seed, args.datasets)
        return

    prepare_datasets(
        sample_size=args.sample_size,
        seed=args.seed,
        cmmlu_repo=args.cmmlu_repo,
        dataset_names=args.datasets,
        force=args.force_resample,
    )


if __name__ == "__main__":
    main()
