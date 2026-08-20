#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tree_trace.html for trace directories.")
    parser.add_argument("logs_dir", nargs="?", type=Path, default=Path("artifacts/logs"))
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--context-chars", type=int, default=140)
    parser.add_argument(
        "--include-datasets",
        action="store_true",
        help="Also render dataset subdirectories when a model-level spec_trace_raw.jsonl exists.",
    )
    return parser.parse_args()


def discover_inputs(logs_dir: Path, include_datasets: bool) -> list[Path]:
    inputs = []
    for model_dir in sorted(path for path in logs_dir.iterdir() if path.is_dir()):
        raw_trace = model_dir / "spec_trace_raw.jsonl"
        if raw_trace.exists():
            inputs.append(model_dir)
            if not include_datasets:
                continue
        for dataset_trace in sorted(model_dir.glob("*/spec_trace.jsonl")):
            inputs.append(dataset_trace.parent)
    return inputs


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name("render_tree_trace.py")
    rendered = []
    skipped = []

    for input_path in discover_inputs(args.logs_dir, args.include_datasets):
        cmd = [
            sys.executable,
            str(script),
            str(input_path),
            "--limit",
            str(args.limit),
            "--context-chars",
            str(args.context_chars),
        ]
        result = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True)
        if result.returncode == 0:
            outputs = [line for line in result.stdout.splitlines() if line.strip()]
            rendered.extend(outputs)
            for output in outputs:
                print(output)
        else:
            skipped.append((input_path, result.stderr.strip() or result.stdout.strip()))

    if skipped:
        print("\nSkipped:", file=sys.stderr)
        for path, reason in skipped:
            print(f"- {path}: {reason}", file=sys.stderr)

    print(f"\nRendered {len(rendered)} tree file(s), skipped {len(skipped)}.", file=sys.stderr)


if __name__ == "__main__":
    main()
