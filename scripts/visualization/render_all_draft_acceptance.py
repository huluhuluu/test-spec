#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render draft acceptance HTML for all trace directories under artifacts/logs."
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        type=Path,
        default=Path("artifacts/logs"),
        help="Root logs directory.",
    )
    parser.add_argument("--per-length-limit", type=int, default=12)
    parser.add_argument("--context-chars", type=int, default=140)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--open-lengths", default="0")
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


def output_path(input_path: Path, logs_dir: Path) -> Path:
    if input_path.parent == logs_dir:
        return input_path / "draft_acceptance_by_length.html"
    return input_path / "draft_acceptance_by_length.html"


def main() -> None:
    args = parse_args()
    logs_dir = args.logs_dir
    script = Path(__file__).with_name("render_draft_acceptance.py")
    rendered = []
    skipped = []

    for input_path in discover_inputs(logs_dir, args.include_datasets):
        out = output_path(input_path, logs_dir)
        cmd = [
            sys.executable,
            str(script),
            str(input_path),
            "--per-length-limit",
            str(args.per_length_limit),
            "--context-chars",
            str(args.context_chars),
            "--limit",
            str(args.limit),
            "--open-lengths",
            args.open_lengths,
            "--out",
            str(out),
        ]
        result = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True)
        if result.returncode == 0:
            rendered.append(out)
            print(out)
        else:
            skipped.append((input_path, result.stderr.strip() or result.stdout.strip()))

    if skipped:
        print("\nSkipped:", file=sys.stderr)
        for path, reason in skipped:
            print(f"- {path}: {reason}", file=sys.stderr)

    print(f"\nRendered {len(rendered)} file(s), skipped {len(skipped)}.", file=sys.stderr)


if __name__ == "__main__":
    main()
