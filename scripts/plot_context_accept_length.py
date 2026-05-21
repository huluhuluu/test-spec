#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def resolve_summary_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "model_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_context_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    summary_path = resolve_summary_path(path)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = data.get("context_accept_length") or []
    if not rows:
        raise ValueError(f"No context_accept_length rows found in {summary_path}")
    label = data.get("model_key") or summary_path.parent.name
    return label, rows


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed = []
    running_sum = 0.0
    for i, value in enumerate(values):
        running_sum += value
        if i >= window:
            running_sum -= values[i - window]
        count = min(i + 1, window)
        smoothed.append(running_sum / count)
    return smoothed


def default_output_path(inputs: list[Path]) -> Path:
    if len(inputs) == 1:
        path = resolve_summary_path(inputs[0])
        return path.with_name("context_accept_length.png")
    return Path("artifacts/reports/context_accept_length.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean accepted draft length by context length."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="model_summary.json path or model log directory.",
    )
    parser.add_argument("--out", type=Path, help="Output image path.")
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Only plot context lengths with at least this many trace events.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window over sorted context lengths.",
    )
    parser.add_argument("--title", default="Context Length vs Accept Length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))

    for input_path in args.inputs:
        label, rows = load_context_rows(input_path)
        rows = [
            row
            for row in rows
            if int(row.get("count", 0)) >= args.min_count
        ]
        if not rows:
            raise ValueError(f"No rows left after --min-count for {input_path}")

        rows = sorted(rows, key=lambda row: int(row["context_len"]))
        x = [int(row["context_len"]) for row in rows]
        y = [float(row["mean_accept_len"]) for row in rows]
        y = moving_average(y, args.smooth_window)
        ax.plot(x, y, linewidth=1.6, label=label)

    ax.set_title(args.title)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Mean accepted draft length")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = args.out or default_output_path(args.inputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
