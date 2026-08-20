#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot accepted and rejected speculative draft tokens by accept length."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Model log directory, dataset log directory, or spec_trace*.jsonl path.",
    )
    parser.add_argument("--out", type=Path, help="Output SVG path.")
    parser.add_argument("--csv", type=Path, help="Output CSV path.")
    parser.add_argument(
        "--title",
        help="Chart title. Defaults to the input directory or trace filename.",
    )
    return parser.parse_args()


def resolve_trace_path(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("spec_trace_raw.jsonl", "spec_trace.jsonl"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No spec_trace_raw.jsonl or spec_trace.jsonl found under {path}")


def read_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def draft_length(row: dict[str, Any]) -> int:
    if row.get("num_draft_tokens") is not None:
        return int(row["num_draft_tokens"])
    draft_tokens = row.get("draft_chunk_token_ids")
    if isinstance(draft_tokens, list):
        return len(draft_tokens)
    return 0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, int]]:
    steps_by_len: Counter[int] = Counter()
    accepted_by_len: Counter[int] = Counter()
    rejected_candidate_by_len: Counter[int] = Counter()
    unaccepted_draft_by_len: Counter[int] = Counter()
    draft_by_len: Counter[int] = Counter()
    max_draft_by_len: defaultdict[int, int] = defaultdict(int)

    for row in rows:
        if row.get("accept_len") is None:
            continue
        accept_len = int(row["accept_len"])
        num_draft = draft_length(row)
        rejected_candidates = row.get("rejected_candidate_token_ids")
        rejected_candidate_count = (
            len(rejected_candidates) if isinstance(rejected_candidates, list) else 0
        )

        steps_by_len[accept_len] += 1
        accepted_by_len[accept_len] += accept_len
        rejected_candidate_by_len[accept_len] += rejected_candidate_count
        unaccepted_draft_by_len[accept_len] += max(0, num_draft - accept_len)
        draft_by_len[accept_len] += num_draft
        max_draft_by_len[accept_len] = max(max_draft_by_len[accept_len], num_draft)

    lengths = sorted(steps_by_len)
    return [
        {
            "accept_len": length,
            "verify_steps": steps_by_len[length],
            "accepted_tokens": accepted_by_len[length],
            "rejected_candidate_tokens": rejected_candidate_by_len[length],
            "unaccepted_draft_tokens": unaccepted_draft_by_len[length],
            "draft_tokens": draft_by_len[length],
            "max_draft_tokens": max_draft_by_len[length],
        }
        for length in lengths
    ]


def write_csv(path: Path, rows: list[dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "accept_len",
        "verify_steps",
        "accepted_tokens",
        "rejected_candidate_tokens",
        "unaccepted_draft_tokens",
        "draft_tokens",
        "max_draft_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nice_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(value))


def y_ticks(max_value: int, count: int = 5) -> list[float]:
    if max_value <= 0:
        return [0]
    return [max_value * i / count for i in range(count + 1)]


def rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" />'


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 12,
    anchor: str = "middle",
    fill: str = "#263238",
    weight: str = "400",
) -> str:
    escaped = html.escape(value)
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{escaped}</text>'
    )


def render_panel(
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    rows: list[dict[str, int]],
    series: list[tuple[str, str, str]],
    ylabel: str,
    title: str,
    show_x_labels: bool,
) -> str:
    left = x0 + 58
    right = x0 + width - 20
    top = y0 + 34
    bottom = y0 + height - (48 if show_x_labels else 24)
    chart_w = right - left
    chart_h = bottom - top
    max_y = max([row[key] for row in rows for key, _, _ in series] + [1])
    group_w = chart_w / max(1, len(rows))
    bar_gap = 3
    bar_w = max(4, (group_w - 10 - bar_gap * (len(series) - 1)) / len(series))

    parts = [
        text(x0 + width / 2, y0 + 18, title, size=15, weight="700"),
        f'<line x1="{left:.2f}" y1="{bottom:.2f}" x2="{right:.2f}" y2="{bottom:.2f}" stroke="#607d8b" />',
        f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{bottom:.2f}" stroke="#607d8b" />',
        text(x0 + 16, top + chart_h / 2, ylabel, size=12, anchor="middle", fill="#455a64"),
    ]

    for tick in y_ticks(max_y):
        y = bottom - (tick / max_y) * chart_h
        parts.append(
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" y2="{y:.2f}" '
            'stroke="#d7dee2" stroke-width="1" />'
        )
        parts.append(text(left - 8, y + 4, nice_number(tick), anchor="end", fill="#607d8b"))

    for index, row in enumerate(rows):
        group_x = left + index * group_w + 5
        for series_index, (key, _label, color) in enumerate(series):
            value = row[key]
            h = (value / max_y) * chart_h
            x = group_x + series_index * (bar_w + bar_gap)
            y = bottom - h
            parts.append(rect(x, y, bar_w, h, color))
        if show_x_labels:
            parts.append(text(group_x + group_w / 2 - 5, bottom + 18, str(row["accept_len"]), size=11))

    legend_x = right - 260
    legend_y = y0 + 18
    for i, (_key, label, color) in enumerate(series):
        x = legend_x + i * 130
        parts.append(rect(x, legend_y - 10, 12, 12, color))
        parts.append(text(x + 18, legend_y, label, size=11, anchor="start", fill="#455a64"))

    return "\n".join(parts)


def render_svg(rows: list[dict[str, int]], title_value: str) -> str:
    width = 1180
    height = 760
    accepted = sum(row["accepted_tokens"] for row in rows)
    unaccepted = sum(row["unaccepted_draft_tokens"] for row in rows)
    rejected_recorded = sum(row["rejected_candidate_tokens"] for row in rows)
    steps = sum(row["verify_steps"] for row in rows)
    mean_accept = accepted / steps if steps else 0.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        text(width / 2, 34, title_value, size=22, weight="700"),
        text(
            width / 2,
            58,
            (
                f"verify steps={steps:,}  accepted={accepted:,}  "
                f"unaccepted draft={unaccepted:,}  recorded first rejected={rejected_recorded:,}  "
                f"mean accept={mean_accept:.2f}"
            ),
            size=13,
            fill="#455a64",
        ),
    ]
    parts.append(
        render_panel(
            x0=30,
            y0=82,
            width=1120,
            height=285,
            rows=rows,
            series=[("verify_steps", "verify steps", "#546e7a")],
            ylabel="steps",
            title="Accept Length Histogram",
            show_x_labels=False,
        )
    )
    parts.append(
        render_panel(
            x0=30,
            y0=400,
            width=1120,
            height=300,
            rows=rows,
            series=[
                ("accepted_tokens", "accepted", "#2e7d32"),
                ("unaccepted_draft_tokens", "unaccepted draft", "#c62828"),
            ],
            ylabel="tokens",
            title="Accepted vs Unaccepted Draft Tokens by Accept Length",
            show_x_labels=True,
        )
    )
    parts.append(text(width / 2, 736, "x-axis: accept_len per verification step", size=12, fill="#607d8b"))
    parts.append("</svg>")
    return "\n".join(parts)


def default_output_paths(trace_path: Path) -> tuple[Path, Path]:
    base_dir = trace_path.parent
    return (
        base_dir / "accept_reject_tokens_by_length.svg",
        base_dir / "accept_reject_tokens_by_length.csv",
    )


def main() -> None:
    args = parse_args()
    trace_path = resolve_trace_path(args.input)
    rows = summarize(read_trace_rows(trace_path))
    if not rows:
        raise ValueError(f"No accept_len rows found in {trace_path}")

    default_svg, default_csv = default_output_paths(trace_path)
    svg_path = args.out or default_svg
    csv_path = args.csv or default_csv
    title_value = args.title or f"{trace_path.parent.name}: accept/reject tokens by length"

    write_csv(csv_path, rows)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(render_svg(rows, title_value), encoding="utf-8")
    print(svg_path)
    print(csv_path)


if __name__ == "__main__":
    main()
