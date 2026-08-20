#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from render_tree_trace import render_tree as render_tree_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate draft token sequences by accept_len and render frequency-sorted HTML."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Model log directory, dataset log directory, or spec_trace*.jsonl path.",
    )
    parser.add_argument(
        "--per-length-limit",
        type=int,
        default=200,
        help="Max aggregate rows to show per accept_len group; 0 means all.",
    )
    parser.add_argument("--out", type=Path, help="Output HTML path.")
    parser.add_argument("--csv", type=Path, help="Output CSV path.")
    parser.add_argument("--open-lengths", default="none")
    return parser.parse_args()


def resolve_trace_path(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("spec_trace_raw.jsonl", "spec_trace.jsonl"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No spec_trace_raw.jsonl or spec_trace.jsonl found under {path}")


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def token_text(token: Any) -> str:
    if token is None:
        return ""
    return str(token)


def visible_token(token: Any) -> str:
    text = token_text(token)
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace(" ", "·")
    )


def draft_tokens(row: dict[str, Any]) -> tuple[str, ...]:
    tokens = row.get("draft_chunk_tokens")
    if isinstance(tokens, list):
        return tuple(token_text(token) for token in tokens)
    block = row.get("draft_block") or {}
    tokens = block.get("candidate_chain_tokens")
    if isinstance(tokens, list):
        return tuple(token_text(token) for token in tokens[1:])
    return ()


def replacement_token(row: dict[str, Any]) -> str:
    replacement = row.get("replacement_token")
    if replacement is None:
        replacement = (row.get("verify") or {}).get("bonus_token")
    return token_text(replacement)


DraftKey = tuple[tuple[str, ...], str]


def aggregate(path: Path) -> tuple[dict[int, Counter[DraftKey]], Counter[int], int]:
    grouped: dict[int, Counter[tuple[tuple[str, ...], str]]] = {}
    totals: Counter[int] = Counter()
    skipped = 0
    for row in iter_rows(path):
        if row.get("accept_len") is None:
            skipped += 1
            continue
        tokens = draft_tokens(row)
        if not tokens:
            skipped += 1
            continue
        accept_len = int(row["accept_len"])
        key = (tokens, replacement_token(row))
        grouped.setdefault(accept_len, Counter())[key] += 1
        totals[accept_len] += 1
    return grouped, totals, skipped


def top_keys_by_length(
    grouped: dict[int, Counter[DraftKey]],
    per_length_limit: int,
) -> dict[int, set[DraftKey]]:
    wanted = {}
    for accept_len, counter in grouped.items():
        rows = counter.most_common(None if per_length_limit <= 0 else per_length_limit)
        wanted[accept_len] = {key for key, _count in rows}
    return wanted


def collect_tree_examples(
    path: Path,
    wanted: dict[int, set[DraftKey]],
) -> dict[int, dict[DraftKey, dict[str, Any]]]:
    examples: dict[int, dict[DraftKey, dict[str, Any]]] = {}
    remaining = sum(len(keys) for keys in wanted.values())
    if remaining == 0:
        return examples
    for row in iter_rows(path):
        if not isinstance(row.get("tree"), dict) or row.get("accept_len") is None:
            continue
        accept_len = int(row["accept_len"])
        keys = wanted.get(accept_len)
        if not keys:
            continue
        tokens = draft_tokens(row)
        if not tokens:
            continue
        key = (tokens, replacement_token(row))
        if key not in keys:
            continue
        length_examples = examples.setdefault(accept_len, {})
        if key in length_examples:
            continue
        length_examples[key] = row
        remaining -= 1
        if remaining <= 0:
            break
    return examples


def split_tokens(tokens: tuple[str, ...], accept_len: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tokens[:accept_len], tokens[accept_len:]


def render_token_spans(tokens: tuple[str, ...], class_name: str) -> str:
    if not tokens:
        return '<span class="empty">∅</span>'
    return "".join(
        f'<span class="tok {class_name}" title="{html.escape(repr(token))}">'
        f"{html.escape(visible_token(token))}</span>"
        for token in tokens
    )


def parse_open_lengths(value: str) -> set[int] | None:
    value = value.strip().lower()
    if value == "all":
        return None
    if value in {"", "none"}:
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def pct(count: int, total: int) -> str:
    return f"{(100.0 * count / total):.2f}%" if total else "0.00%"


def compact_text(value: Any, max_chars: int) -> str:
    text = token_text(value).replace("\r", "\\r").replace("\t", "\\t")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def render_tree_example(row: dict[str, Any]) -> str:
    verify = row.get("verify") or {}
    accepted_path = " ".join(visible_token(token) for token in verify.get("accepted_draft_tokens") or [])
    frontier = " ".join(visible_token(token) for token in verify.get("frontier_child_tokens") or [])
    target = visible_token((verify.get("first_divergence") or {}).get("target_token"))
    bonus = visible_token(verify.get("bonus_token"))
    prefix = compact_text(row.get("prefix_text") or "", 140)
    committed = compact_text(row.get("committed_text") or "", 140)
    return f"""
<div class="tree-event">
  <div class="tree-meta">rid={html.escape(str(row.get('rid')))} step={row.get('step_index')} context={row.get('context_len')} accept_len={row.get('accept_len')}</div>
  <div class="tree-context">
    <div><span>prefix tail</span><pre>{html.escape(prefix)}</pre></div>
    <div><span>committed</span><pre>{html.escape(committed)}</pre></div>
  </div>
  <div class="tree-facts">
    <span>accepted path: <code>{html.escape(accepted_path or '∅')}</code></span>
    <span>frontier: <code>{html.escape(frontier or '∅')}</code></span>
    <span>target: <code>{html.escape(target or '∅')}</code></span>
    <span>bonus: <code>{html.escape(bonus or '∅')}</code></span>
  </div>
  {render_tree_html(row)}
</div>"""


def render_html(
    grouped: dict[int, Counter[tuple[tuple[str, ...], str]]],
    totals: Counter[int],
    trace_path: Path,
    per_length_limit: int,
    open_lengths: set[int] | None,
    skipped: int,
    tree_examples: dict[int, dict[DraftKey, dict[str, Any]]],
) -> str:
    body = []
    total_events = sum(totals.values())
    for accept_len in sorted(grouped):
        counter = grouped[accept_len]
        rows = counter.most_common(None if per_length_limit <= 0 else per_length_limit)
        shown = len(rows)
        unique = len(counter)
        open_attr = " open" if open_lengths is None or accept_len in open_lengths else ""
        body.append(
            f'<details class="length-group" id="accept-len-{accept_len}"{open_attr}>'
            f"<summary>accept_len = {accept_len} "
            f"<span>{totals[accept_len]:,} events · {unique:,} unique · {shown:,} shown</span></summary>"
        )
        for rank, ((tokens, replacement), count) in enumerate(rows, 1):
            accepted, rejected = split_tokens(tokens, accept_len)
            tree_row = tree_examples.get(accept_len, {}).get((tokens, replacement))
            tree_html = (
                '<details class="case-tree"><summary>tree</summary>'
                + render_tree_example(tree_row)
                + "</details>"
                if tree_row is not None
                else ""
            )
            body.append(
                f"""
<section class="row">
  <div class="meta">
    <span class="rank">#{rank}</span>
    <span>{count:,} accepted</span>
    <span>{pct(count, totals[accept_len])} of accept_len={accept_len}</span>
    {tree_html}
  </div>
  <div class="draft">
    <span class="accepted">{render_token_spans(accepted, "ok")}</span>
    <span class="sep">|</span>
    <span class="rejected">{render_token_spans(rejected, "bad")}</span>
  </div>
  <div class="note">replacement={html.escape(repr(visible_token(replacement)))}</div>
</section>"""
            )
        body.append("</details>")

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Draft Acceptance Frequency Summary</title>
<style>
body {{
  margin: 0;
  padding: 24px;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #263238;
  background: #f5f7f8;
}}
h1 {{
  margin: 0 0 6px;
  font-size: 22px;
}}
.sub {{
  margin-bottom: 18px;
  color: #607d8b;
  font-size: 13px;
}}
details.length-group {{
  margin: 18px 0 12px;
}}
summary {{
  position: sticky;
  top: 0;
  z-index: 1;
  display: block;
  cursor: pointer;
  padding: 9px 10px;
  border: 1px solid #cfd8dc;
  border-radius: 6px;
  background: #eef3f5;
  font-size: 16px;
  font-weight: 700;
}}
summary::-webkit-details-marker {{
  display: none;
}}
summary::before {{
  content: "▶";
  display: inline-block;
  width: 18px;
  color: #607d8b;
}}
details[open] > summary::before {{
  content: "▼";
}}
summary span {{
  margin-left: 8px;
  color: #607d8b;
  font-size: 12px;
  font-weight: 400;
}}
.row {{
  margin: 0 0 9px;
  padding: 10px 12px;
  border: 1px solid #d8e0e5;
  border-radius: 6px;
  background: #fff;
}}
.meta {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 7px;
  color: #455a64;
  font-size: 12px;
  font-weight: 700;
}}
.rank {{
  color: #263238;
}}
.draft {{
  overflow-x: auto;
  white-space: nowrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
  line-height: 2.2;
}}
.tok {{
  display: inline-block;
  margin: 0 2px 3px 0;
  padding: 1px 5px;
  border-radius: 4px;
  border: 1px solid transparent;
}}
.ok {{
  background: #e7f4ea;
  border-color: #b9dfc1;
  color: #1b5e20;
}}
.bad {{
  background: #fdebea;
  border-color: #f4c2bd;
  color: #8f1d15;
}}
.sep {{
  display: inline-block;
  margin: 0 8px;
  color: #c62828;
  font-weight: 800;
  font-size: 18px;
  vertical-align: -1px;
}}
.empty {{
  color: #90a4ae;
}}
.note {{
  margin-top: 5px;
  color: #607d8b;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}}
.case-tree {{
  flex-basis: 100%;
  margin-top: 4px;
}}
.case-tree > summary {{
  position: static;
  display: inline-block;
  padding: 3px 9px;
  border-color: #b0bec5;
  background: #f2f5f7;
  font-size: 12px;
  font-weight: 700;
}}
.case-tree > summary::before {{
  content: "";
  display: none;
}}
.tree-event {{
  margin: 10px 0;
  padding: 10px 12px;
  border: 1px solid #d8e0e5;
  border-radius: 6px;
  background: #fff;
}}
.tree-meta {{
  margin-bottom: 8px;
  color: #455a64;
  font-size: 12px;
  font-weight: 700;
}}
.tree-context {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 10px;
  margin-bottom: 8px;
}}
.tree-context span {{
  display: block;
  margin-bottom: 3px;
  color: #607d8b;
  font-size: 11px;
  font-weight: 700;
}}
.tree-context pre {{
  max-height: 90px;
  overflow: auto;
  margin: 0;
  padding: 7px;
  border-radius: 4px;
  background: #f2f5f7;
  font-size: 12px;
  white-space: pre-wrap;
}}
.tree-facts {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  margin-bottom: 10px;
  color: #455a64;
  font-size: 12px;
}}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.tree, .tree ul {{
  position: relative;
  margin: 0;
  padding-left: 22px;
  list-style: none;
}}
.tree ul {{
  margin-left: 10px;
}}
.tree li {{
  position: relative;
  margin: 5px 0;
  padding-left: 12px;
}}
.tree li::before {{
  content: "";
  position: absolute;
  top: 12px;
  left: -10px;
  width: 18px;
  border-top: 1px solid #b0bec5;
}}
.tree li::after {{
  content: "";
  position: absolute;
  top: -6px;
  bottom: 10px;
  left: -10px;
  border-left: 1px solid #b0bec5;
}}
.tree > li::before,
.tree > li::after {{
  display: none;
}}
.node {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 6px;
  border: 1px solid #d8e0e5;
  border-radius: 4px;
  background: #fff;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}}
.idx {{
  color: #78909c;
  font-size: 10px;
}}
.node.accepted {{
  border-color: #9bd2a7;
  background: #e7f4ea;
  color: #1b5e20;
}}
.node.frontier {{
  border-color: #f0b8b2;
  background: #fdebea;
  color: #8f1d15;
}}
.node.divergence {{
  box-shadow: 0 0 0 2px #ffcc80 inset;
}}
@media (max-width: 820px) {{
  .tree-context {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
<h1>Draft Acceptance Frequency Summary</h1>
<div class="sub">{html.escape(str(trace_path))} · {total_events:,} trace events · skipped {skipped:,} non-token events</div>
{''.join(body)}
</body>
</html>
"""


def write_csv(
    path: Path,
    grouped: dict[int, Counter[tuple[tuple[str, ...], str]]],
    totals: Counter[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "accept_len",
                "rank",
                "count",
                "percent_within_accept_len",
                "draft_tokens",
                "accepted_tokens",
                "rejected_tokens",
                "replacement_token",
            ],
        )
        writer.writeheader()
        for accept_len in sorted(grouped):
            for rank, ((tokens, replacement), count) in enumerate(grouped[accept_len].most_common(), 1):
                accepted, rejected = split_tokens(tokens, accept_len)
                writer.writerow(
                    {
                        "accept_len": accept_len,
                        "rank": rank,
                        "count": count,
                        "percent_within_accept_len": pct(count, totals[accept_len]),
                        "draft_tokens": json.dumps(tokens, ensure_ascii=False),
                        "accepted_tokens": json.dumps(accepted, ensure_ascii=False),
                        "rejected_tokens": json.dumps(rejected, ensure_ascii=False),
                        "replacement_token": replacement,
                    }
                )


def default_output_paths(trace_path: Path) -> tuple[Path, Path]:
    return (
        trace_path.parent / "draft_acceptance_frequency.html",
        trace_path.parent / "draft_acceptance_frequency.csv",
    )


def main() -> None:
    args = parse_args()
    trace_path = resolve_trace_path(args.input)
    grouped, totals, skipped = aggregate(trace_path)
    if not grouped:
        raise ValueError("No per-token draft trace events found.")
    tree_examples = collect_tree_examples(
        trace_path,
        top_keys_by_length(grouped, args.per_length_limit),
    )

    default_html, default_csv = default_output_paths(trace_path)
    out = args.out or default_html
    csv_out = args.csv or default_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_html(
            grouped,
            totals,
            trace_path,
            args.per_length_limit,
            parse_open_lengths(args.open_lengths),
            skipped,
            tree_examples,
        ),
        encoding="utf-8",
    )
    write_csv(csv_out, grouped, totals)
    print(out)
    print(csv_out)


if __name__ == "__main__":
    main()
