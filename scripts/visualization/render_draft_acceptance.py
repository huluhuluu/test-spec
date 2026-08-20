#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render per-step speculative draft tokens with an accept/reject separator."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Model log directory, dataset log directory, or spec_trace*.jsonl path.",
    )
    parser.add_argument("--rid", help="Only render one request id.")
    parser.add_argument("--step", type=int, help="Only render one step index.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max trace events to render after grouping; 0 means no global limit.",
    )
    parser.add_argument(
        "--per-length-limit",
        type=int,
        default=20,
        help="Max events to render for each accept_len group; 0 means no per-group limit.",
    )
    parser.add_argument(
        "--format",
        choices=["html", "txt"],
        default="html",
        help="Output format.",
    )
    parser.add_argument("--out", type=Path, help="Output path.")
    parser.add_argument(
        "--show-prefix",
        action="store_true",
        help="Deprecated; prefix context is shown by default unless --context-chars=0.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=140,
        help="Characters of prefix/committed context to show per event; 0 hides context.",
    )
    parser.add_argument(
        "--open-lengths",
        default="0",
        help="Comma-separated accept_len groups to expand by default; use 'all' or 'none'.",
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


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_rows(
    path: Path,
    *,
    rid: str | None,
    step: int | None,
    limit: int,
    per_length_limit: int,
) -> list[dict[str, Any]]:
    kept = []
    per_length_counts: dict[int, int] = {}
    for row in iter_rows(path):
        if row.get("accept_len") is None:
            continue
        if rid is not None and row.get("rid") != rid:
            continue
        if step is not None and int(row.get("step_index", -1)) != step:
            continue
        accept_len = int(row.get("accept_len") or 0)
        if per_length_limit > 0:
            current = per_length_counts.get(accept_len, 0)
            if current >= per_length_limit:
                continue
            per_length_counts[accept_len] = current + 1
        kept.append(row)
        if limit > 0 and len(kept) >= limit:
            break
    return sorted(kept, key=lambda row: (int(row.get("accept_len") or 0), str(row.get("rid")), int(row.get("step_index") or 0)))


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


def compact_text(value: Any, max_chars: int) -> str:
    text = token_text(value).replace("\r", "\\r").replace("\t", "\\t")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def draft_tokens(row: dict[str, Any]) -> list[str]:
    tokens = row.get("draft_chunk_tokens")
    if isinstance(tokens, list):
        return [token_text(token) for token in tokens]
    block = row.get("draft_block") or {}
    tokens = block.get("candidate_chain_tokens")
    if isinstance(tokens, list):
        return [token_text(token) for token in tokens[1:]]
    return []


def split_draft(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    tokens = draft_tokens(row)
    accept_len = int(row.get("accept_len") or 0)
    return tokens[:accept_len], tokens[accept_len:]


def separator_note(row: dict[str, Any]) -> str:
    accept_len = int(row.get("accept_len") or 0)
    tokens = draft_tokens(row)
    if accept_len >= len(tokens):
        return "all draft tokens accepted"
    rejected = tokens[accept_len] if accept_len < len(tokens) else ""
    replacement = row.get("replacement_token")
    if replacement is None:
        replacement = (row.get("verify") or {}).get("bonus_token")
    return f"first rejected={visible_token(rejected)!r}, replacement={visible_token(replacement)!r}"


def render_token_spans(tokens: list[str], class_name: str) -> str:
    if not tokens:
        return '<span class="empty">∅</span>'
    return "".join(
        f'<span class="tok {class_name}" title="{html.escape(repr(token))}">'
        f"{html.escape(visible_token(token))}</span>"
        for token in tokens
    )


def rows_by_accept_len(rows: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        accept_len = int(row.get("accept_len") or 0)
        grouped.setdefault(accept_len, []).append(row)
    return [(length, grouped[length]) for length in sorted(grouped)]


def context_html(row: dict[str, Any], context_chars: int) -> str:
    if context_chars <= 0:
        return ""
    prefix = compact_text(row.get("prefix_text") or "", context_chars)
    committed = compact_text(row.get("committed_text") or "", context_chars)
    return (
        '<div class="context">'
        f'<div><span>prefix tail</span><pre>{html.escape(prefix)}</pre></div>'
        f'<div><span>committed</span><pre>{html.escape(committed)}</pre></div>'
        "</div>"
    )


def parse_open_lengths(value: str) -> set[int] | None:
    value = value.strip().lower()
    if value == "all":
        return None
    if value in {"", "none"}:
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def render_html(
    rows: list[dict[str, Any]],
    trace_path: Path,
    context_chars: int,
    open_lengths: set[int] | None,
) -> str:
    body = []
    event_index = 0
    for accept_len_group, group_rows in rows_by_accept_len(rows):
        open_attr = " open" if open_lengths is None or accept_len_group in open_lengths else ""
        body.append(
            f'<details class="length-group" id="accept-len-{accept_len_group}"{open_attr}>'
            f"<summary>accept_len = {accept_len_group} <span>{len(group_rows)} shown</span></summary>"
        )
        for row in group_rows:
            event_index += 1
            accepted, rejected = split_draft(row)
            total = len(accepted) + len(rejected)
            meta = (
                f"#{event_index} rid={row.get('rid')} step={row.get('step_index')} "
                f"context={row.get('context_len')} accept={accept_len_group}/{total}"
            )
            body.append(
                f"""
<section class="event">
  <div class="meta">{html.escape(meta)}</div>
  {context_html(row, context_chars)}
  <div class="draft">
    <span class="accepted">{render_token_spans(accepted, "ok")}</span>
    <span class="sep">|</span>
    <span class="rejected">{render_token_spans(rejected, "bad")}</span>
  </div>
  <div class="note">{html.escape(separator_note(row))}</div>
</section>"""
            )
        body.append("</details>")

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Draft Acceptance Trace</title>
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
  margin: 22px 0 12px;
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
.event {{
  margin: 0 0 12px;
  padding: 12px 14px;
  border: 1px solid #d8e0e5;
  border-radius: 6px;
  background: #fff;
}}
.meta {{
  margin-bottom: 8px;
  color: #455a64;
  font-size: 12px;
  font-weight: 700;
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
  margin-top: 6px;
  color: #607d8b;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}}
.context {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 10px;
  margin-bottom: 8px;
}}
.context span {{
  display: block;
  margin-bottom: 3px;
  color: #607d8b;
  font-size: 11px;
  font-weight: 700;
}}
.context pre {{
  max-height: 90px;
  overflow: auto;
  margin: 0;
  padding: 7px;
  border-radius: 4px;
  background: #f2f5f7;
  font-size: 12px;
  white-space: pre-wrap;
}}
@media (max-width: 820px) {{
  .context {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
<h1>Draft Acceptance Trace</h1>
<div class="sub">{html.escape(str(trace_path))} · grouped by accept_len · showing {len(rows)} events · separator marks accept_len</div>
{''.join(body)}
</body>
</html>
"""


def render_txt(rows: list[dict[str, Any]], context_chars: int) -> str:
    blocks = []
    event_index = 0
    for accept_len_group, group_rows in rows_by_accept_len(rows):
        blocks.append(f"accept_len = {accept_len_group} ({len(group_rows)} shown)")
        for row in group_rows:
            event_index += 1
            accepted, rejected = split_draft(row)
            total = len(accepted) + len(rejected)
            accepted_text = " ".join(visible_token(token) for token in accepted) or "∅"
            rejected_text = " ".join(visible_token(token) for token in rejected) or "∅"
            lines = [
                (
                    f"#{event_index} rid={row.get('rid')} step={row.get('step_index')} "
                    f"context={row.get('context_len')} accept={accept_len_group}/{total}"
                ),
            ]
            if context_chars > 0:
                lines.extend(
                    [
                        f"prefix tail: {compact_text(row.get('prefix_text') or '', context_chars)}",
                        f"committed: {compact_text(row.get('committed_text') or '', context_chars)}",
                    ]
                )
            lines.extend(
                [
                    f"draft: {accepted_text} | {rejected_text}",
                    separator_note(row),
                ]
            )
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def default_output_path(trace_path: Path, output_format: str) -> Path:
    suffix = "html" if output_format == "html" else "txt"
    return trace_path.parent / f"draft_acceptance_trace.{suffix}"


def main() -> None:
    args = parse_args()
    trace_path = resolve_trace_path(args.input)
    rows = collect_rows(
        trace_path,
        rid=args.rid,
        step=args.step,
        limit=args.limit,
        per_length_limit=args.per_length_limit,
    )
    if not rows:
        raise ValueError("No trace events matched the filters.")

    text = (
        render_html(rows, trace_path, args.context_chars, parse_open_lengths(args.open_lengths))
        if args.format == "html"
        else render_txt(rows, args.context_chars)
    )
    out = args.out or default_output_path(trace_path, args.format)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
