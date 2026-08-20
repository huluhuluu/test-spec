#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render speculative draft trees as HTML.")
    parser.add_argument(
        "input",
        type=Path,
        help="Model log directory, dataset log directory, or spec_trace*.jsonl path.",
    )
    parser.add_argument("--rid", help="Only render one request id.")
    parser.add_argument("--step", type=int, help="Only render one step index.")
    parser.add_argument("--limit", type=int, default=80, help="Max tree events to render.")
    parser.add_argument("--context-chars", type=int, default=140)
    parser.add_argument("--out", type=Path)
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


def token_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def visible_token(value: Any) -> str:
    return (
        token_text(value)
        .replace("\\", "\\\\")
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


def collect_rows(
    path: Path,
    *,
    rid: str | None,
    step: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for row in iter_rows(path):
        if not isinstance(row.get("tree"), dict):
            continue
        if rid is not None and row.get("rid") != rid:
            continue
        if step is not None and int(row.get("step_index", -1)) != step:
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def node_classes(
    node_index: int,
    accepted_nodes: set[int],
    frontier_nodes: set[int],
    divergence_node: int | None,
) -> str:
    classes = ["node"]
    if node_index in accepted_nodes:
        classes.append("accepted")
    if node_index in frontier_nodes:
        classes.append("frontier")
    if divergence_node is not None and node_index == divergence_node:
        classes.append("divergence")
    return " ".join(classes)


def render_node(
    node_index: int,
    nodes_by_id: dict[int, dict[str, Any]],
    accepted_nodes: set[int],
    frontier_nodes: set[int],
    divergence_node: int | None,
) -> str:
    node = nodes_by_id[node_index]
    children = [int(child) for child in node.get("child_node_indices", [])]
    label = html.escape(visible_token(node.get("token")))
    title = html.escape(
        f"node={node_index} token_id={node.get('token_id')} position={node.get('position')}"
    )
    child_html = ""
    if children:
        child_html = "<ul>" + "".join(
            render_node(child, nodes_by_id, accepted_nodes, frontier_nodes, divergence_node)
            for child in children
        ) + "</ul>"
    return (
        "<li>"
        f'<span class="{node_classes(node_index, accepted_nodes, frontier_nodes, divergence_node)}" title="{title}">'
        f'<span class="idx">{node_index}</span>{label}</span>'
        f"{child_html}</li>"
    )


def render_tree(row: dict[str, Any]) -> str:
    tree = row.get("tree") or {}
    verify = row.get("verify") or {}
    nodes = tree.get("nodes") or []
    nodes_by_id = {int(node["node_index"]): node for node in nodes}
    root_indices = [int(root) for root in tree.get("root_node_indices") or []]
    accepted_nodes = {int(idx) for idx in verify.get("accepted_tree_node_indices") or []}
    frontier_nodes = {int(idx) for idx in verify.get("frontier_child_node_indices") or []}
    first_divergence = verify.get("first_divergence") or {}
    divergence_node = first_divergence.get("frontier_node_index")
    divergence_node = int(divergence_node) if divergence_node is not None else None

    if not nodes_by_id or not root_indices:
        return '<div class="empty">No tree nodes.</div>'

    return '<ul class="tree">' + "".join(
        render_node(root, nodes_by_id, accepted_nodes, frontier_nodes, divergence_node)
        for root in root_indices
    ) + "</ul>"


def render_event(row: dict[str, Any], index: int, context_chars: int) -> str:
    verify = row.get("verify") or {}
    accepted_path = " ".join(visible_token(token) for token in verify.get("accepted_draft_tokens") or [])
    frontier = " ".join(visible_token(token) for token in verify.get("frontier_child_tokens") or [])
    target = visible_token((verify.get("first_divergence") or {}).get("target_token"))
    bonus = visible_token(verify.get("bonus_token"))
    prefix = compact_text(row.get("prefix_text") or "", context_chars)
    committed = compact_text(row.get("committed_text") or "", context_chars)
    return f"""
<section class="event">
  <div class="meta">#{index} rid={html.escape(str(row.get('rid')))} step={row.get('step_index')} context={row.get('context_len')} accept_len={row.get('accept_len')}</div>
  <div class="context">
    <div><span>prefix tail</span><pre>{html.escape(prefix)}</pre></div>
    <div><span>committed</span><pre>{html.escape(committed)}</pre></div>
  </div>
  <div class="facts">
    <span>accepted path: <code>{html.escape(accepted_path or '∅')}</code></span>
    <span>frontier: <code>{html.escape(frontier or '∅')}</code></span>
    <span>target: <code>{html.escape(target or '∅')}</code></span>
    <span>bonus: <code>{html.escape(bonus or '∅')}</code></span>
  </div>
  {render_tree(row)}
</section>"""


def render_html(rows: list[dict[str, Any]], trace_path: Path, context_chars: int) -> str:
    body = "".join(render_event(row, index, context_chars) for index, row in enumerate(rows, 1))
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Speculative Tree Trace</title>
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
.event {{
  margin: 0 0 18px;
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
.facts {{
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
.accepted {{
  border-color: #9bd2a7;
  background: #e7f4ea;
  color: #1b5e20;
}}
.frontier {{
  border-color: #f0b8b2;
  background: #fdebea;
  color: #8f1d15;
}}
.divergence {{
  box-shadow: 0 0 0 2px #ffcc80 inset;
}}
@media (max-width: 820px) {{
  .context {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
<h1>Speculative Tree Trace</h1>
<div class="sub">{html.escape(str(trace_path))} · showing {len(rows)} tree events · green=accepted path, red=frontier candidates, orange outline=divergence node</div>
{body}
</body>
</html>
"""


def default_output_path(trace_path: Path) -> Path:
    return trace_path.parent / "tree_trace.html"


def main() -> None:
    args = parse_args()
    trace_path = resolve_trace_path(args.input)
    rows = collect_rows(trace_path, rid=args.rid, step=args.step, limit=args.limit)
    if not rows:
        raise ValueError("No tree trace events matched the filters.")
    out = args.out or default_output_path(trace_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(rows, trace_path, args.context_chars), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
