#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


Signature = tuple[tuple[int, tuple["SignatureNode", ...]], ...]
SignatureNode = tuple[int, tuple[Any, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the most common speculative token trees.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--context-chars", type=int, default=180)
    parser.add_argument("--out", type=Path, default=Path("artifacts/reports/common_token_trees.html"))
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
    return "" if value is None else str(value)


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
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def root_indices(tree: dict[str, Any], nodes: list[dict[str, Any]]) -> list[int]:
    roots = tree.get("root_node_indices")
    if roots:
        return [int(root) for root in roots]
    return [int(node["node_index"]) for node in nodes if node.get("parent_node_index") is None]


def tree_signature(row: dict[str, Any]) -> Signature:
    tree = row.get("tree") or {}
    nodes = tree.get("nodes") or []
    by_id = {int(node["node_index"]): node for node in nodes}

    def node_sig(node_index: int) -> SignatureNode:
        node = by_id[node_index]
        children = tuple(node_sig(int(child)) for child in node.get("child_node_indices", []))
        return int(node["token_id"]), children

    return tuple(node_sig(root) for root in root_indices(tree, nodes))


def analyze_input(path: Path) -> dict[str, Any]:
    trace_path = resolve_trace_path(path)
    counter: Counter[Signature] = Counter()
    representative: dict[Signature, dict[str, Any]] = {}
    accept_lens: dict[Signature, Counter[int]] = defaultdict(Counter)
    event_count = 0

    for row in iter_rows(trace_path):
        tree = row.get("tree") or {}
        nodes = tree.get("nodes") or []
        if not nodes:
            continue
        sig = tree_signature(row)
        counter[sig] += 1
        representative.setdefault(sig, row)
        accept_lens[sig][int(row.get("accept_len") or 0)] += 1
        event_count += 1

    label = path.name if path.is_dir() else path.parent.name
    return {
        "label": label,
        "trace_path": trace_path,
        "event_count": event_count,
        "counter": counter,
        "representative": representative,
        "accept_lens": accept_lens,
    }


def percent(value: int, total: int) -> str:
    return f"{100.0 * value / total:.2f}%" if total else "0.00%"


def accept_summary(counter: Counter[int]) -> str:
    total = sum(counter.values())
    if not total:
        return ""
    avg = sum(length * count for length, count in counter.items()) / total
    top = ", ".join(f"{length}:{count}" for length, count in counter.most_common(6))
    return f"avg={avg:.2f}; dist {top}"


def render_node(node_index: int, nodes_by_id: dict[int, dict[str, Any]]) -> str:
    node = nodes_by_id[node_index]
    children = [int(child) for child in node.get("child_node_indices", [])]
    label = html.escape(visible_token(node.get("token")))
    title = html.escape(f"node={node_index} token_id={node.get('token_id')} position={node.get('position')}")
    child_html = ""
    if children:
        child_html = "<ul>" + "".join(render_node(child, nodes_by_id) for child in children) + "</ul>"
    return (
        "<li>"
        f'<span class="node" title="{title}"><span class="idx">{node_index}</span>{label}</span>'
        f"{child_html}</li>"
    )


def render_tree(row: dict[str, Any]) -> str:
    tree = row.get("tree") or {}
    nodes = tree.get("nodes") or []
    nodes_by_id = {int(node["node_index"]): node for node in nodes}
    if not nodes_by_id:
        return '<div class="empty">No tree nodes.</div>'
    roots = root_indices(tree, nodes)
    return '<ul class="tree">' + "".join(render_node(root, nodes_by_id) for root in roots) + "</ul>"


def render_case(result: dict[str, Any], sig: Signature, rank: int, count: int, context_chars: int) -> str:
    row = result["representative"][sig]
    tree = row.get("tree") or {}
    nodes = tree.get("nodes") or []
    prefix = html.escape(compact_text(row.get("prefix_text") or "", context_chars))
    committed = html.escape(compact_text(row.get("committed_text") or "", context_chars))
    accept = html.escape(accept_summary(result["accept_lens"][sig]))
    pct = percent(count, result["event_count"])
    return f"""
<details class="case" open>
  <summary>
    <span>#{rank}</span>
    <strong>{count:,}</strong>
    <em>{pct} of tree events</em>
    <em>{len(nodes)} nodes</em>
    <em>accept {accept}</em>
  </summary>
  <div class="context">
    <div><span>representative prefix tail</span><pre>{prefix}</pre></div>
    <div><span>representative committed</span><pre>{committed}</pre></div>
  </div>
  {render_tree(row)}
</details>"""


def render_model(result: dict[str, Any], top: int, min_count: int, context_chars: int) -> str:
    cases = [
        (sig, count)
        for sig, count in result["counter"].most_common()
        if count >= min_count
    ][:top]
    if cases:
        body = "".join(
            render_case(result, sig, rank, count, context_chars)
            for rank, (sig, count) in enumerate(cases, 1)
        )
    else:
        body = '<div class="empty">No repeated full token trees matched this threshold.</div>'
    repeated_events = sum(count for count in result["counter"].values() if count > 1)
    return f"""
<section class="model">
  <h2>{html.escape(result['label'])}</h2>
  <div class="stats">
    <div><strong>{result['event_count']:,}</strong><span>tree events</span></div>
    <div><strong>{len(result['counter']):,}</strong><span>unique full token trees</span></div>
    <div><strong>{repeated_events:,}</strong><span>events in repeated trees</span></div>
    <div><strong>{percent(repeated_events, result['event_count'])}</strong><span>repeated full-tree rate</span></div>
  </div>
  {body}
</section>"""


def render_html(results: list[dict[str, Any]], top: int, min_count: int, context_chars: int) -> str:
    models = "".join(render_model(result, top, min_count, context_chars) for result in results)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Common Token Trees</title>
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
  font-size: 24px;
}}
h2 {{
  margin: 0 0 12px;
  font-size: 20px;
}}
.sub {{
  margin-bottom: 18px;
  color: #607d8b;
  font-size: 13px;
}}
.model {{
  margin-bottom: 24px;
  padding: 16px;
  border: 1px solid #d8e0e5;
  border-radius: 6px;
  background: #fff;
}}
.stats {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}}
.stats div {{
  padding: 10px;
  border-radius: 6px;
  background: #f2f5f7;
}}
.stats strong {{
  display: block;
  font-size: 18px;
}}
.stats span {{
  color: #607d8b;
  font-size: 12px;
}}
.case {{
  margin-top: 12px;
  padding: 10px;
  border: 1px solid #e1e8ec;
  border-radius: 6px;
  background: #fbfcfd;
}}
.case summary {{
  cursor: pointer;
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  align-items: center;
  color: #37474f;
}}
.case summary strong {{
  color: #1b5e20;
}}
.case summary em {{
  color: #607d8b;
  font-style: normal;
  font-size: 12px;
}}
.context {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 10px;
  margin: 10px 0;
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
  background: #eef3f5;
  font-size: 12px;
  white-space: pre-wrap;
}}
.tree, .tree ul {{
  position: relative;
  margin: 0;
  padding-left: 24px;
  list-style: none;
}}
.tree ul {{
  margin-left: 10px;
}}
.tree li {{
  position: relative;
  margin: 7px 0;
  padding-left: 14px;
}}
.tree li::before {{
  content: "";
  position: absolute;
  top: 13px;
  left: -8px;
  width: 18px;
  border-top: 1px solid #b0bec5;
}}
.tree li::after {{
  content: "";
  position: absolute;
  top: -8px;
  bottom: -8px;
  left: -8px;
  border-left: 1px solid #b0bec5;
}}
.tree li:last-child::after {{
  bottom: auto;
  height: 21px;
}}
.tree > li::before,
.tree > li::after {{
  display: none;
}}
.node {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  min-height: 22px;
  padding: 2px 7px;
  border: 1px solid #90a4ae;
  border-radius: 4px;
  background: #fff;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}}
.idx {{
  color: #78909c;
  font-size: 10px;
}}
.empty {{
  padding: 12px;
  color: #607d8b;
  background: #f2f5f7;
  border-radius: 6px;
}}
@media (max-width: 760px) {{
  body {{ padding: 14px; }}
  .context {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<h1>Most Common Full Token Trees</h1>
<div class="sub">Trees are grouped per model by exact topology and token ids. Each item shows one representative trace for that repeated tree.</div>
{models}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    results = [analyze_input(path) for path in args.inputs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(results, args.top, args.min_count, args.context_chars), encoding="utf-8")
    print(args.out)
    for result in results:
        repeated = sum(count for count in result["counter"].values() if count > 1)
        print(
            result["label"],
            "events",
            result["event_count"],
            "unique_trees",
            len(result["counter"]),
            "repeated_events",
            repeated,
        )


if __name__ == "__main__":
    main()
