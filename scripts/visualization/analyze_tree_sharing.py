#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze internal sharing in speculative draft trees.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("artifacts/reports/tree_sharing.html"))
    parser.add_argument("--csv-dir", type=Path, default=Path("artifacts/reports/tree_sharing"))
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


def tree_paths(row: dict[str, Any]) -> tuple[list[tuple[int, ...]], list[tuple[str, ...]], tuple[tuple[int, ...], ...]]:
    tree = row.get("tree") or {}
    nodes = tree.get("nodes") or []
    if not nodes:
        return [], [], ()
    by_id = {int(node["node_index"]): node for node in nodes}
    cache: dict[int, tuple[int, ...]] = {}

    def path_ids(node_index: int) -> tuple[int, ...]:
        if node_index in cache:
            return cache[node_index]
        node = by_id[node_index]
        token_id = int(node["token_id"])
        parent = node.get("parent_node_index")
        if parent is None:
            path = (token_id,)
        else:
            path = path_ids(int(parent)) + (token_id,)
        cache[node_index] = path
        return path

    id_paths = [path_ids(int(node["node_index"])) for node in nodes]
    token_lookup = {int(node["token_id"]): token_text(node.get("token")) for node in nodes}
    text_paths = [tuple(token_lookup.get(token_id, str(token_id)) for token_id in path) for path in id_paths]
    signature = tuple(sorted(id_paths))
    return id_paths, text_paths, signature


def analyze(path: Path) -> dict[str, Any]:
    trace_path = resolve_trace_path(path)
    token_counter: Counter[int] = Counter()
    token_text_by_id: dict[int, str] = {}
    path_counter: Counter[tuple[int, ...]] = Counter()
    path_text: dict[tuple[int, ...], tuple[str, ...]] = {}
    signature_counter: Counter[tuple[tuple[int, ...], ...]] = Counter()
    event_count = 0
    node_count = 0

    for row in iter_rows(trace_path):
        tree = row.get("tree") or {}
        nodes = tree.get("nodes") or []
        if not nodes:
            continue
        event_count += 1
        node_count += len(nodes)
        for node in nodes:
            token_id = int(node["token_id"])
            token_counter[token_id] += 1
            token_text_by_id.setdefault(token_id, token_text(node.get("token")))
        id_paths, text_paths, signature = tree_paths(row)
        for id_path, text_path in zip(id_paths, text_paths):
            path_counter[id_path] += 1
            path_text.setdefault(id_path, text_path)
        signature_counter[signature] += 1

    unique_tokens = len(token_counter)
    unique_paths = len(path_counter)
    unique_signatures = len(signature_counter)
    repeated_token_instances = sum(count for count in token_counter.values() if count > 1)
    repeated_path_instances = sum(count for count in path_counter.values() if count > 1)
    repeated_signature_events = sum(count for count in signature_counter.values() if count > 1)

    return {
        "label": path.name if path.is_dir() else path.parent.name,
        "input": path,
        "trace_path": trace_path,
        "event_count": event_count,
        "node_count": node_count,
        "unique_tokens": unique_tokens,
        "unique_paths": unique_paths,
        "unique_signatures": unique_signatures,
        "token_counter": token_counter,
        "token_text_by_id": token_text_by_id,
        "path_counter": path_counter,
        "path_text": path_text,
        "signature_counter": signature_counter,
        "repeated_token_instances": repeated_token_instances,
        "repeated_path_instances": repeated_path_instances,
        "repeated_signature_events": repeated_signature_events,
    }


def percent(value: int, total: int) -> str:
    return f"{(100.0 * value / total):.2f}%" if total else "0.00%"


def write_csvs(result: dict[str, Any], csv_dir: Path, top: int) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    safe_label = result["label"].replace("/", "_")
    token_path = csv_dir / f"{safe_label}_tokens.csv"
    path_path = csv_dir / f"{safe_label}_paths.csv"
    with token_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "token_id", "token", "count", "percent_nodes"])
        writer.writeheader()
        for rank, (token_id, count) in enumerate(result["token_counter"].most_common(), 1):
            writer.writerow(
                {
                    "rank": rank,
                    "token_id": token_id,
                    "token": result["token_text_by_id"].get(token_id, ""),
                    "count": count,
                    "percent_nodes": percent(count, result["node_count"]),
                }
            )
    with path_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "path_len", "path_tokens", "path_token_ids", "count", "percent_nodes"])
        writer.writeheader()
        for rank, (path_ids, count) in enumerate(result["path_counter"].most_common(), 1):
            writer.writerow(
                {
                    "rank": rank,
                    "path_len": len(path_ids),
                    "path_tokens": json.dumps(result["path_text"].get(path_ids, ()), ensure_ascii=False),
                    "path_token_ids": json.dumps(path_ids),
                    "count": count,
                    "percent_nodes": percent(count, result["node_count"]),
                }
            )


def bar(width_pct: float) -> str:
    width = max(0.5, min(100.0, width_pct))
    return f'<div class="bar"><span style="width:{width:.2f}%"></span></div>'


def render_top_tokens(result: dict[str, Any], top: int) -> str:
    max_count = result["token_counter"].most_common(1)[0][1] if result["token_counter"] else 1
    rows = []
    for rank, (token_id, count) in enumerate(result["token_counter"].most_common(top), 1):
        token = visible_token(result["token_text_by_id"].get(token_id, ""))
        rows.append(
            "<tr>"
            f"<td>{rank}</td><td><code>{token_id}</code></td><td><code>{html.escape(token)}</code></td>"
            f"<td>{count:,}</td><td>{percent(count, result['node_count'])}</td>"
            f"<td>{bar(100.0 * count / max_count)}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_top_paths(result: dict[str, Any], top: int) -> str:
    max_count = result["path_counter"].most_common(1)[0][1] if result["path_counter"] else 1
    rows = []
    for rank, (path_ids, count) in enumerate(result["path_counter"].most_common(top), 1):
        tokens = " ".join(visible_token(token) for token in result["path_text"].get(path_ids, ()))
        rows.append(
            "<tr>"
            f"<td>{rank}</td><td>{len(path_ids)}</td><td><code>{html.escape(tokens)}</code></td>"
            f"<td>{count:,}</td><td>{percent(count, result['node_count'])}</td>"
            f"<td>{bar(100.0 * count / max_count)}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_html(results: list[dict[str, Any]], top: int) -> str:
    sections = []
    for result in results:
        node_count = result["node_count"]
        event_count = result["event_count"]
        sections.append(
            f"""
<section class="model">
  <h2>{html.escape(result['label'])}</h2>
  <div class="summary">
    <div><strong>{event_count:,}</strong><span>tree events</span></div>
    <div><strong>{node_count:,}</strong><span>tree nodes</span></div>
    <div><strong>{result['unique_tokens']:,}</strong><span>unique token ids</span></div>
    <div><strong>{result['unique_paths']:,}</strong><span>unique root-to-node paths</span></div>
    <div><strong>{percent(result['repeated_token_instances'], node_count)}</strong><span>nodes using repeated token ids</span></div>
    <div><strong>{percent(result['repeated_path_instances'], node_count)}</strong><span>nodes using repeated paths</span></div>
    <div><strong>{percent(result['repeated_signature_events'], event_count)}</strong><span>events using repeated full trees</span></div>
  </div>
  <details open>
    <summary>Top shared token ids</summary>
    <table><thead><tr><th>#</th><th>token id</th><th>token</th><th>count</th><th>% nodes</th><th></th></tr></thead><tbody>{render_top_tokens(result, top)}</tbody></table>
  </details>
  <details>
    <summary>Top shared root-to-node paths</summary>
    <table><thead><tr><th>#</th><th>path len</th><th>path tokens</th><th>count</th><th>% nodes</th><th></th></tr></thead><tbody>{render_top_paths(result, top)}</tbody></table>
  </details>
</section>"""
        )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Tree Sharing Analysis</title>
<style>
body {{
  margin: 0;
  padding: 24px;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #263238;
  background: #f5f7f8;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 24px;
}}
h2 {{
  margin: 0 0 12px;
  font-size: 20px;
}}
.sub {{
  margin-bottom: 18px;
  color: #607d8b;
}}
.model {{
  margin-bottom: 22px;
  padding: 16px;
  border: 1px solid #d8e0e5;
  border-radius: 6px;
  background: #fff;
}}
.summary {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}}
.summary div {{
  padding: 10px;
  border-radius: 6px;
  background: #f2f5f7;
}}
.summary strong {{
  display: block;
  font-size: 18px;
}}
.summary span {{
  color: #607d8b;
  font-size: 12px;
}}
details {{
  margin-top: 10px;
}}
summary {{
  cursor: pointer;
  font-weight: 700;
  color: #37474f;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 8px;
  font-size: 13px;
}}
th, td {{
  padding: 6px 8px;
  border-bottom: 1px solid #e1e8ec;
  text-align: left;
  vertical-align: top;
}}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap;
}}
.bar {{
  width: 140px;
  height: 8px;
  margin-top: 4px;
  border-radius: 999px;
  background: #e1e8ec;
}}
.bar span {{
  display: block;
  height: 8px;
  border-radius: 999px;
  background: #2e7d32;
}}
</style>
</head>
<body>
<h1>Tree Sharing Analysis</h1>
<div class="sub">Internal sharing is computed per model. Paths are root-to-node token-id paths inside each draft tree.</div>
{''.join(sections)}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    results = [analyze(path) for path in args.inputs]
    for result in results:
        write_csvs(result, args.csv_dir, args.top)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(results, args.top), encoding="utf-8")
    print(args.out)
    for result in results:
        print(result["label"], "events", result["event_count"], "nodes", result["node_count"])


if __name__ == "__main__":
    main()
