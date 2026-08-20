#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrieNode:
    token: str
    count: int = 0
    children: dict[str, "TrieNode"] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render frequent root-to-node paths as prefix token trees.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Model log directories or *_paths.csv files.")
    parser.add_argument("--csv-dir", type=Path, default=Path("artifacts/reports/tree_sharing"))
    parser.add_argument("--top-paths", type=int, default=120)
    parser.add_argument("--top-roots", type=int, default=8)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("artifacts/reports/common_prefix_token_trees.html"))
    return parser.parse_args()


def label_for_input(path: Path) -> str:
    if path.name.endswith("_paths.csv"):
        return path.name[: -len("_paths.csv")]
    return path.name if path.is_dir() else path.parent.name


def csv_for_input(path: Path, csv_dir: Path) -> Path:
    if path.is_file() and path.name.endswith("_paths.csv"):
        return path
    return csv_dir / f"{label_for_input(path)}_paths.csv"


def visible_token(value: Any) -> str:
    return (
        ("" if value is None else str(value))
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace(" ", "·")
    )


def add_path(root: TrieNode, tokens: list[str], count: int) -> None:
    node = root
    for token in tokens:
        node = node.children.setdefault(token, TrieNode(token=token))
    node.count = max(node.count, count)


def load_model(path: Path, csv_dir: Path, top_paths: int, min_len: int, min_count: int) -> dict[str, Any]:
    csv_path = csv_for_input(path, csv_dir)
    roots: dict[str, TrieNode] = {}
    selected = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            path_len = int(row["path_len"])
            count = int(row["count"])
            if path_len < min_len or count < min_count:
                continue
            tokens = json.loads(row["path_tokens"])
            if not tokens:
                continue
            selected.append((tokens, count))
            root = roots.setdefault(tokens[0], TrieNode(token=tokens[0]))
            add_path(root, tokens[1:], count)
            if len(selected) >= top_paths:
                break
    return {
        "label": label_for_input(path),
        "csv_path": csv_path,
        "roots": roots,
        "selected": selected,
    }


def subtree_score(node: TrieNode) -> int:
    return max([node.count, *(subtree_score(child) for child in node.children.values())])


def render_node(node: TrieNode, depth: int = 0) -> str:
    children = sorted(node.children.values(), key=subtree_score, reverse=True)
    label = html.escape(visible_token(node.token))
    count = f'<span class="count">{node.count:,}</span>' if node.count else ""
    child_html = ""
    if children:
        child_html = "<ul>" + "".join(render_node(child, depth + 1) for child in children) + "</ul>"
    return f'<li><span class="node depth-{min(depth, 6)}">{label}{count}</span>{child_html}</li>'


def render_root(root: TrieNode, rank: int) -> str:
    score = subtree_score(root)
    label = html.escape(visible_token(root.token))
    children = sorted(root.children.values(), key=subtree_score, reverse=True)
    body = "<ul class=\"tree\">" + "".join(render_node(child, 1) for child in children) + "</ul>"
    return f"""
<details class="root" open>
  <summary><span>#{rank}</span><strong>{label}</strong><em>best path count {score:,}</em></summary>
  {body}
</details>"""


def render_model(model: dict[str, Any], top_roots: int) -> str:
    roots = sorted(model["roots"].values(), key=subtree_score, reverse=True)[:top_roots]
    if roots:
        body = "".join(render_root(root, rank) for rank, root in enumerate(roots, 1))
    else:
        body = '<div class="empty">No shared prefix paths matched this threshold.</div>'
    return f"""
<section class="model">
  <h2>{html.escape(model['label'])}</h2>
  <div class="meta">
    <span>{len(model['selected']):,} selected frequent paths</span>
    <span>{html.escape(str(model['csv_path']))}</span>
  </div>
  {body}
</section>"""


def render_html(models: list[dict[str, Any]], args: argparse.Namespace) -> str:
    body = "".join(render_model(model, args.top_roots) for model in models)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Common Prefix Token Trees</title>
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
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  margin-bottom: 12px;
  color: #607d8b;
  font-size: 12px;
}}
.root {{
  margin-top: 12px;
  padding: 10px;
  border: 1px solid #e1e8ec;
  border-radius: 6px;
  background: #fbfcfd;
}}
.root summary {{
  cursor: pointer;
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  align-items: center;
}}
.root summary strong {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
.root summary em {{
  color: #607d8b;
  font-style: normal;
  font-size: 12px;
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
  gap: 7px;
  min-height: 22px;
  padding: 2px 7px;
  border: 1px solid #90a4ae;
  border-radius: 4px;
  background: #fff;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}}
.depth-1 {{ border-color: #78909c; }}
.depth-2 {{ border-color: #00897b; }}
.depth-3 {{ border-color: #5e35b1; }}
.depth-4 {{ border-color: #ef6c00; }}
.count {{
  color: #1b5e20;
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 11px;
  font-weight: 700;
}}
.empty {{
  padding: 12px;
  color: #607d8b;
  background: #f2f5f7;
  border-radius: 6px;
}}
</style>
</head>
<body>
<h1>Common Prefix Token Trees</h1>
<div class="sub">Built from the top {args.top_paths} repeated root-to-node paths per model, min path length {args.min_len}. Counts on nodes are exact full-path counts for that prefix ending point.</div>
{body}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    models = [load_model(path, args.csv_dir, args.top_paths, args.min_len, args.min_count) for path in args.inputs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(models, args), encoding="utf-8")
    print(args.out)
    for model in models:
        print(model["label"], "selected_paths", len(model["selected"]), "roots", len(model["roots"]))


if __name__ == "__main__":
    main()
