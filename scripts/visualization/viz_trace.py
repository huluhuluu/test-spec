#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render EAGLE3 speculative tree traces.")
    parser.add_argument("--trace", type=Path, required=True, help="Path to spec_trace.jsonl")
    parser.add_argument("--rid", help="Only render one request id")
    parser.add_argument("--step", type=int, help="Only render one step index")
    parser.add_argument("--limit", type=int, default=5, help="Max events to render")
    parser.add_argument("--out", type=Path, help="Optional output text file")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    rid: str | None,
    step: int | None,
) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        if rid is not None and row.get("rid") != rid:
            continue
        if step is not None and int(row.get("step_index", -1)) != step:
            continue
        kept.append(row)
    return kept


def node_label(node: dict[str, Any], accepted_set: set[int], frontier_children: set[int]) -> str:
    token = node.get("token")
    idx = int(node["node_index"])
    marker = ""
    if idx in accepted_set:
        marker = " [accepted]"
    elif idx in frontier_children:
        marker = " [frontier]"
    return f"{idx}: {repr(token)}{marker}"


def render_subtree(
    nodes_by_id: dict[int, dict[str, Any]],
    node_index: int,
    accepted_set: set[int],
    frontier_children: set[int],
    prefix: str,
    is_last: bool,
    lines: list[str],
) -> None:
    node = nodes_by_id[node_index]
    branch = "└─ " if is_last else "├─ "
    lines.append(prefix + branch + node_label(node, accepted_set, frontier_children))
    children = [int(x) for x in node.get("child_node_indices", [])]
    child_prefix = prefix + ("   " if is_last else "│  ")
    for idx, child in enumerate(children):
        render_subtree(
            nodes_by_id,
            child,
            accepted_set,
            frontier_children,
            child_prefix,
            idx == len(children) - 1,
            lines,
        )


def render_event(row: dict[str, Any]) -> str:
    tree = row.get("tree") or {}
    verify = row.get("verify") or {}
    nodes = tree.get("nodes") or []
    root_indices = [int(x) for x in tree.get("root_node_indices") or []]
    accepted_nodes = [int(x) for x in verify.get("accepted_tree_node_indices") or []]
    frontier_children = [int(x) for x in verify.get("frontier_child_node_indices") or []]
    nodes_by_id = {int(node["node_index"]): node for node in nodes}

    lines: list[str] = []
    lines.append(f"rid: {row.get('rid')}")
    lines.append(f"step: {row.get('step_index')}")
    lines.append(f"prefix: {row.get('prefix_text', '')!r}")
    lines.append(
        "accepted_path: "
        + " -> ".join(repr(tok) for tok in (verify.get("accepted_draft_tokens") or []))
    )
    lines.append(f"bonus_token: {verify.get('bonus_token')!r}")
    lines.append("tree:")

    accepted_set = set(accepted_nodes)
    frontier_set = set(frontier_children)
    for idx, root in enumerate(root_indices):
        render_subtree(
            nodes_by_id,
            root,
            accepted_set,
            frontier_set,
            "",
            idx == len(root_indices) - 1,
            lines,
        )

    first_divergence = verify.get("first_divergence") or {}
    lines.append(
        "first_divergence: "
        + json.dumps(
            {
                "frontier_node_index": first_divergence.get("frontier_node_index"),
                "draft_child_tokens": first_divergence.get("draft_child_tokens"),
                "target_token": first_divergence.get("target_token"),
            },
            ensure_ascii=False,
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.trace)
    rows = filter_rows(rows, rid=args.rid, step=args.step)
    rows = rows[: args.limit]

    blocks = [render_event(row) for row in rows]
    output = "\n\n" + ("-" * 80) + "\n\n"
    text = output.join(blocks) if blocks else ""

    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
