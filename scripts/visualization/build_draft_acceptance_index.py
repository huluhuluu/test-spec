#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an index page for draft acceptance HTML files.")
    parser.add_argument("logs_dir", nargs="?", type=Path, default=Path("artifacts/logs"))
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_dir = args.logs_dir
    out = args.out or logs_dir / "draft_acceptance_index.html"
    page_dirs = sorted(
        {
            page.parent
            for pattern in ("**/draft_acceptance_by_length.html", "**/draft_acceptance_frequency.html")
            for page in logs_dir.glob(pattern)
        }
    )

    items = []
    for page_dir in page_dirs:
        label = str(page_dir.relative_to(logs_dir))
        links = []
        detail_page = page_dir / "draft_acceptance_by_length.html"
        frequency_page = page_dir / "draft_acceptance_frequency.html"
        if detail_page.exists():
            rel = detail_page.relative_to(logs_dir)
            links.append(f'<a href="{html.escape(str(rel))}">samples</a>')
        if frequency_page.exists():
            rel = frequency_page.relative_to(logs_dir)
            links.append(f'<a href="{html.escape(str(rel))}">frequency</a>')
        items.append(f'<li><strong>{html.escape(label)}</strong><span>{" · ".join(links)}</span></li>')

    out.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Draft Acceptance Visualizations</title>
<style>
body {{
  margin: 0;
  padding: 28px;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #263238;
  background: #f5f7f8;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 24px;
}}
.sub {{
  margin-bottom: 18px;
  color: #607d8b;
}}
ul {{
  margin: 0;
  padding: 0;
  columns: 2;
  column-gap: 24px;
}}
li {{
  break-inside: avoid;
  list-style: none;
  margin: 0 0 8px;
  padding: 9px 11px;
  border: 1px solid #d8e0e5;
  border-radius: 6px;
  background: #fff;
}}
li strong {{
  display: block;
  margin-bottom: 5px;
}}
li span {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}}
a {{
  color: #1565c0;
  text-decoration: none;
}}
a:hover {{
  text-decoration: underline;
}}
@media (max-width: 820px) {{
  ul {{
    columns: 1;
  }}
}}
</style>
</head>
<body>
<h1>Draft Acceptance Visualizations</h1>
<div class="sub">{len(page_dirs)} generated directories under {html.escape(str(logs_dir))}</div>
<ul>
{''.join(items)}
</ul>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
