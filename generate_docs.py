#!/usr/bin/env python3
"""
generate_docs.py — Build static HTML documentation from the fruit-thrower index.

Reads every indexed unit (module, class, function, method) directly from the
vector store — no Python imports needed — and renders a multi-page HTML site:

    docs/
        index.html          — module listing
        <module.name>.html  — one page per module, all its units as sections

Usage:
    python generate_docs.py
    python generate_docs.py --index-dir ./.code_index --out ./docs
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vector_store import CodeVectorStore, SimpleTFIDFStore

CSS = """
:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --border: #2a2d3a;
  --accent: #7c6af7;
  --accent2: #5bc4a0;
  --text: #d4d8e8;
  --muted: #7a7f96;
  --code-bg: #12151f;
  --fn: #e8a87c;
  --cls: #78c1f3;
  --mod: #a8d8a0;
  --method: #f3c478;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg); color: var(--text);
  display: flex; min-height: 100vh; font-size: 15px; line-height: 1.6;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* sidebar */
nav {
  width: 260px; min-width: 260px; background: var(--surface);
  border-right: 1px solid var(--border); padding: 24px 0;
  position: sticky; top: 0; height: 100vh; overflow-y: auto;
}
nav h2 { padding: 0 20px 16px; font-size: 13px; color: var(--muted);
          text-transform: uppercase; letter-spacing: .08em; }
nav ul { list-style: none; }
nav ul li a {
  display: block; padding: 5px 20px; font-size: 13px;
  color: var(--text); border-left: 2px solid transparent;
}
nav ul li a:hover, nav ul li a.active {
  color: var(--accent); border-left-color: var(--accent);
  background: rgba(124,106,247,.07); text-decoration: none;
}
nav .badge {
  float: right; font-size: 11px; color: var(--muted);
  background: var(--border); border-radius: 10px; padding: 1px 7px;
}

/* main content */
main { flex: 1; padding: 40px 48px; max-width: 900px; }
h1.page-title { font-size: 26px; color: var(--text); margin-bottom: 8px; }
p.page-meta { color: var(--muted); font-size: 13px; margin-bottom: 32px; }
hr.divider { border: none; border-top: 1px solid var(--border); margin: 32px 0; }

/* unit cards */
.unit {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 20px 24px; margin-bottom: 20px;
}
.unit-header { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.unit-name { font-size: 17px; font-weight: 600; font-family: monospace; }
.unit-name.function { color: var(--fn); }
.unit-name.method   { color: var(--method); }
.unit-name.class    { color: var(--cls); }
.unit-name.module   { color: var(--mod); }
.unit-kind {
  font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); background: var(--border);
  padding: 2px 8px; border-radius: 4px;
}
.unit-location { font-size: 12px; color: var(--muted); margin-top: 4px; font-family: monospace; }
.unit-location a { color: var(--muted); }
.signature-block {
  background: var(--code-bg); border: 1px solid var(--border);
  border-radius: 6px; padding: 12px 16px; margin: 14px 0 0;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 13px; white-space: pre-wrap; color: #c9d1d9; overflow-x: auto;
}
.docstring {
  margin-top: 14px; font-size: 14px; color: var(--text);
  white-space: pre-wrap; line-height: 1.7;
}
.no-doc { margin-top: 10px; font-size: 13px; color: var(--muted); font-style: italic; }

/* index page */
.module-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px 20px; margin-bottom: 12px;
  display: flex; justify-content: space-between; align-items: center;
}
.module-card a { font-size: 15px; font-weight: 500; font-family: monospace; color: var(--mod); }
.module-card .counts { font-size: 12px; color: var(--muted); }
.module-card .counts span { margin-left: 10px; }
"""

INDEX_SIDEBAR = """
<nav>
  <h2>fruit-thrower</h2>
  <ul>
    <li><a href="index.html" class="active">All Modules</a></li>
  </ul>
</nav>
"""

def module_sidebar(modules: list[tuple[str, dict]], current: str, out_dir: Path) -> str:
    items = []
    for name, counts in sorted(modules):
        href = module_filename(name)
        cls = ' class="active"' if name == current else ""
        n = counts.get("function", 0) + counts.get("method", 0) + counts.get("class", 0)
        items.append(f'<li><a href="{href}"{cls}>{name}<span class="badge">{n}</span></a></li>')
    return f"<nav><h2>Modules</h2><ul>{''.join(items)}</ul></nav>"


def module_filename(module: str) -> str:
    return module.replace(".", "_") + ".html"


def html_page(title: str, nav: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — fin-kit docs</title>
<style>{CSS}</style>
</head>
<body>
{nav}
<main>{body}</main>
</body>
</html>"""


def render_unit(u: dict) -> str:
    kind = u["kind"]
    name = u["name"]
    sig = u.get("signature", "")
    doc = u.get("docstring", "").strip()
    parent = u.get("parent", "")
    fp = u.get("file_path", "")
    line = u.get("line_start", "")

    qualified = f"{parent}.{name}" if parent else name
    location = f"{fp}:{line}" if fp else ""

    sig_html = f'<div class="signature-block">{_esc(sig)}</div>' if sig and kind != "module" else ""
    doc_html = f'<div class="docstring">{_esc(doc)}</div>' if doc else '<div class="no-doc">No docstring.</div>'
    loc_html = f'<div class="unit-location">{_esc(location)}</div>' if location else ""

    return f"""<div class="unit" id="{_esc(name)}">
  <div class="unit-header">
    <span class="unit-name {kind}">{_esc(qualified)}</span>
    <span class="unit-kind">{kind}</span>
  </div>
  {loc_html}
  {sig_html}
  {doc_html}
</div>"""


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


KIND_ORDER = {"module": 0, "class": 1, "function": 2, "method": 3}


def build(index_dir: str, out_dir: Path):
    try:
        store = CodeVectorStore(persist_dir=index_dir)
    except Exception:
        store = SimpleTFIDFStore(persist_dir=index_dir)

    # Fetch all records from whichever backend is in use
    if hasattr(store, "_records"):
        records = store._records
    else:
        result = store._collection.get(include=["metadatas", "documents"])
        records = []
        for uid, meta in zip(result["ids"], result["metadatas"]):
            records.append({"id": uid, **meta})

    if not records:
        print("Index is empty. Run: python main.py index --repo ./fin-kit")
        sys.exit(1)

    # Group by module (skip __init__ modules — surface their units under the package module)
    by_module: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        mod = r["module"]
        # Collapse __init__ into the package name
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        by_module[mod].append(r)

    # Count functions/classes per module for sidebar badges
    module_counts: list[tuple[str, dict]] = []
    for mod, units in by_module.items():
        counts: dict[str, int] = defaultdict(int)
        for u in units:
            counts[u["kind"]] += 1
        module_counts.append((mod, dict(counts)))

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Index page ──────────────────────────────────────────────────────────
    cards = []
    for mod, counts in sorted(module_counts):
        href = module_filename(mod)
        fn_count = counts.get("function", 0) + counts.get("method", 0)
        cls_count = counts.get("class", 0)
        cards.append(
            f'<div class="module-card">'
            f'<a href="{href}">{_esc(mod)}</a>'
            f'<div class="counts">'
            f'<span>⚡ {fn_count} fn</span>'
            f'<span>🧱 {cls_count} class</span>'
            f'</div></div>'
        )
    total_units = len(records)
    total_mods = len(by_module)
    index_body = f"""
<h1 class="page-title">fin-kit documentation</h1>
<p class="page-meta">{total_mods} modules · {total_units} indexed units · generated by fruit-thrower</p>
<hr class="divider">
{''.join(cards)}"""

    (out_dir / "index.html").write_text(
        html_page("Index", INDEX_SIDEBAR, index_body), encoding="utf-8"
    )

    # ── Module pages ─────────────────────────────────────────────────────────
    for mod, units in sorted(by_module.items()):
        units_sorted = sorted(units, key=lambda u: (KIND_ORDER.get(u["kind"], 9), u["line_start"]))
        nav = module_sidebar(module_counts, mod, out_dir)

        unit_html = "".join(render_unit(u) for u in units_sorted)
        fn_n = sum(1 for u in units if u["kind"] in ("function", "method"))
        cls_n = sum(1 for u in units if u["kind"] == "class")

        body = f"""
<h1 class="page-title">{_esc(mod)}</h1>
<p class="page-meta">{fn_n} functions/methods · {cls_n} classes</p>
<hr class="divider">
{unit_html}"""
        (out_dir / module_filename(mod)).write_text(
            html_page(mod, nav, body), encoding="utf-8"
        )

    print(f"Generated {len(by_module)} module pages → {out_dir}/index.html")
    return out_dir / "index.html"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="./.code_index")
    parser.add_argument("--out", default="./docs")
    args = parser.parse_args()
    index_path = build(args.index_dir, Path(args.out))
    import subprocess
    subprocess.run(["open", str(index_path)])


if __name__ == "__main__":
    main()
