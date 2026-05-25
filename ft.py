#!/usr/bin/env python3
"""
ft.py — fruit-thrower CLI

Usage:
    python ft.py search "rolling sharpe ratio"
    python ft.py search "cusum filter" --n 3 --kind function
    python ft.py generate "a function that computes rolling Sharpe ratio"
    python ft.py generate "HRP weights from a covariance matrix" --module mlfinlab/portfolio/hrp.py
    python ft.py search "rolling sharpe"   # confirm it's now indexed
    python ft.py docs                      # serve live HTML docs at localhost:8080
    python ft.py docs --build              # build static HTML to docs/
    python ft.py convert --repo ./fin-kit --dry-run   # preview Sphinx→NumPy conversion
    python ft.py convert --repo ./fin-kit              # convert and re-index
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

# Load .env from repo root if present
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

DEFAULT_SERVER = "http://127.0.0.1:8000/sse"


async def _call(server: str, tool: str, args: dict) -> str:
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return result.content[0].text


def search(server: str, query: str, n: int, kind: str | None, module: str | None):
    args = {"query": query, "n_results": n}
    if kind:
        args["kind"] = kind
    if module:
        args["module"] = module
    print(asyncio.run(_call(server, "search_code", args)))


def generate(
    server: str,
    request: str,
    module_path: str | None,
    agent: str | None,
    model: str | None,
):
    args = {"request": request}
    if module_path:
        args["module_path"] = module_path
    if agent:
        args["agent"] = agent
    if model:
        args["model"] = model
    print(asyncio.run(_call(server, "generate_function", args)))


def convert(
    repo: str,
    api_key: str | None,
    openai_api_key: str | None,
    dry_run: bool,
    module_filter: str | None,
    force: bool,
    index_dir: str,
    agent: str | None = None,
):
    """Convert Sphinx :param:/:return: docstrings to NumPy style, then re-index."""
    _ft_dir = Path(__file__).parent
    sys.path.insert(0, str(_ft_dir))

    from docstring_generator import convert_all_docstrings
    from parser import parse_repository

    # Key resolution: explicit flags > env vars; no check needed here — _make_client raises clearly
    if not dry_run:
        ant_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        oai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not ant_key and not oai_key:
            print(
                "ERROR: Set ANTHROPIC_API_KEY or OPENAI_API_KEY (or pass --api-key / --openai-key)"
            )
            sys.exit(1)

    print(f"Parsing repository: {repo}")
    units = parse_repository(repo)
    print(f"Parsed {len(units)} units")

    results = convert_all_docstrings(
        units=units,
        repo_root=repo,
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
        dry_run=dry_run,
        module_filter=module_filter,
        force=force,
        agent=agent,
    )

    if dry_run or not results:
        return

    print(f"\nConverted {len(results)} docstrings. Re-indexing...")
    from vector_store import CodeVectorStore, SimpleTFIDFStore

    try:
        store = CodeVectorStore(persist_dir=index_dir)
    except Exception:
        store = SimpleTFIDFStore(persist_dir=index_dir)

    updated_units = parse_repository(repo)
    store.upsert(updated_units)
    print(f"Index stats: {json.dumps(store.stats(), indent=2)}")
    print("\nDone. Re-generate docs with: python generate_docs.py")


def docs(repo: str, port: int, build: bool, output_dir: str):
    try:
        import pdoc  # noqa: F401 — just check it's installed
    except ImportError:
        print("pdoc not installed. Run: pip install pdoc")
        sys.exit(1)

    repo_path = Path(repo).resolve()
    sys.path.insert(0, str(repo_path))

    if build:
        out = Path(output_dir).resolve()
        print(f"Building static docs → {out}")
        subprocess.run(
            [sys.executable, "-m", "pdoc", "--output-dir", str(out), "mlfinlab"],
            cwd=str(repo_path),
            check=True,
        )
        print(f"Done. Open {out}/mlfinlab.html")
    else:
        print(f"Serving live docs at http://localhost:{port}  (Ctrl-C to stop)")
        subprocess.run(
            [sys.executable, "-m", "pdoc", "--port", str(port), "mlfinlab"],
            cwd=str(repo_path),
        )


def main():
    parser = argparse.ArgumentParser(
        prog="ft",
        description="fruit-thrower CLI — search and generate fin-kit functions",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"MCP server SSE URL (default: {DEFAULT_SERVER})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # search
    p_search = sub.add_parser("search", help="Semantic search over indexed functions")
    p_search.add_argument("query", help="Natural language or code query")
    p_search.add_argument(
        "-n", type=int, default=5, help="Number of results (default: 5)"
    )
    p_search.add_argument(
        "--kind",
        choices=["function", "class", "method", "module"],
        help="Filter by unit kind",
    )
    p_search.add_argument("--module", help="Filter by module name prefix")

    # docs
    p_docs = sub.add_parser("docs", help="Serve or build pdoc HTML documentation")
    p_docs.add_argument(
        "--repo",
        default=str(Path(__file__).parent / "fin-kit"),
        help="Path to the repo (default: ./fin-kit)",
    )
    p_docs.add_argument(
        "--port", type=int, default=8080, help="Port for live server (default: 8080)"
    )
    p_docs.add_argument(
        "--build", action="store_true", help="Build static HTML instead of serving"
    )
    p_docs.add_argument(
        "--output-dir",
        default="./pdoc_out",
        help="Output dir for --build (default: ./pdoc_out)",
    )

    # generate
    p_gen = sub.add_parser(
        "generate", help="Generate a new function via Codex (or Claude/OpenAI)"
    )
    p_gen.add_argument(
        "request", help="Natural language description of the function to create"
    )
    p_gen.add_argument(
        "--module",
        dest="module_path",
        help="Target file path in repo, e.g. mlfinlab/portfolio/hrp.py",
    )
    p_gen.add_argument(
        "--agent",
        choices=["codex", "claude", "openai"],
        help="Override coding agent (default: $FRUIT_CODE_AGENT or codex)",
    )
    p_gen.add_argument("--model", help="Override model ID")

    # convert
    p_conv = sub.add_parser(
        "convert", help="Convert Sphinx-style docstrings to NumPy style"
    )
    p_conv.add_argument(
        "--repo",
        default=str(Path(__file__).parent / "fin-kit"),
        help="Path to repository root (default: ./fin-kit)",
    )
    p_conv.add_argument(
        "--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY)"
    )
    p_conv.add_argument(
        "--openai-key", help="OpenAI API key fallback (or set OPENAI_API_KEY)"
    )
    p_conv.add_argument(
        "--dry-run",
        action="store_true",
        help="List units that would be converted, without writing files",
    )
    p_conv.add_argument(
        "--module",
        dest="module_filter",
        help="Only convert units in modules containing this string",
    )
    p_conv.add_argument(
        "--force",
        action="store_true",
        help="Convert all docstrings, not just Sphinx-style ones",
    )
    p_conv.add_argument(
        "--index-dir",
        default=str(Path(__file__).parent / ".code_index"),
        help="Vector index directory for re-indexing after conversion",
    )
    p_conv.add_argument(
        "--agent",
        choices=["codex", "claude", "openai"],
        help="LLM backend (default: codex if installed, else openai/anthropic)",
    )

    args = parser.parse_args()

    if args.cmd == "search":
        search(args.server, args.query, args.n, args.kind, args.module)
    elif args.cmd == "generate":
        generate(args.server, args.request, args.module_path, args.agent, args.model)
    elif args.cmd == "docs":
        docs(args.repo, args.port, args.build, args.output_dir)
    elif args.cmd == "convert":
        convert(
            args.repo,
            args.api_key,
            args.openai_key,
            args.dry_run,
            args.module_filter,
            args.force,
            args.index_dir,
            args.agent,
        )


if __name__ == "__main__":
    main()
