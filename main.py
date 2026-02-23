#!/usr/bin/env python3
"""
main.py — CLI for parsing, indexing, and searching your Python repository.

Commands:
    index      Parse repo + build vector index
    search     Semantic search over indexed code
    generate   Generate docstrings for undocumented functions
    stats      Show index statistics

Examples:
    python main.py index --repo /path/to/mylib
    python main.py search "CUSUM filter for events"
    python main.py search "z score outlier detection" --kind function
    python main.py generate --repo /path/to/mylib --dry-run
    python main.py stats
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parser import parse_repository
from vector_store import SimpleTFIDFStore


def cmd_index(args):
    print(f"Parsing repository: {args.repo}")
    units = parse_repository(
        repo_root=args.repo,
        exclude_dirs=set(args.exclude) if args.exclude else None,
    )
    print(f"\nParsed {len(units)} total units")

    store = SimpleTFIDFStore(persist_dir=args.index_dir)
    store.upsert(units)
    print(f"\nIndex stats: {json.dumps(store.stats(), indent=2)}")


def cmd_search(args):
    store = SimpleTFIDFStore(persist_dir=args.index_dir)
    if store.count() == 0:
        print("Index is empty. Run: python main.py index --repo /path/to/repo")
        return

    results = store.search(
        query=args.query,
        n_results=args.n,
        kind_filter=args.kind,
        module_filter=args.module,
    )

    if not results:
        print("No results found.")
        return

    print(f"\nTop {len(results)} results for: '{args.query}'\n{'='*60}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['kind'].upper()}: {r['module']}.{r['name']}")
        print(f"    Score : {r['score']:.4f}")
        print(f"    File  : {r['file_path']}:{r['line_start']}")
        print(f"    Sig   : {r['signature'].splitlines()[0]}")
        if r["docstring"]:
            first_line = r["docstring"].strip().splitlines()[0]
            print(f"    Doc   : {first_line[:100]}")
        print(f"    ID    : {r['id']}")


def cmd_generate(args):
    print(f"Parsing repository: {args.repo}")
    units = parse_repository(args.repo)
    missing = [u for u in units if not u.docstring and u.kind in ("function", "method", "class")]
    print(f"\nFound {len(missing)} units without docstrings")

    if args.dry_run:
        print("\nDRY RUN — showing units that would be documented:")
        for u in missing:
            print(f"  {u.kind:8s} {u.module}.{u.name} (line {u.line_start})")
        return

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY env var or pass --api-key")
        sys.exit(1)

    try:
        from docstring_generator import generate_missing_docstrings
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    results = generate_missing_docstrings(
        units=missing,
        repo_root=args.repo,
        api_key=api_key,
        dry_run=False,
    )
    print(f"\nGenerated {len(results)} docstrings.")
    print("Re-indexing with updated docstrings...")
    store = SimpleTFIDFStore(persist_dir=args.index_dir)
    updated_units = parse_repository(args.repo)
    store.upsert(updated_units)


def cmd_stats(args):
    store = SimpleTFIDFStore(persist_dir=args.index_dir)
    if store.count() == 0:
        print("Index is empty.")
        return
    stats = store.stats()
    print(json.dumps(stats, indent=2))
    print(f"\nIndex location: {args.index_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Code RAG — Parse, index, and search your Python repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--index-dir",
        default="./.code_index",
        help="Directory for the vector index (default: ./.code_index)"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Parse repo and build index")
    p_index.add_argument("--repo", required=True, help="Path to repository root")
    p_index.add_argument("--exclude", nargs="*", help="Directory names to exclude")

    # search
    p_search = sub.add_parser("search", help="Search indexed code")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-n", type=int, default=5, help="Number of results")
    p_search.add_argument("--kind", choices=["function", "class", "method", "module"])
    p_search.add_argument("--module", help="Filter by module prefix")

    # generate
    p_gen = sub.add_parser("generate", help="Generate missing docstrings")
    p_gen.add_argument("--repo", required=True, help="Path to repository root")
    p_gen.add_argument("--api-key", help="Anthropic API key")
    p_gen.add_argument("--dry-run", action="store_true", help="List units without writing")

    # stats
    sub.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()
    {"index": cmd_index, "search": cmd_search, "generate": cmd_generate, "stats": cmd_stats}[args.command](args)


if __name__ == "__main__":
    main()
