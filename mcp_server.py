"""
mcp_server.py — MCP server exposing repository code search to AI agents.

Run this server and connect it to Claude (or any MCP-compatible agent) to
give the agent semantic search over your internal Python library.

Usage:
    python mcp_server.py --index-dir ./.code_index --repo /path/to/repo

MCP tools exposed:
    • search_code         — Semantic search over functions/classes
    • get_unit_source     — Fetch full source of a specific unit
    • list_modules        — List all indexed modules
    • get_module_summary  — Get all units in a given module
    • index_repository    — Trigger re-indexing of the repo
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

from parser import parse_repository, ParsedUnit
from vector_store import SimpleTFIDFStore


def load_store(index_dir: str) -> SimpleTFIDFStore:
    return SimpleTFIDFStore(persist_dir=index_dir)


def build_server(index_dir: str, repo_root: str) -> "Server":
    """Build and return the configured MCP server."""
    store = load_store(index_dir)
    server = Server("code-rag")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_code",
                description=(
                    "Semantic search over indexed Python functions, classes, and modules. "
                    "Use this to find relevant internal library code before writing new code. "
                    "Returns function signatures, docstrings, and source snippets."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language or code query, e.g. 'CUSUM filter for time series events'"
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["function", "class", "method", "module"],
                            "description": "Filter by unit kind (optional)"
                        },
                        "module": {
                            "type": "string",
                            "description": "Filter by module name prefix, e.g. 'filters' (optional)"
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="get_unit_source",
                description=(
                    "Retrieve the full source code of a specific function or class by its ID. "
                    "Use the 'id' field from search_code results."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "unit_id": {
                            "type": "string",
                            "description": "Unit ID from search_code results"
                        }
                    },
                    "required": ["unit_id"]
                }
            ),
            types.Tool(
                name="list_modules",
                description="List all indexed Python modules in the repository.",
                inputSchema={"type": "object", "properties": {}}
            ),
            types.Tool(
                name="get_module_summary",
                description="Get all functions and classes defined in a specific module.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "module": {
                            "type": "string",
                            "description": "Dotted module name, e.g. 'filters.cusum'"
                        }
                    },
                    "required": ["module"]
                }
            ),
            types.Tool(
                name="index_repository",
                description=(
                    "Re-index the repository to pick up new or changed files. "
                    "Run this after adding new code to the library."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "exclude_dirs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Directories to exclude (optional)"
                        }
                    }
                }
            ),
            types.Tool(
                name="get_index_stats",
                description="Return statistics about the current code index.",
                inputSchema={"type": "object", "properties": {}}
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name == "search_code":
            results = store.search(
                query=arguments["query"],
                n_results=arguments.get("n_results", 5),
                kind_filter=arguments.get("kind"),
                module_filter=arguments.get("module"),
            )
            if not results:
                return [types.TextContent(type="text", text="No results found.")]

            output = []
            for r in results:
                block = (
                    f"## {r['kind'].upper()}: {r['module']}.{r['name']}\n"
                    f"**Score**: {r['score']:.3f}  |  "
                    f"**File**: {r['file_path']}:{r['line_start']}\n\n"
                    f"**Signature**:\n```python\n{r['signature']}\n```\n\n"
                )
                if r["docstring"]:
                    block += f"**Docstring**:\n{r['docstring'][:500]}\n\n"
                block += f"**Source preview**:\n```python\n{r['source'][:800]}\n```\n"
                block += f"\n**Unit ID**: `{r['id']}`\n---\n"
                output.append(block)

            return [types.TextContent(type="text", text="\n".join(output))]

        elif name == "get_unit_source":
            unit_id = arguments["unit_id"]
            all_records = store._records
            match = next((r for r in all_records if r["id"] == unit_id), None)
            if not match:
                return [types.TextContent(type="text", text=f"Unit '{unit_id}' not found.")]
            text = (
                f"# {match['kind'].upper()}: {match['module']}.{match['name']}\n"
                f"File: {match['file_path']} (lines {match['line_start']}–{match['line_end']})\n\n"
                f"```python\n{match['source']}\n```"
            )
            return [types.TextContent(type="text", text=text)]

        elif name == "list_modules":
            modules = sorted(set(r["module"] for r in store._records))
            text = "## Indexed Modules\n" + "\n".join(f"- `{m}`" for m in modules)
            return [types.TextContent(type="text", text=text)]

        elif name == "get_module_summary":
            module = arguments["module"]
            units = [r for r in store._records if r["module"] == module]
            if not units:
                return [types.TextContent(type="text", text=f"No units found in module '{module}'.")]
            lines = [f"## Module: {module}\n"]
            for u in sorted(units, key=lambda x: x["line_start"]):
                lines.append(f"### {u['kind']}: {u['name']} (line {u['line_start']})")
                lines.append(f"```python\n{u['signature']}\n```")
                if u["docstring"]:
                    lines.append(u["docstring"][:300])
                lines.append("")
            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "index_repository":
            exclude = set(arguments.get("exclude_dirs", []))
            units = parse_repository(repo_root, exclude_dirs=exclude or None)
            store.upsert(units)
            return [types.TextContent(
                type="text",
                text=f"Indexed {len(units)} units from {repo_root}.\n{json.dumps(store.stats(), indent=2)}"
            )]

        elif name == "get_index_stats":
            stats = store.stats()
            return [types.TextContent(type="text", text=f"```json\n{json.dumps(stats, indent=2)}\n```")]

        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


def main():
    parser = argparse.ArgumentParser(description="Code RAG MCP Server")
    parser.add_argument("--index-dir", default="./.code_index", help="Path to vector index directory")
    parser.add_argument("--repo", required=True, help="Path to repository root")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to listen on")
    parser.add_argument(
        "--transport",
        choices=["streamable", "sse"],
        default="streamable",
        help="MCP transport to expose (default: streamable)",
    )
    parser.add_argument(
        "--mount-path",
        default="/mcp",
        help="Path to mount streamable HTTP transport (default: /mcp)",
    )
    args = parser.parse_args()

    if not _MCP_AVAILABLE:
        print("ERROR: pip install mcp")
        sys.exit(1)

    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    import uvicorn

    server = build_server(args.index_dir, args.repo)

    if args.transport == "sse":
        from mcp.server.sse import SseServerTransport

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await server.run(streams[0], streams[1], server.create_initialization_options())

        app = Starlette(routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ])
        print(f"Code RAG MCP (SSE) running at http://{args.host}:{args.port}/sse")

    else:
        import anyio
        from contextlib import asynccontextmanager
        from mcp.server.streamable_http import StreamableHTTPServerTransport

        streamable = StreamableHTTPServerTransport(mcp_session_id=None)

        @asynccontextmanager
        async def lifespan(app):
            async with streamable.connect() as streams:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(server.run, streams[0], streams[1], server.create_initialization_options())
                    yield
                    tg.cancel_scope.cancel()

        app = Starlette(
            routes=[Mount(args.mount_path, app=streamable.handle_request)],
            lifespan=lifespan,
        )
        print(f"Code RAG MCP (streamable HTTP) running at http://{args.host}:{args.port}{args.mount_path}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
