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
    • get_index_stats     — Stats on the current index
    • generate_function   — Generate a new function via AI agent and index it
"""
import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from code_agent import generate_function as _agent_generate

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
from vector_store import CodeVectorStore, SimpleTFIDFStore


def load_store(index_dir: str):
    """Prefer semantic vector search; fall back to TF-IDF if Chroma isn't available."""
    try:
        return CodeVectorStore(persist_dir=index_dir)
    except Exception as exc:
        logger.warning("Semantic store unavailable (%s); falling back to TF-IDF", exc)
        return SimpleTFIDFStore(persist_dir=index_dir)


def build_server(index_dir: str, repo_root: str) -> "Server":
    """Build and return the configured MCP server."""
    store = load_store(index_dir)
    # If the index is empty at startup, search_code will silently return
    # "No results found." for every query and the agent has no signal that
    # something's wrong. Log a loud warning + try to rebuild on the fly
    # so the failure is recoverable instead of silently broken.
    try:
        unit_count = store.count() if hasattr(store, "count") else len(getattr(store, "_records", []))
    except Exception as exc:
        logger.warning("could not introspect index size: %s", exc)
        unit_count = -1
    if unit_count == 0:
        logger.error(
            "code index at %s is EMPTY — every search_code call will return "
            "'No results found.' Attempting to rebuild from %s now.",
            index_dir, repo_root,
        )
        try:
            units = parse_repository(repo_root)
            n = store.upsert(units) if hasattr(store, "upsert") else 0
            logger.warning("auto-rebuild indexed %d units from %s", n, repo_root)
        except Exception:
            logger.exception("auto-rebuild failed; index remains empty")
    elif unit_count > 0:
        logger.info("code index ready: %d units at %s", unit_count, index_dir)
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
            types.Tool(
                name="generate_function",
                description=(
                    "Generate a new Python function from a natural language request using an AI "
                    "coding agent (Claude or OpenAI, configurable via FRUIT_CODE_AGENT env var). "
                    "The agent is given context from the existing indexed codebase so the generated "
                    "function matches library conventions. After generation the code is written to "
                    "the repo and the index is updated, making the new function immediately "
                    "searchable via search_code."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "string",
                            "description": (
                                "Natural language description of the function to create. "
                                "Be as specific as possible: inputs, outputs, algorithm, "
                                "edge-case handling, and any relevant domain constraints."
                            ),
                        },
                        "module_path": {
                            "type": "string",
                            "description": (
                                "Relative path inside the repo where the function should be "
                                "written, e.g. 'mlfinlab/filters/filters.py'. "
                                "If the file already exists the new code is appended. "
                                "If omitted, the function is written to "
                                "'.tool_builder/generated/<uuid>.py' and indexed from there."
                            ),
                        },
                        "agent": {
                            "type": "string",
                            "enum": ["codex", "claude", "openai"],
                            "description": (
                                "Coding agent to use. Overrides FRUIT_CODE_AGENT env var. "
                                "Defaults to 'codex' (Codex CLI binary)."
                            ),
                        },
                        "model": {
                            "type": "string",
                            "description": (
                                "Model ID override, e.g. 'claude-opus-4-6' or 'gpt-4-turbo'. "
                                "Defaults to FRUIT_CLAUDE_MODEL / FRUIT_OPENAI_MODEL env vars."
                            ),
                        },
                        "context_n": {
                            "type": "integer",
                            "description": (
                                "Number of related existing functions to retrieve from the "
                                "index and pass as context to the agent (default: 3)."
                            ),
                            "default": 3,
                        },
                    },
                    "required": ["request"],
                },
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
                # Diagnose why we returned empty so the calling agent can
                # adjust its strategy (or the operator can spot a broken
                # deployment) instead of assuming "this concept doesn't
                # exist in the library".
                try:
                    total = store.count() if hasattr(store, "count") else len(getattr(store, "_records", []))
                except Exception:
                    total = -1
                if total == 0:
                    msg = (
                        "No results — the code index is EMPTY. "
                        "The deployment is broken; ask the operator to rebuild the index."
                    )
                else:
                    msg = (
                        f"No results found for query='{arguments['query']}'. "
                        f"Index has {total} units; try a shorter / more focused query "
                        "(e.g. one concept like 'CUSUM filter' or 'rolling z-score'), "
                        "or call list_modules to see what's available."
                    )
                return [types.TextContent(type="text", text=msg)]

            output = []
            for r in results:
                block = (
                    f"## {r['kind'].upper()}: {r['module']}.{r['name']}\n"
                    f"**Score**: {r['score']:.3f}  |  "
                    f"**File**: `{r['file_path']}:{r['line_start']}`\n\n"
                    f"**Signature**:\n```python\n{r['signature']}\n```\n\n"
                )
                if r["docstring"]:
                    block += f"**Docstring**:\n{r['docstring']}\n\n"
                block += f"**Unit ID**: `{r['id']}`\n---\n"
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

        elif name == "generate_function":
            request_text = arguments.get("request", "").strip()
            if not request_text:
                return [types.TextContent(type="text", text="Error: 'request' is required.")]

            # Input length guard — prevent abuse via extremely long prompts
            _MAX_REQUEST_LEN = 10_000
            if len(request_text) > _MAX_REQUEST_LEN:
                return [types.TextContent(
                    type="text",
                    text=f"Error: request too long ({len(request_text)} chars, max {_MAX_REQUEST_LEN}).",
                )]

            module_path_arg: Optional[str] = arguments.get("module_path", "").strip() or None
            # Validate module_path to prevent path traversal
            if module_path_arg and (".." in module_path_arg or module_path_arg.startswith("/")):
                return [types.TextContent(type="text", text="Error: module_path must be a relative path without '..'")]

            agent_arg: Optional[str] = arguments.get("agent") or None
            model_arg: Optional[str] = arguments.get("model") or None
            context_n: int = int(arguments.get("context_n", 3))

            # --- 1. Fetch context from the existing index ---
            context_str = ""
            try:
                ctx_results = store.search(query=request_text, n_results=context_n, kind_filter="function")
                if ctx_results:
                    snippets = []
                    for r in ctx_results:
                        snippets.append(
                            f"### {r['module']}.{r['name']}\n"
                            f"```python\n{r['source'][:600]}\n```"
                        )
                    context_str = "\n\n".join(snippets)
            except Exception:
                logger.debug("Context retrieval failed; proceeding without context")

            # --- 2. Call the coding agent ---
            try:
                result = _agent_generate(
                    request=request_text,
                    context=context_str,
                    agent=agent_arg,
                    model_override=model_arg,
                    repo_root=repo_root,
                    module_path=module_path_arg,
                )
            except (ValueError, ImportError, FileNotFoundError, RuntimeError) as exc:
                return [types.TextContent(type="text", text=f"Agent error: {exc}")]

            code = result["code"]
            agent_used = result["agent_used"]
            model_used = result["model_used"]
            auth_method = result.get("auth_method", "unknown")
            codex_wrote_files = result.get("wrote_files", False)
            codex_changed_files = result.get("changed_files", [])

            # --- 3. Write to repo ---
            # Codex writes files itself when not sandboxed.
            # If sandboxed (wrote_files=False), fall back to writing extracted code ourselves.
            repo_root_path = Path(repo_root).resolve()
            write_mode = None
            target = None

            needs_write = agent_used != "codex" or not codex_wrote_files
            if needs_write:
                if module_path_arg:
                    target = (repo_root_path / module_path_arg).resolve()
                    # Ensure target is within repo root
                    if not str(target).startswith(str(repo_root_path)):
                        return [types.TextContent(type="text", text="Error: module_path escapes repository root.")]
                else:
                    generated_dir = Path(".tool_builder") / "generated"
                    generated_dir.mkdir(parents=True, exist_ok=True)
                    target = generated_dir / f"{uuid.uuid4().hex[:8]}.py"

                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    existing = target.read_text(encoding="utf-8")
                    target.write_text(existing.rstrip() + "\n\n\n" + code + "\n", encoding="utf-8")
                    write_mode = "appended"
                else:
                    target.write_text(code + "\n", encoding="utf-8")
                    write_mode = "created"

            # --- 4. Re-index ---
            index_errors = []
            indexed_count = 0
            try:
                if agent_used == "codex" and codex_wrote_files:
                    # Re-index only the files Codex actually changed
                    from parser import parse_file
                    new_units = []
                    for rel_path in codex_changed_files:
                        abs_path = str(Path(repo_root).resolve() / rel_path)
                        try:
                            new_units += parse_file(abs_path, repo_root=repo_root)
                        except Exception:
                            logger.warning("Failed to parse %s during re-index", rel_path, exc_info=True)
                    if new_units:
                        store.upsert(new_units)
                    indexed_count = len(new_units)
                elif target is not None:
                    from parser import parse_file
                    new_units = parse_file(str(target), repo_root=str(repo_root_path))
                    store.upsert(new_units)
                    indexed_count = len(new_units)
            except Exception as exc:
                index_errors.append(str(exc))

            # --- 5. Build response ---
            text = (
                f"# Generated function\n\n"
                f"- **Agent**: `{agent_used}` / `{model_used}`\n"
                f"- **Auth**: `{auth_method}`\n"
            )
            if agent_used == "codex" and codex_wrote_files:
                text += f"- **Codex wrote files directly**: {', '.join(f'`{f}`' for f in codex_changed_files)}\n"
                text += f"- **Indexed**: {indexed_count} unit(s) added/updated\n"
            else:
                text += f"- **File**: `{target}` ({write_mode})\n"
                text += f"- **Indexed**: {indexed_count} unit(s) added/updated\n"
            if index_errors:
                text += f"- **Index warning**: {'; '.join(index_errors)}\n"

            # When Codex wrote files directly, `code` is its prose output.
            # Read the actual written file so the block contains real Python.
            display_code = code
            if agent_used == "codex" and codex_wrote_files:
                read_path = None
                if module_path_arg:
                    read_path = repo_root_path / module_path_arg
                elif codex_changed_files:
                    for rel in codex_changed_files:
                        candidate = repo_root_path / rel
                        if candidate.exists() and candidate.suffix == ".py":
                            read_path = candidate
                            break
                if read_path and read_path.exists():
                    display_code = read_path.read_text(encoding="utf-8")

            text += f"\n```python\n{display_code}\n```\n"
            return [types.TextContent(type="text", text=text)]

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
