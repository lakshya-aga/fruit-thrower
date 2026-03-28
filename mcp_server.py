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
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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


_REQUESTS_DIR = Path(os.environ.get("FRUIT_TOOL_REQUESTS_DIR", ".tool_builder/requests"))
_TRACE_LOG = Path(os.environ.get("FRUIT_TOOL_TRACE_LOG", ".tool_builder/trace.jsonl"))
_DEFAULT_BUILDER_PROMPT = (
    "Implement the requested fin-kit/library function and integrate discoverability in fruit-thrower. "
    "Use request spec JSON, add/update code, tests/docs, then commit to agent branch."
)


def _trace(event: str, request_id: str, **fields) -> None:
    _TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "request_id": request_id,
        **fields,
    }
    with _TRACE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _create_tool_request(arguments: dict) -> dict:
    tool_name = arguments.get("tool_name", "").strip()
    module_path = arguments.get("module_path", "").strip()
    summary = arguments.get("summary", "").strip()
    code = arguments.get("code", "")
    if not tool_name or not module_path or not summary or not code:
        raise ValueError("tool_name, module_path, summary, and code are required")

    request_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    _REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    request_file = _REQUESTS_DIR / f"{request_id}.json"
    payload = {
        "request_id": request_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "module_path": module_path,
        "summary": summary,
        "code": code,
        "tags": arguments.get("tags", []),
        "example": arguments.get("example", ""),
        "notes": arguments.get("notes", ""),
    }
    request_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _trace(
        "request_saved",
        request_id,
        tool_name=tool_name,
        module_path=module_path,
        request_file=str(request_file),
    )

    builder_cmd = os.environ.get("FRUIT_TOOL_BUILDER_CMD", "").strip()
    spawned = False
    spawn_error = None
    builder_report = None
    if builder_cmd:
        cmd = builder_cmd.format(request_file=str(request_file), request_id=request_id)
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(Path(__file__).resolve().parent),
                capture_output=True,
                text=True,
            )
            spawned = proc.returncode == 0
            _trace(
                "builder_spawned" if spawned else "builder_run_failed",
                request_id,
                command=cmd,
                returncode=proc.returncode,
                stdout=(proc.stdout or "")[-2000:],
                stderr=(proc.stderr or "")[-2000:],
            )
            rep = Path(".tool_builder") / "reports" / f"{request_id}.json"
            if rep.exists():
                try:
                    builder_report = json.loads(rep.read_text(encoding="utf-8"))
                except Exception:
                    builder_report = None
        except Exception as exc:  # pragma: no cover
            spawn_error = str(exc)
            _trace("builder_spawn_failed", request_id, command=cmd, error=spawn_error)
    else:
        _trace("builder_not_configured", request_id)

    return {
        "request_id": request_id,
        "request_file": str(request_file),
        "builder_spawned": spawned,
        "builder_command": builder_cmd or None,
        "spawn_error": spawn_error,
        "builder_report": builder_report,
    }


def load_store(index_dir: str):
    """Prefer semantic vector search; fall back to TF-IDF if Chroma isn't available."""
    try:
        return CodeVectorStore(persist_dir=index_dir)
    except Exception as exc:
        print(f"WARN: semantic store unavailable ({exc}); falling back to TF-IDF")
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
            types.Tool(
                name="request_tool_addition",
                description=(
                    "Submit a proposed new internal library function/tool implementation. "
                    "IMPORTANT: the request must be well-formed and paste-ready for the target repo "
                    "(production-quality code block, module path, and clear summary). "
                    "The MCP stores a formal request spec and can spawn an external builder agent "
                    "(configured by FRUIT_TOOL_BUILDER_CMD) to implement on agent branch + docs."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "module_path": {"type": "string", "description": "Target path, e.g. fin_kit/signals/new_signal.py"},
                        "summary": {"type": "string"},
                        "code": {"type": "string", "description": "Proposed Python code, paste-ready for the repo (imports, function body, and sane defaults)."},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "example": {"type": "string"},
                        "notes": {"type": "string"}
                    },
                    "required": ["tool_name", "module_path", "summary", "code"]
                }
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
                return [types.TextContent(type="text", text="No results found.")]

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

        elif name == "request_tool_addition":
            try:
                req = _create_tool_request(arguments)
            except ValueError as exc:
                rid = datetime.now(timezone.utc).strftime("invalid-%Y%m%dT%H%M%SZ")
                _trace("request_invalid", rid, error=str(exc), provided_keys=sorted(list(arguments.keys())))
                return [types.TextContent(type="text", text=f"Invalid request: {exc}")]

            _trace("request_accepted", req["request_id"], builder_spawned=req["builder_spawned"])
            status = "✅ builder agent spawn requested" if req["builder_spawned"] else "ℹ️ request queued"
            text = (
                f"# Tool addition request accepted\n\n"
                f"- Request ID: `{req['request_id']}`\n"
                f"- Request file: `{req['request_file']}`\n"
                f"- Status: {status}\n"
                f"- Builder command: `{req['builder_command'] or '(unset)'}`\n"
            )
            if req.get("builder_report"):
                br = req["builder_report"]
                text += f"- Viability: `{br.get('viable')}` ({br.get('reason')})\n"
                if br.get("pr_url"):
                    text += f"- Pull request: {br['pr_url']}\n"
            text += (
                f"\nSet `FRUIT_TOOL_BUILDER_CMD` to auto-run a builder agent using request spec.\n"
                f"Prompt hint: {_DEFAULT_BUILDER_PROMPT}\n"
            )
            if req["spawn_error"]:
                text += f"\nSpawn error: `{req['spawn_error']}`\n"
            return [types.TextContent(type="text", text=text)]

        elif name == "generate_function":
            request_text = arguments.get("request", "").strip()
            if not request_text:
                return [types.TextContent(type="text", text="Error: 'request' is required.")]

            module_path_arg: Optional[str] = arguments.get("module_path", "").strip() or None
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
                pass  # context is best-effort; proceed without it

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
                    target = repo_root_path / module_path_arg
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
                            pass
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
