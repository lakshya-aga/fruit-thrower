# fruit-thrower

Code RAG + MCP Server — parse, index, and expose your internal Python library to AI agents via MCP. Includes an agentic `generate_function` tool that writes new library functions on demand using Codex, Claude, or OpenAI.

## Architecture

```
fin-kit (.py files)
       │
       ▼
  parser.py          ← AST extraction of functions/classes/modules
       │
       ▼
  vector_store.py    ← TF-IDF (offline) or ChromaDB + Anthropic embeddings
       │
       ▼
  mcp_server.py      ← MCP server (8 tools)
       │
       ▼
Claude / Codex       ← Searches library, generates missing functions
```

---

## Installation

The recommended way to run fruit-thrower is via Docker. The image is published to GHCR on every push to `main` and includes the full fin-kit library, Codex CLI, and all dependencies.

### Prerequisites

- Docker
- Codex installed and authenticated on your host machine (for code generation)

### 1. Authenticate Codex (one-time)

```bash
codex auth login
# Opens browser → sign in with OpenAI → saves to ~/.codex/auth.json
```

### 2. Pull and run

```bash
docker pull ghcr.io/lakshya-aga/fruit-thrower:latest

docker run -d \
  --name fruit-thrower \
  -p 8090:8090 \
  -v ~/.codex:/root/.codex:ro \
  -v $(pwd)/.code_index-fin-kit:/app/.code_index-fin-kit \
  -v $(pwd)/.tool_builder:/app/.tool_builder \
  -e FRUIT_CODE_AGENT=codex \
  ghcr.io/lakshya-aga/fruit-thrower:latest
```

Or with docker compose (mounts `~/.codex` automatically):

```bash
curl -O https://raw.githubusercontent.com/lakshya-aga/fruit-thrower/main/docker-compose.yml
docker compose up -d
```

### 3. Index the library

On first run the index is empty. Trigger indexing once:

```python
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

async def index():
    async with streamablehttp_client("http://localhost:8090/mcp/") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.call_tool("index_repository", {})
            print(result.content[0].text)

asyncio.run(index())
```

### 4. Connect Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fruit-thrower": {
      "url": "http://localhost:8090/mcp"
    }
  }
}
```

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_code` | Semantic search → signatures, docstrings, source snippets |
| `get_unit_source` | Fetch full source of a unit by ID |
| `list_modules` | List all indexed modules |
| `get_module_summary` | All functions/classes in a given module |
| `index_repository` | Re-index after adding new code |
| `get_index_stats` | Index statistics by kind/module |
| `request_tool_addition` | Submit a proposed function for the agentic builder |
| `generate_function` | Generate a new function from natural language via Codex/Claude/OpenAI |

---

## Usage

### Search

```python
result = await s.call_tool("search_code", {
    "query": "CUSUM filter for financial time series",
    "n_results": 5
})
```

### Generate a function

```python
result = await s.call_tool("generate_function", {
    "request": "a function called ewma_vol that takes a pd.Series of returns and a span int and returns rolling exponential weighted volatility annualised",
    "module_path": "mlfinlab/features/volatility.py"
})
```

The agent retrieves related functions from the index as context, writes the code to the repo, and re-indexes so the new function is immediately searchable.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FRUIT_CODE_AGENT` | Agent backend: `codex`, `claude`, `openai`, `mock` | `codex` |
| `FRUIT_CODEX_BIN` | Path to Codex binary | `/usr/bin/codex` |
| `ANTHROPIC_API_KEY` | Anthropic API key (semantic embeddings + Claude agent) | — |
| `OPENAI_API_KEY` | OpenAI API key (alternative to Codex auth mount) | — |
| `FRUIT_CLAUDE_MODEL` | Claude model for code generation | `claude-sonnet-4-6` |
| `FRUIT_OPENAI_MODEL` | OpenAI model for code generation | `gpt-4o` |
| `FRUIT_TOOL_BUILDER_CMD` | Shell command for the agentic builder pipeline | — |

---

## Codex Auth Options

| Option | How |
|--------|-----|
| **Mount host auth (recommended)** | `codex auth login` on host, mount `~/.codex:/root/.codex:ro` |
| **API key** | Pass `OPENAI_API_KEY` as env var |
| **Bake auth.json** | Mount a specific file: `-v /path/to/auth.json:/root/.codex/auth.json` |

---

## CI/CD

Pushing to `main` triggers `.github/workflows/build-container.yml`:
1. Checks out repo with submodules (`fin-kit` baked into image)
2. Builds multi-platform image (`linux/amd64` + `linux/arm64`)
3. Pushes `ghcr.io/lakshya-aga/fruit-thrower:latest` and `ghcr.io/lakshya-aga/fruit-thrower:sha-<commit>` to GHCR

A separate `deploy-ec2.yml` workflow deploys to EC2 via SSH after a successful build.
