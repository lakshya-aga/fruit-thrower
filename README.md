# Code RAG + MCP Server

Parse, vectorise, and expose your internal Python library to AI agents via MCP.

## Architecture

```
Your Repo (.py files)
       │
       ▼
  parser.py          ← AST-based extraction of functions/classes/modules
       │
       ▼
  vector_store.py    ← TF-IDF (offline) or ChromaDB + Anthropic embeddings
       │
       ▼
  mcp_server.py      ← MCP server with 6 tools
       │
       ▼
Claude (agent)       ← Searches your library before writing new code
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install scikit-learn mcp
# For semantic embeddings (recommended):
pip install anthropic chromadb
```

### 2. Index your repository

```bash
python main.py index --repo /path/to/your/lib
```

This parses all `.py` files and builds a searchable index at `./.code_index/`.

### 3. Search from the CLI

```bash
python main.py search "CUSUM filter for financial time series"
python main.py search "z score outlier detection" --kind function
python main.py search "structural break" --module filters
```

### 4. Generate missing docstrings (optional)

```bash
# Preview which functions are missing docstrings
python main.py generate --repo /path/to/your/lib --dry-run

# Generate and patch source files
export ANTHROPIC_API_KEY=sk-ant-...
python main.py generate --repo /path/to/your/lib
```

### 5. Connect to Claude via MCP

```bash
python mcp_server.py --repo /path/to/your/lib --index-dir ./.code_index
```

## MCP Tools (what Claude gets)

| Tool | Description |
|------|-------------|
| `search_code` | Semantic search → returns signatures, docstrings, source |
| `get_unit_source` | Fetch full source of a unit by ID |
| `list_modules` | List all indexed modules |
| `get_module_summary` | All functions/classes in a given module |
| `index_repository` | Trigger re-indexing (after you add new code) |
| `get_index_stats` | Index statistics by kind/module |

---

## How It Works

### Parsing (`parser.py`)

Uses Python's `ast` module to extract:
- **Module-level docstrings** — context for what the file does
- **Class definitions** — with class docstring and base classes
- **Functions & Methods** — signature, docstring, full source, line numbers
- **Qualified names** — e.g. `MyClass.my_method` for methods

Each unit gets a stable `id` (SHA-256 of `file_path::name`) for idempotent upserts.

### Embedding text (`ParsedUnit.to_embed_text`)

Each unit is represented as:
```
[FUNCTION] filters.cusum_filter

Signature: def cusum_filter(raw_time_series, threshold, time_stamps=True):

Docstring:
The Symmetric CUSUM Filter...

Source:
def cusum_filter(...):
    ...
```

This rich text captures intent, interface, and implementation for search.

### Vector Store

Two backends available:

| Backend | When to use |
|---------|------------|
| `SimpleTFIDFStore` | Offline, no API key needed, fast setup |
| `CodeVectorStore` | Production, semantic search with Anthropic embeddings |

Switch by changing the import in `mcp_server.py`.

### Docstring Generation

When `generate` is run, units without docstrings are sent to Claude with the
full source and patched back into the file using AST line numbers. The format
follows Google-style conventions matching your existing codebase.

---

## Project Structure

```
code_rag/
├── parser.py              # AST parser → ParsedUnit dataclass
├── vector_store.py        # TF-IDF + ChromaDB vector stores
├── docstring_generator.py # Claude-powered docstring generation
├── mcp_server.py          # MCP server (stdio transport)
├── main.py                # CLI: index / search / generate / stats
└── README.md              # This file
```

---

## Upgrading to Semantic Embeddings

Replace `SimpleTFIDFStore` with `CodeVectorStore` in `mcp_server.py`:

```python
from vector_store import CodeVectorStore

store = CodeVectorStore(persist_dir=index_dir)
```

This uses Anthropic's `voyage-code-3` model, specifically trained for code
retrieval. Cosine similarity on these embeddings dramatically outperforms
TF-IDF for natural language queries like "filter events based on volatility".

---

## Workflow for Your Agent

When Claude has this MCP server connected, a typical coding session looks like:

1. User asks Claude to write code using the internal library
2. Claude calls `search_code("relevant concept")` before writing
3. Claude finds `cusum_filter`, reads its signature and docstring
4. Claude reuses the existing function rather than reimplementing it
5. New code is idiomatic and consistent with the library's patterns

---

After Setup run:
```
python test_client.py
```

---

## Docker Deployment

The server ships as a Docker image published to GitHub Container Registry (GHCR) via GitHub Actions on every push to `main`.

The image includes the **Codex CLI** (`@openai/codex`) so the `codex` agent backend works out of the box. Three auth options for Codex:

| Option | How |
|--------|-----|
| **1 — API key** | Pass `OPENAI_API_KEY` as an env var |
| **2 — Mount host auth** | Log in once on your Mac, mount `~/.codex` into the container (recommended) |
| **3 — Bake `auth.json`** | Mount a specific `/path/to/auth.json:/root/.codex/auth.json` |

### Option 2 — mount host Codex auth (recommended)

```bash
# One-time login on your Mac (opens browser)
codex auth login

# Then run via docker compose
docker compose up -d
```

`docker-compose.yml` mounts `~/.codex` read-only and sets `FRUIT_CODE_AGENT=codex` automatically.

### Option 1 — API key

```bash
docker run -d \
  --name fruit-thrower \
  -p 8090:8090 \
  -v $(pwd)/.code_index-fin-kit:/app/.code_index-fin-kit \
  -v $(pwd)/.tool_builder:/app/.tool_builder \
  -e FRUIT_CODE_AGENT=codex \
  -e OPENAI_API_KEY=sk-... \
  ghcr.io/lakshya-aga/fruit-thrower:latest
```

The MCP endpoint is available at `http://localhost:8090/mcp`.

### Build locally

```bash
git clone --recurse-submodules https://github.com/lakshya-aga/fruit-thrower
cd fruit-thrower
docker build -t fruit-thrower .
docker run -d -p 8090:8090 fruit-thrower
```

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key for semantic embeddings + Claude agent | — |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI agent backend) | — |
| `FRUIT_CODE_AGENT` | Coding agent backend: `codex`, `claude`, `openai`, `mock` | `codex` |
| `FRUIT_CLAUDE_MODEL` | Claude model override for code generation | `claude-sonnet-4-6` |
| `FRUIT_OPENAI_MODEL` | OpenAI model override | `gpt-4-turbo` |
| `FRUIT_TOOL_BUILDER_CMD` | Shell command to run the agentic builder on new requests | — |

### GitHub Actions (CI/CD)

Pushing to `main` (or `deploy`) triggers `.github/workflows/build-container.yml`, which:
1. Checks out the repo with submodules
2. Builds the Docker image
3. Pushes `ghcr.io/<owner>/fruit-thrower:latest` and `ghcr.io/<owner>/fruit-thrower:sha-<commit>` to GHCR

A separate `deploy-ec2.yml` workflow handles EC2 deployment via SSH after a successful build.

### Transport modes

The container defaults to **Streamable HTTP** (`/mcp`). To use SSE instead:

```bash
docker run -d -p 8090:8090 fruit-thrower \
  python mcp_server.py --transport sse --host 0.0.0.0 --port 8090 \
  --repo /app/fin-kit --index-dir /app/.code_index-fin-kit
```

SSE endpoint: `http://localhost:8090/sse`

### Connect Claude Desktop to the container

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