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