#!/bin/bash
# Start the fruit-thrower MCP server (SSE transport, port 8000)
# Usage: ./start.sh [--port 8000] [--repo ./fin-kit]

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/mcp_server.py" \
  --repo "${FRUIT_REPO:-$DIR/fin-kit}" \
  --index-dir "${FRUIT_INDEX:-$DIR/.code_index}" \
  --transport sse \
  --port "${FRUIT_PORT:-8000}" \
  "$@"
