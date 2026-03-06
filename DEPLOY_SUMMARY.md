# Deployment Summary (deploy branch)

This branch documents and includes the changes made to deploy `fruit-thrower` publicly for MCP clients (including OpenAI Agents SDK) with `fin-kit` indexed.

## Code Changes

### 1) `mcp_server.py` transport upgrade
- Added support for **two MCP transport modes**:
  - `streamable` (default) — for modern MCP clients / OpenAI Agents SDK
  - `sse` — legacy compatibility mode
- Added new CLI flags:
  - `--transport` (`streamable` or `sse`)
  - `--mount-path` (default `/mcp`)
- Implemented streamable HTTP server path with `mcp.server.streamable_http.StreamableHTTPServerTransport`.

Result: tools are now discoverable via `POST /mcp/` using standard MCP initialize + tools/list flow.

### 2) Git ignore hardening
- Updated `.gitignore` to avoid committing local deployment/runtime artifacts:
  - `.venv/`
  - `.code_index*/`
  - `__pycache__/`

## Deployment Work Performed on Host

The following were configured on the server (not fully represented as repo files):

1. **Service + runtime**
   - Created Python virtual env in `fruit-thrower/.venv`
   - Installed runtime deps (`mcp`, `starlette`, `uvicorn`, `scikit-learn`, etc.)
   - Indexed `fin-kit` into `fruit-thrower/.code_index-fin-kit`

2. **Systemd service**
   - Added `/etc/systemd/system/fruit-thrower.service`
   - Service runs:
     - repo: `/home/ubuntu/.openclaw/workspace/fruit-thrower`
     - index: `/home/ubuntu/.openclaw/workspace/fruit-thrower/.code_index-fin-kit`
     - bind: `127.0.0.1:8090`

3. **Nginx reverse proxy + auth**
   - Added `/etc/nginx/conf.d/fruit-thrower.conf`
   - Reverse proxy to local app
   - Authentication supports:
     - `X-API-Key: <token>`
     - `Authorization: Bearer <token>`

4. **Public endpoints**
   - HTTP MCP endpoint: `http://<host>/mcp/`
   - HTTPS quick tunnel endpoint (Cloudflare quick tunnel): `https://<random>.trycloudflare.com/mcp/`

## Verified Behavior

- `initialize` returns MCP server capabilities.
- `tools/list` returns all expected tools:
  - `search_code`
  - `get_unit_source`
  - `list_modules`
  - `get_module_summary`
  - `index_repository`
  - `get_index_stats`

## Notes

- Cloudflare **quick tunnel URL is ephemeral** and may change on restart.
- For production stability, use a named tunnel or dedicated domain + TLS.
