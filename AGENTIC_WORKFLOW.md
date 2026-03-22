# Agentic Tool Addition Workflow (fruit-thrower + fin-kit)

## Flow
1. User calls MCP tool `request_tool_addition`.
2. Request JSON is stored in `.tool_builder/requests/<id>.json`.
3. Optional builder command (`FRUIT_TOOL_BUILDER_CMD`) is spawned.
4. Every step is logged to `.tool_builder/trace.jsonl` (override with `FRUIT_TOOL_TRACE_LOG`).
5. Builder script (`scripts/agent_pipeline.py`) triages viability.
5. If viable, implementation is committed to `agent` branch.
6. Human reviews PR and merges.
7. Post-merge: pull + reindex to update MCP responses.

## Configure auto-builder

```bash
export FRUIT_TOOL_BUILDER_CMD='python scripts/agent_pipeline.py --request {request_file} --push'
```

## Post-merge update

```bash
git checkout main && git pull
python main.py --index-dir ./.code_index-fin-kit index --repo /path/to/fin-kit
sudo systemctl restart fruit-thrower
```
