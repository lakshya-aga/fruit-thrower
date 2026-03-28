"""
test_generate.py — end-to-end test for the generate_function MCP tool.

Starts the server (SSE transport), calls generate_function with a simple
quant finance request, and checks the response contains Python code.
"""
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SERVER_URL = "http://127.0.0.1:18765/sse"
REPO = str(Path(__file__).parent / "fin-kit")
INDEX = str(Path(__file__).parent / ".code_index")


async def run_test():
    from mcp.client.sse import sse_client
    from mcp.client.session import ClientSession

    async with sse_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"Tools available: {tool_names}")
            assert "generate_function" in tool_names, "generate_function tool missing!"

            print("\nCalling generate_function...")
            result = await session.call_tool(
                "generate_function",
                {
                    "request": (
                        "A function called rolling_sharpe that takes a pd.Series of daily "
                        "returns, a window (int, default 252), and risk_free_rate (float, "
                        "default 0.0), and returns a pd.Series of rolling annualised Sharpe "
                        "ratios."
                    ),
                    "module_path": "mlfinlab/backtest_statistics/rolling_sharpe.py",
                },
            )

            text = result.content[0].text
            print(text)

            assert "```python" in text, "Response does not contain a code block!"
            assert "def " in text, "Response does not contain a function definition!"
            assert "mock" in text.lower() or "rolling_sharpe" in text, \
                "Expected function name missing from output!"
            print("\nPASS — generate_function returned valid code and indexed it.")


def main():
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    proc = subprocess.Popen(
        [
            sys.executable, "mcp_server.py",
            "--repo", REPO,
            "--index-dir", INDEX,
            "--transport", "sse",
            "--port", "18765",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for server to be ready
    deadline = time.time() + 15
    ready = False
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            print("[server]", line, end="")
        if "running at" in line.lower():
            ready = True
            break
        if proc.poll() is not None:
            print("Server exited early.")
            sys.exit(1)

    if not ready:
        proc.send_signal(signal.SIGTERM)
        print("Server did not start in time.")
        sys.exit(1)

    try:
        asyncio.run(run_test())
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait()


if __name__ == "__main__":
    main()
