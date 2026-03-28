"""
demo_run.py — Simulates a realistic agent session on fruit-thrower.

The agent is tasked with building a portfolio construction feature.
It searches the index for relevant existing code, finds gaps, then
requests a new function via generate_function (Codex backend).
"""
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SERVER_URL = "http://127.0.0.1:18766/sse"
REPO = str(Path(__file__).parent / "fin-kit")
INDEX = str(Path(__file__).parent / ".code_index")
DIVIDER = "─" * 72


def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


async def run_demo():
    from mcp.client.sse import sse_client
    from mcp.client.session import ClientSession

    async with sse_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── Step 1: agent orientation ──────────────────────────────────
            section("STEP 1 — Agent lists available tools")
            tools = await session.list_tools()
            for t in tools.tools:
                print(f"  • {t.name}")

            # ── Step 2: first search ───────────────────────────────────────
            section("STEP 2 — Agent searches: 'portfolio optimisation minimum variance'")
            r = await session.call_tool(
                "search_code",
                {"query": "portfolio optimisation minimum variance weights", "n_results": 3},
            )
            print(r.content[0].text)

            # ── Step 3: second search (refine) ─────────────────────────────
            section("STEP 3 — Agent refines: 'risk parity equal risk contribution'")
            r = await session.call_tool(
                "search_code",
                {"query": "risk parity equal risk contribution portfolio weights", "n_results": 3},
            )
            print(r.content[0].text)

            # ── Step 4: third search (look for anything close) ─────────────
            section("STEP 4 — Agent broadens: 'covariance matrix portfolio weights allocation'")
            r = await session.call_tool(
                "search_code",
                {"query": "covariance matrix portfolio weights allocation", "n_results": 3, "kind": "function"},
            )
            print(r.content[0].text)

            # ── Step 5: check module list for portfolio module ─────────────
            section("STEP 5 — Agent checks module list for any portfolio-related module")
            r = await session.call_tool("list_modules", {})
            lines = r.content[0].text.splitlines()
            portfolio_lines = [l for l in lines if "portfolio" in l.lower() or "weight" in l.lower() or "alloc" in l.lower()]
            if portfolio_lines:
                print("\n".join(portfolio_lines))
            else:
                print("  (no dedicated portfolio/weight/allocation module found)")

            # ── Step 6: agent concludes and requests generation ────────────
            section("STEP 6 — Agent concludes: function not present → calling generate_function")
            print("""
  Agent reasoning:
    • search_code found volatility estimators and backtest stats but no
      portfolio construction / weight optimisation functions.
    • No 'portfolio' module exists in the index.
    • Requesting a new `equal_risk_contribution` function via Codex.
""")

            r = await session.call_tool(
                "generate_function",
                {
                    "request": (
                        "A function called equal_risk_contribution that takes a pd.DataFrame "
                        "of asset returns (DatetimeIndex, ticker columns) and returns a "
                        "pd.Series of portfolio weights (indexed by ticker) such that each "
                        "asset contributes equally to total portfolio variance. "
                        "Use an iterative algorithm (e.g. Newton or simple normalisation loop). "
                        "Include convergence tolerance and max_iter parameters. "
                        "Raise ValueError if the covariance matrix is singular."
                    ),
                    "module_path": "mlfinlab/portfolio/equal_risk_contribution.py",
                },
            )
            print(r.content[0].text)

            # ── Step 7: verify it's now searchable ─────────────────────────
            section("STEP 7 — Agent verifies the new function is searchable")
            r = await session.call_tool(
                "search_code",
                {"query": "equal risk contribution portfolio weights", "n_results": 2},
            )
            print(r.content[0].text)


def main():
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [
            sys.executable, "mcp_server.py",
            "--repo", REPO,
            "--index-dir", INDEX,
            "--transport", "sse",
            "--port", "18766",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 15
    ready = False
    while time.time() < deadline:
        line = proc.stdout.readline()
        if "running at" in line.lower():
            print(f"[server] {line.strip()}\n")
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
        asyncio.run(run_demo())
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait()


if __name__ == "__main__":
    main()
