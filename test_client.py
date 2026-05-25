# test_client.py
import asyncio

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


async def main():
    async with sse_client("http://127.0.0.1:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            # Call search
            result = await session.call_tool("search_code", {"query": "CUSUM filter"})
            print(result.content[0].text)

            # List modules
            result = await session.call_tool("list_modules", {})
            print(result.content[0].text)


asyncio.run(main())
