import anyio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

async def main():
    params = StdioServerParameters(
        command="C:/Python313/Scripts/uv.exe",
        args=[
            "run",
            "--directory",
            "C:/workspace/graphiti/mcp_server",
            "--project",
            ".",
            "graphiti_mcp_server.py",
            "--transport",
            "stdio",
        ],
    )
    async with stdio_client(params) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        result = await session.initialize()
        print("Initialized:", result.serverInfo)

anyio.run(main)
