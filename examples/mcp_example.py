import asyncio
import os
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from asymetry import instrument_openai_agents

instrument_openai_agents()


async def main():
    # 1. Configure the Filesystem MCP
    # We allow it to access the current directory '.'
    # (In a real app, you might point this to a specific /logs folder)
    fs_params = {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            os.getcwd(),  # Allow access to current folder
        ],
    }

    print("📁 Connecting to Filesystem MCP...")

    async with MCPServerStdio(params=fs_params, client_session_timeout_seconds=30) as fs_server:

        # --- AGENT: The File Analyst ---
        # This agent can literally 'read' and 'write' to your hard drive via MCP
        analyst_agent = Agent(
            name="FileAnalyst",
            instructions=(
                "You are a file system expert. Use tools to list files in the directory. "
                "If you find any .md files, read them and create a new file "
                "Output the summary"
            ),
            mcp_servers=[fs_server],
        )

        # --- EXECUTION ---
        # Note: Create a dummy 'note.txt' in your folder first to see it work!
        user_query = "Scan my current folder and generate a summary report of what you find. If you can't or don't know the command, say you dont now and return"

        result = await Runner.run(analyst_agent, user_query)

        print(f"\n✅ Final Result:\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
