import asyncio
import sys

from mcp_agent.core.fastagent import FastAgent, PromptExitError


fast_agent = FastAgent(
    name="MCP Filtering Tests",
    parse_cli_args=False,
    quiet=False
)

@fast_agent.agent(
    name=f"test-mcp-filtering",    
    model="gpt-4.1",
    instruction="You are a creative writer.  You have been supplied with tools to help you with creative tasks.",
    servers=["creativity"],
    tools={"creativity": ["reverse_string", "code*"]}  # Only supply the reverse string, and the coding tools by wildcard
)
async def test():
    return "Debug info printed"



async def main():
    try:
        async with fast_agent.run() as agent:
            try:
                # Debug: Print available tools
                test_agent = agent._agent("test-mcp-filtering")
                tools = await test_agent.list_tools()
                print(f"Available tools: {[tool.name for tool in tools.tools]}")
                
                await agent.interactive(agent_name="test-mcp-filtering")

            except PromptExitError:
                print("👋 Goodbye!")
                
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
