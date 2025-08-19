import asyncio
import os

from mcp.types import CallToolResult

from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.tool_agent_sync import SimpleTool, ToolAgentSynchronous
from fast_agent.core import Core
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.llm.model_factory import ModelFactory
from mcp_agent.mcp.helpers.content_helpers import text_content


async def main():
    core: Core = Core()
    await core.initialize()
    test: AgentConfig = AgentConfig("hello", model="kimi")
    agent: LlmAgent = LlmAgent(test)
    await agent.attach_llm(ModelFactory.create_factory("haiku"))
    print(
        await agent.send("hello world, render some xml tags both inside and outside of code fences")
    )


if __name__ == "__main__":
    asyncio.run(main())
