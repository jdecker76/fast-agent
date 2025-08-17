import asyncio
import os

from mcp import Tool

from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.tool_agent_sync import ToolAgentSynchronous
from fast_agent.core import Core
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.llm.model_factory import ModelFactory


async def main():
    print(os.getcwd())
    core: Core = Core()
    await core.initialize()
    test: AgentConfig = AgentConfig("hello", model="kimi")
    agent: LlmAgent = LlmAgent(test)
    await agent.attach_llm(ModelFactory.create_factory("haiku"))
    print(await agent.send("hello world"))

    input_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city to check the weather for"}
        },
    }

    tool = Tool(
        name="weather",
        description="call this to check the weather in a city",
        inputSchema=input_schema,
    )

    tool_agent: ToolAgentSynchronous = ToolAgentSynchronous(
        AgentConfig(name="tools", model="haiku")
    )

    tool_agent: ToolAgentSynchronous = ToolAgentSynchronous(test, tools=[tool])
    await tool_agent.attach_llm(ModelFactory.create_factory("haiku"))
    await tool_agent.send("how is the weather in San Francisco?")


if __name__ == "__main__":
    asyncio.run(main())
