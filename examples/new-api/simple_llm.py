import asyncio

from mcp_agent.agents.llm_agent import LlmAgent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.llm.model_factory import ModelFactory


async def main():
    """Main async function"""
    print("Hello from async main!")
    test: AgentConfig = AgentConfig("hello", model="kimi")
    agent: LlmAgent = LlmAgent(test)
    await agent.attach_llm(ModelFactory.create_factory("haiku"))
    print(await agent.send("hello world"))
    # Your async code here


if __name__ == "__main__":
    asyncio.run(main())
