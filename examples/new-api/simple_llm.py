import asyncio

from fast_agent.agents.llm_agent import LlmAgent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.llm.model_factory import ModelFactory


async def main():
    test: AgentConfig = AgentConfig("hello", model="kimi")
    agent: LlmAgent = LlmAgent(test)
    await agent.attach_llm(ModelFactory.create_factory("haiku"))
    print(await agent.send("hello world"))


if __name__ == "__main__":
    asyncio.run(main())
