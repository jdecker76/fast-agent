import pytest

from fast_agent.core import Core
from mcp_agent.agents.llm_agent import LlmAgent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.llm.model_factory import ModelFactory


@pytest.mark.asyncio
async def test_anthropic_llm():
    test: AgentConfig = AgentConfig(
        "hello",
    )

    # Pass the config file path from the test directory
    import os

    config_path = os.path.join(os.path.dirname(__file__), "fastagent.config.yaml")

    core: Core = Core(settings=config_path)
    async with core.run() as app:
        agent: LlmAgent = LlmAgent(test, app.context)
        await agent.attach_llm(ModelFactory.create_factory("haiku"))
        result = await agent.send("hello, world")
        print(result)
