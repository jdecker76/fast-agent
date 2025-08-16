from typing import TYPE_CHECKING

import pytest

from fast_agent.core import Core
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.agents.llm_agent import LlmAgent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.model_factory import ModelFactory

if TYPE_CHECKING:
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


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

        result: PromptMessageMultipart = await agent.generate("hello, world")
        assert result.stop_reason is LlmStopReason.END_TURN

        result: PromptMessageMultipart = await agent.generate(
            "write a 300 word story", RequestParams(maxTokens=15)
        )
        assert result.stop_reason is LlmStopReason.MAX_TOKENS
    await test_anthropic_llm()
