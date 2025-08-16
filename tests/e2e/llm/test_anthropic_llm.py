import os
import unittest
from typing import TYPE_CHECKING, Annotated

import pytest
from pydantic import BaseModel, Field

from fast_agent.core import Core
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.agents.llm_agent import LlmAgent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.model_factory import ModelFactory
from mcp_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class FormattedResponse(BaseModel):
    thinking: Annotated[
        str, Field(description="Your reflection on the conversation that is not seen by the user.")
    ]
    message: str


class TestAnthropicLLM(unittest.IsolatedAsyncioTestCase):
    """Test cases for Anthropic LLM functionality."""

    async def asyncSetUp(self):
        """Set up test environment with Core and agent."""
        self.test_config = AgentConfig("test")

        # Pass the config file path from the test directory
        config_path = os.path.join(os.path.dirname(__file__), "fastagent.config.yaml")

        # Initialize Core and agent
        self.core = Core(settings=config_path)
        await self.core.initialize()

        self.agent: LlmAgent = LlmAgent(self.test_config, self.core.context)
        await self.agent.attach_llm(ModelFactory.create_factory("haiku"))

    async def test_basic_generation(self):
        """Test basic generation returns END_TURN stop reason."""
        result: PromptMessageMultipart = await self.agent.generate("hello, world")
        assert result.stop_reason is LlmStopReason.END_TURN

    async def test_max_tokens_limit(self):
        """Test generation with max tokens limit returns MAX_TOKENS stop reason."""
        result: PromptMessageMultipart = await self.agent.generate(
            "write a 300 word story", RequestParams(maxTokens=15)
        )
        assert result.stop_reason is LlmStopReason.MAX_TOKENS

    async def test_structured_output(self):
        """Test structured output generation with FormattedResponse model."""
        structured_output, result = await self.agent.structured(
            "lets discuss the weather", FormattedResponse
        )
        assert structured_output
        assert LlmStopReason.END_TURN == result.stop_reason
        if Provider.ANTHROPIC == self.agent.llm.provider:
            assert result.tool_calls
            assert 1 == len(result.tool_calls)
            pass  # make sure we have a tool use block
            # assert structured_output == {"type": "text", "text": "Let's discuss the weather."}

    async def test_tool_user_stop(self) -> None:
        pass


# Keep the original test function for backward compatibility if needed
@pytest.mark.asyncio
async def test_anthropic_llm():
    """Original test function - now just runs the test class."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAnthropicLLM)
    runner = unittest.TextTestRunner()
    runner.run(suite)
