"""
LLM Agent class that adds interaction behaviors to LlmDecorator.

This class extends LlmDecorator with LLM-specific interaction behaviors including:
- UI display methods for messages, tools, and prompts
- Stop reason handling
- Tool call tracking
- Chat display integration
"""

from typing import List, Optional, Union

from mcp.types import CallToolResult, PromptMessage
from rich.text import Text

from fast_agent.agents.llm_decorator import LlmDecorator
from fast_agent.context import Context
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.ui.console_display import ConsoleDisplay


class LlmAgent(LlmDecorator):
    """
    An LLM agent that adds interaction behaviors to the base LlmDecorator.

    This class provides LLM-specific functionality including UI display methods,
    tool call tracking, and chat interaction patterns while delegating core
    LLM operations to the attached AugmentedLLMProtocol.
    """

    def __init__(
        self,
        config: AgentConfig,
        context: Context | None = None,
    ) -> None:
        super().__init__(config=config, context=context)

        # Initialize display component
        self.display = ConsoleDisplay(config=self._context.config if self._context else None)

        # Tool call tracking for current turn
        self._current_turn_tool_calls = 0

    def _reset_turn_tool_calls(self) -> None:
        """Reset tool call counter for new turn."""
        self._current_turn_tool_calls = 0

    async def show_assistant_message(
        self,
        message_text: str | Text | None,
        highlight_namespaced_tool: str = "",
        title: str = "ASSISTANT",
    ) -> None:
        """Display an assistant message in a formatted panel."""
        if message_text is None:
            message_text = Text("No content to display", style="dim green italic")

        await self.display.show_assistant_message(
            message_text,
            aggregator=None,
            highlight_namespaced_tool=highlight_namespaced_tool,
            title=title,
            name=self.name,
        )

    def show_user_message(self, message, model: str | None, chat_turn: int) -> None:
        """Display a user message in a formatted panel."""
        self.display.show_user_message(message, model, chat_turn, name=self.name)

    # async def show_prompt_loaded(
    #     self,
    #     prompt_name: str,
    #     description: Optional[str] = None,
    #     message_count: int = 0,
    #     arguments: Optional[dict[str, str]] = None,
    # ) -> None:
    #     """
    #     Display information about a loaded prompt template.

    #     Args:
    #         prompt_name: The name of the prompt
    #         description: Optional description of the prompt
    #         message_count: Number of messages in the prompt
    #         arguments: Optional dictionary of arguments passed to the prompt
    #     """
    #     # Get aggregator from attached LLM if available
    #     aggregator = None
    #     if self._llm and hasattr(self._llm, "aggregator"):
    #         aggregator = self._llm.aggregator

    #     await self.display.show_prompt_loaded(
    #         prompt_name=prompt_name,
    #         description=description,
    #         message_count=message_count,
    #         agent_name=self.name,
    #         aggregator=aggregator,
    #         arguments=arguments,
    #     )

    async def generate(
        self,
        messages: Union[
            str,
            PromptMessage,
            PromptMessageMultipart,
            List[Union[str, PromptMessage, PromptMessageMultipart]],
        ],
        request_params=None,
        tools=None,
    ) -> PromptMessageMultipart:
        """
        Enhanced generate method that resets tool call tracking.
        """
        # Reset tool call counter for new turn
        self._reset_turn_tool_calls()
        
        # Delegate to parent implementation
        return await super().generate(messages, request_params, tools)
