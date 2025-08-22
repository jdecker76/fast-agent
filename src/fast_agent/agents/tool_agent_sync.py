from abc import ABC
from typing import List

from mcp.types import CallToolResult, Tool

from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.context import Context
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.content_helpers import text_content
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

logger = get_logger(__name__)


class SimpleTool(ABC, Tool):
    async def execute(self, **kwargs) -> CallToolResult: ...


class ToolAgentSynchronous(LlmAgent):
    """
    A Tool Calling agent. Loops LLM responses, and delegates to a call_tool method.

    This class provides default implementations of the standard agent methods
    and delegates LLM operations to an attached AugmentedLLMProtocol instance.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: list[Tool] = [],
        context: Context | None = None,
    ) -> None:
        super().__init__(config=config, context=context)
        self._tools = tools

    async def generate_impl(
        self,
        messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageMultipart:
        """
        Generate a response using the LLM, and handle tool calls if necessary.
        Messages are already normalized to List[PromptMessageMultipart].
        """
        if tools is None:
            tools = self._tools

        # Keep track of the conversation for tool calling loop
        conversation = messages.copy()

        while True:
            # Call parent's generate_impl which delegates to LLM
            result = await super().generate_impl(conversation, request_params, tools=tools)

            if LlmStopReason.TOOL_USE == result.stop_reason:
                # Add assistant's tool-calling message to conversation
                conversation.append(result)
                # Run tools and get user message with results
                tool_result_message = await self.run_tools(result, tools)
                # Add tool results to conversation
                conversation.append(tool_result_message)
            else:
                break

        return result

    async def _name_map(self, tools: List[Tool]) -> dict[str, Tool]:
        """
        Create a mapping of tool names to tool instances.
        """
        return {tool.name: tool for tool in tools}

    async def run_tools(
        self, request: PromptMessageMultipart, tools: List[Tool]
    ) -> PromptMessageMultipart:
        """Runs the tools in the request, and returns a new User message with the results"""
        if not request.tool_calls:
            logger.warning("No tool calls found in request", data=request)

        tool_results: dict[str, CallToolResult] = {}
        tool_map = await self._name_map(tools)
        for correlation_id, tool_request in (request.tool_calls or {}).items():
            tool = tool_map.get(tool_request.params.name)
            self.display.show_tool_call(
                name=self.name,
                tool_args=tool_request.params.arguments,
                available_tools=[],
                tool_name=tool_request.params.name,
            )
            if isinstance(tool, SimpleTool):
                tool_results[correlation_id] = await tool.execute(*tool_request.params.arguments)
                logger.debug(
                    f"Tool {tool.name} executed",
                    data={"tool": tool, "result": tool_results[correlation_id]},
                )
            else:
                logger.warning("Unsupported tool type", data={"tool": tool_request})
                tool_results[correlation_id] = CallToolResult(
                    content=[text_content("Tool call failed")], isError=True
                )
            self.display.show_tool_result(name=self.name, result=tool_results[correlation_id])

        return PromptMessageMultipart(role="user", tool_results=tool_results)
