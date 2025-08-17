from typing import List, Union

from mcp.types import CallToolResult, PromptMessage, Tool

from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.context import Context
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging import logger
from mcp_agent.mcp.helpers.content_helpers import text_content
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class SimpleTool(Tool):
    async def execute(self, *args, **kwargs):
        pass


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
        context: Context | None = None,  # noqa: F821
    ) -> None:
        super().__init__(config=config, context=context)
        self._tools = tools
        self._logger = logger.get_logger(__name__)

    async def generate(
        self,
        messages: Union[
            str,
            PromptMessage,
            PromptMessageMultipart,
            List[Union[str, PromptMessage, PromptMessageMultipart]],
        ],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageMultipart:
        """
        Generate a response using the LLM, and handle tool calls if necessary.
        """

        if tools is None:
            tools = self._tools

        while True:
            result = await super().generate(messages, request_params, tools=tools)
            if LlmStopReason.TOOL_USE == result.stop_reason:
                messages = await self.run_tools(result)
            else:
                break

        return result

    async def run_tools(self, request: PromptMessageMultipart) -> PromptMessageMultipart:
        """Runs the tools in the request, and returns a new User message with the results"""
        if not request.tool_calls:
            self._logger.warning("No tool calls found in request", data=request)

        tool_results: dict[str, CallToolResult] = {}
        for correlation_id, tool in (request.tool_calls or {}).items():
            if isinstance(tool, SimpleTool):
                await self.execute_tool(tool)
            else:
                self._logger.warning("Unsupported tool type", data={"tool": tool})
                tool_results[correlation_id] = CallToolResult(
                    content=[text_content("Tool call failed")], isError=True
                )

        return PromptMessageMultipart(role="user", tool_results=tool_results)

    async def execute_tool(self, tool: SimpleTool) -> None:
        result = await tool.execute()
        self._logger.debug(f"Tool {tool.name} executed", data={"tool": tool, "result": result})
