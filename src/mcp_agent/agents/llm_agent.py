"""
Base Agent class that implements the AgentProtocol interface.

This class provides default implementations of the standard agent methods
and delegates operations to an attached AugmentedLLMProtocol instance.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from mcp import ListToolsResult, Tool
from mcp.types import (
    GetPromptResult,
    PromptMessage,
)
from opentelemetry import trace
from pydantic import BaseModel

from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.usage_tracking import UsageAccumulator
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol, LlmAgentProtocol, LLMFactoryProtocol
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Define a TypeVar for models
ModelT = TypeVar("ModelT", bound=BaseModel)

# Define a TypeVar for AugmentedLLM and its subclasses
LLM = TypeVar("LLM", bound=AugmentedLLMProtocol)

if TYPE_CHECKING:
    from mcp_agent.context import Context


class LlmAgent(LlmAgentProtocol):
    """
    This class provides default implementations of the standard agent methods
    and delegates LLM operations to an attached AugmentedLLMProtocol instance.
    """

    def __init__(
        self,
        config: AgentConfig,
        context: Optional["Context"] = None,
        **kwargs: Any,
    ) -> None:
        self.config = config

        self._context = context
        self._name = self.config.name
        self._tracer = trace.get_tracer(__name__)
        self.instruction = self.config.instruction
        self.logger = get_logger(f"{__name__}.{self._name}")

        # Store the default request params from config
        self._default_request_params = self.config.default_request_params

        # Initialize the LLM to None (will be set by attach_llm)
        self._llm: Optional[AugmentedLLMProtocol] = None

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    @property
    def agent_type(self) -> AgentType:
        """
        Return the type of this agent.
        """
        return AgentType.LLM

    @property
    def name(self) -> str:
        """
        Return the name of this agent.
        """
        return self._name

    async def attach_llm(
        self,
        llm_factory: LLMFactoryProtocol,
        model: Optional[str] = None,
        request_params: Optional[RequestParams] = None,
        **additional_kwargs,
    ) -> AugmentedLLMProtocol:
        """
        Create and attach an LLM instance to this agent.

        Parameters have the following precedence (highest to lowest):
        1. Explicitly passed parameters to this method
        2. Agent's default_request_params
        3. LLM's default values

        Args:
            llm_factory: A factory function that constructs an AugmentedLLM
            model: Optional model name override
            request_params: Optional request parameters override
            **additional_kwargs: Additional parameters passed to the LLM constructor

        Returns:
            The created LLM instance
        """
        # Start with agent's default params
        effective_params = (
            self._default_request_params.model_copy() if self._default_request_params else None
        )

        # Override with explicitly passed request_params
        if request_params:
            if effective_params:
                # Update non-None values
                for k, v in request_params.model_dump(exclude_unset=True).items():
                    if v is not None:
                        setattr(effective_params, k, v)
            else:
                effective_params = request_params

        # Override model if explicitly specified
        if model and effective_params:
            effective_params.model = model

        # Create the LLM instance
        self._llm = llm_factory(
            agent=self, request_params=effective_params, context=self._context, **additional_kwargs
        )

        return self._llm

    async def __call__(
        self,
        message: Union[str, PromptMessage, PromptMessageMultipart],
    ) -> str:
        """
        Make the agent callable to send messages or start an interactive prompt.

        Args:
            message: Optional message to send to the agent
            agent_name: Optional name of the agent (for consistency with DirectAgentApp)
            default: Default message to use in interactive prompt mode

        Returns:
            The agent's response as a string or the result of the interactive session
        """
        return await self.send(message)

    async def send(self, message: Union[str, PromptMessage, PromptMessageMultipart]) -> str:
        """
        Convenience method to return a string directly
        """
        response = await self.generate(message)
        return response.last_text()

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
        Create a completion with the LLM using the provided messages.

        Template Method pattern: This method normalizes inputs and delegates
        to the abstract _generate_impl() method that subclasses must implement.

        Args:
            messages: Message(s) in various formats:
                - String: Converted to a user PromptMessageMultipart
                - PromptMessage: Converted to PromptMessageMultipart
                - PromptMessageMultipart: Used directly
                - List of any combination of the above
            request_params: Optional parameters to configure the request

        Returns:
            The LLM's response as a PromptMessageMultipart
        """

        assert self._llm
        with self._tracer.start_as_current_span(f"Agent: '{self._name}' generate"):
            return await self._llm.generate(messages, request_params, tools)

    # async def _generate_impl(
    #     self,
    #     normalized_messages: List[PromptMessageMultipart],
    #     request_params: RequestParams | None = None,
    # ) -> PromptMessageMultipart:
    #     """
    #     Default implementation for regular agents - delegates to attached LLM.

    #     Workflow agents should override this method to implement custom logic.

    #     Args:
    #         normalized_messages: Already normalized list of PromptMessageMultipart
    #         request_params: Optional parameters to configure the request

    #     Returns:
    #         The LLM's response as a PromptMessageMultipart
    #     """
    #     assert self._llm, (
    #         "No LLM attached to agent. Workflow agents should override _generate_impl()."
    #     )

    #     return await self._llm.generate(normalized_messages, request_params)

    async def apply_prompt_template(self, prompt_result: GetPromptResult, prompt_name: str) -> str:
        """
        Apply a prompt template as persistent context that will be included in all future conversations.
        Delegates to the attached LLM.

        Args:
            prompt_result: The GetPromptResult containing prompt messages
            prompt_name: The name of the prompt being applied

        Returns:
            String representation of the assistant's response if generated
        """
        assert self._llm
        return await self._llm.apply_prompt_template(prompt_result, prompt_name)

    async def structured(
        self,
        messages: Union[
            str,
            PromptMessage,
            PromptMessageMultipart,
            List[Union[str, PromptMessage, PromptMessageMultipart]],
        ],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Apply the prompt and return the result as a Pydantic model.
        Delegates to the attached LLM.

        Args:
            messages: Message(s) in various formats:
                - String: Converted to a user PromptMessageMultipart
                - PromptMessage: Converted to PromptMessageMultipart
                - PromptMessageMultipart: Used directly
                - List of any combination of the above
            model: The Pydantic model class to parse the result into
            request_params: Optional parameters to configure the LLM request

        Returns:
            An instance of the specified model, or None if coercion fails
        """
        assert self._llm
        with self._tracer.start_as_current_span(f"Agent: '{self._name}' structured"):
            return await self._llm.structured(messages, model, request_params)

    @property
    def message_history(self) -> List[PromptMessageMultipart]:
        """
        Return the agent's message history as PromptMessageMultipart objects.

        This history can be used to transfer state between agents or for
        analysis and debugging purposes.

        Returns:
            List of PromptMessageMultipart objects representing the conversation history
        """
        if self._llm:
            return self._llm.message_history
        return []

    @property
    def usage_accumulator(self) -> UsageAccumulator | None:
        """
        Return the usage accumulator for tracking token usage across turns.

        Returns:
            UsageAccumulator object if LLM is attached, None otherwise
        """
        if self._llm:
            return self._llm.usage_accumulator
        return None

    async def list_tools(self) -> ListToolsResult | None:
        return ListToolsResult(tools=[])

    async def list_servers(self) -> List[str]:
        return ["foo"]
