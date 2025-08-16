"""
Interface definitions to prevent circular imports.
This module defines protocols (interfaces) that can be used to break circular dependencies.
"""

from datetime import timedelta
from typing import (
    Any,
    AsyncContextManager,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from a2a.types import AgentCard
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel

from mcp import ClientSession, Tool
from mcp.types import GetPromptResult, Prompt, PromptMessage, ReadResourceResult
from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.usage_tracking import UsageAccumulator
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

__all__ = [
    "MCPConnectionManagerProtocol",
    "ServerRegistryProtocol",
    "ServerConnection",
    "AugmentedLLMProtocol",
    "AgentProtocol",
    "ModelFactoryClassProtocol",
    "LLMFactoryProtocol",
    "ModelFactoryFunctionProtocol",
    "ModelT",
]


@runtime_checkable
class MCPConnectionManagerProtocol(Protocol):
    """Protocol for MCPConnectionManager functionality needed by ServerRegistry."""

    async def get_server(
        self,
        server_name: str,
        client_session_factory: Optional[
            Callable[
                [
                    MemoryObjectReceiveStream,
                    MemoryObjectSendStream,
                    Optional[timedelta],
                ],
                ClientSession,
            ]
        ] = None,
    ) -> "ServerConnection": ...

    async def disconnect_server(self, server_name: str) -> None: ...

    async def disconnect_all_servers(self) -> None: ...


@runtime_checkable
class ServerRegistryProtocol(Protocol):
    """Protocol defining the minimal interface of ServerRegistry needed by gen_client."""

    @property
    def connection_manager(self) -> MCPConnectionManagerProtocol: ...

    def initialize_server(
        self,
        server_name: str,
        client_session_factory: Optional[
            Callable[
                [
                    MemoryObjectReceiveStream,
                    MemoryObjectSendStream,
                    Optional[timedelta],
                ],
                ClientSession,
            ]
        ] = None,
        init_hook: Optional[Callable] = None,
    ) -> AsyncContextManager[ClientSession]:
        """Initialize a server and yield a client session."""
        ...


class ServerConnection(Protocol):
    """Protocol for server connection objects returned by MCPConnectionManager."""

    @property
    def session(self) -> ClientSession: ...


ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMFactoryProtocol(Protocol):
    """Protocol for LLM factory functions that create AugmentedLLM instances.

    This protocol defines the standard signature for factory functions that
    create LLM instances for agents. The factory takes an agent, optional
    request parameters, and additional keyword arguments, and returns an
    AugmentedLLMProtocol instance.
    """

    def __call__(self, agent: "LlmAgentProtocol", **kwargs: Any) -> "AugmentedLLMProtocol":
        """Create and return an AugmentedLLM instance.

        Args:
            agent: The agent that will use this LLM
            request_params: Optional parameters to configure the LLM
            **kwargs: Additional implementation-specific parameters

        Returns:
            An instance implementing AugmentedLLMProtocol
        """
        ...


class ModelFactoryFunctionProtocol(Protocol):
    """
    Returns an LLM Model Factory for the specified model string

    """

    def __call__(self, model: str | None = None) -> LLMFactoryProtocol:
        """Create and return an LLM factory.

        Args:
            model: Optional model specification string

        Returns:
            An LLMFactoryProtocol that can create LLM instances
        """
        ...


class AugmentedLLMProtocol(Protocol):
    """Protocol defining the interface for augmented LLMs"""

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
        """Apply the prompt and return the result as a Pydantic model, or None if coercion fails"""
        ...

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
        Apply messages directly to the LLM.

        Args:
            messages: Message(s) in various formats:
                - String: Converted to a user PromptMessageMultipart
                - PromptMessage: Converted to PromptMessageMultipart
                - PromptMessageMultipart: Used directly
                - List of any combination of the above
            request_params: Optional parameters to configure the LLM request

        Returns:
            A PromptMessageMultipart containing the Assistant response, including Tool Content
        """
        ...

    # TODO -- prompt_name and display should probably be at agent level.
    async def apply_prompt_template(
        self, prompt_result: "GetPromptResult", prompt_name: str
    ) -> str:
        """
        Apply a prompt template as persistent context that will be included in all future conversations.

        Args:
            prompt_result: The GetPromptResult containing prompt messages
            prompt_name: The name of the prompt being applied

        Returns:
            String representation of the assistant's response if generated
        """
        ...

    @property
    def message_history(self) -> List[PromptMessageMultipart]:
        """
        Return the LLM's message history as PromptMessageMultipart objects.

        Returns:
            List of PromptMessageMultipart objects representing the conversation history
        """
        ...

    @property
    def usage_accumulator(self) -> UsageAccumulator | None:
        """
        Return the LLM's usage information
        """
        ...

    @property
    def provider(self) -> Provider:
        """
        Return the LLM provider type.

        Returns:
            The Provider enum value representing the LLM provider
        """
        ...


class LlmAgentProtocol(AugmentedLLMProtocol, Protocol):
    """Protocol defining the interface for LLM agents that can be used with MCP"""

    async def __call__(self, message: Union[str, PromptMessage, PromptMessageMultipart]) -> str:
        """Make the agent callable for sending messages directly."""
        ...

    async def send(self, message: Union[str, PromptMessage, PromptMessageMultipart]) -> str:
        """Convenience method for directly returning strings"""
        ...

    @property
    def llm(self) -> AugmentedLLMProtocol:
        """Return the LLM instance used by this agent, or a runtime error if not attached"""
        ...

    @property
    def name(self) -> str:
        """Agent name"""
        ...

    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent"""
        ...

    async def initialize(self) -> None:
        """Initialize the LLM agent"""
        ...

    async def shutdown(self) -> None:
        """Shut down the LLM agent"""
        ...


class AgentProtocol(LlmAgentProtocol, Protocol):
    """Protocol defining the standard agent interface"""

    async def apply_prompt(
        self,
        prompt: Union[str, "GetPromptResult"],
        arguments: Dict[str, str] | None = None,
        as_template: bool = False,
    ) -> str:
        """Apply an MCP prompt template by name or GetPromptResult object"""
        ...

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str] | None = None,
        server_name: str | None = None,
    ) -> GetPromptResult: ...

    async def list_prompts(self, server_name: str | None = None) -> Mapping[str, List[Prompt]]: ...

    async def list_resources(self, server_name: str | None = None) -> Mapping[str, List[str]]: ...

    async def list_mcp_tools(self, server_name: str | None = None) -> Mapping[str, List[Tool]]: ...

    async def get_resource(
        self, resource_uri: str, server_name: str | None = None
    ) -> ReadResourceResult:
        """Get a resource from a specific server or search all servers"""
        ...

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessage, PromptMessageMultipart],
        resource_uri: str,
        server_name: str | None = None,
    ) -> str:
        """Send a message with an attached MCP resource"""
        ...

    async def agent_card(self) -> AgentCard:
        """Return an A2A Agent Card for this Agent"""
        ...

    async def initialize(self) -> None:
        """Initialize the agent"""
        ...

    async def shutdown(self) -> None:
        """Shut down the agent"""
        ...


class ModelFactoryClassProtocol(Protocol):
    """
    Protocol defining the minimal interface of the ModelFactory class needed by sampling.
    This allows sampling.py to depend on this protocol rather than the concrete ModelFactory class.
    """

    @classmethod
    def create_factory(
        cls, model_string: str, request_params: Optional[RequestParams] = None
    ) -> Callable[..., Any]:
        """
        Creates a factory function that can be used to construct an LLM instance.

        Args:
            model_string: The model specification string
            request_params: Optional parameters to configure LLM behavior

        Returns:
            A factory function that can create an LLM instance
        """
        ...
