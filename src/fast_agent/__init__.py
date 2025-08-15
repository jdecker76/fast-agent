"""fast-agent - An MCP native agent application framework"""

from typing import TYPE_CHECKING

# Core fast-agent components
from fast_agent.core import Core
from fast_agent.context import Context
from fast_agent.context_dependent import ContextDependent
from fast_agent.mcp_server_registry import ServerRegistry

# Configuration and settings
from fast_agent.config import (
    Settings,
    MCPSettings,
    MCPServerSettings,
    MCPServerAuthSettings,
    MCPSamplingSettings,
    MCPElicitationSettings,
    MCPRootSettings,
    AnthropicSettings,
    OpenAISettings,
    DeepSeekSettings,
    GoogleSettings,
    XAISettings,
    GenericSettings,
    OpenRouterSettings,
    AzureSettings,
    GroqSettings,
    OpenTelemetrySettings,
    TensorZeroSettings,
    BedrockSettings,
    HuggingFaceSettings,
    LoggerSettings,
)

# Progress and event tracking
from fast_agent.event_progress import ProgressAction, ProgressEvent

# Agents
from fast_agent.agents.tool_agent_sync import ToolAgentSynchronous

# Import important MCP types for re-export
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    Prompt,
    PromptMessage,
    ReadResourceResult,
    Role,
    TextContent,
    Tool,
)

# MCP content creation utilities (if still needed from mcp_agent)
if TYPE_CHECKING:
    from mcp_agent.core.mcp_content import (
        Assistant,
        MCPFile,
        MCPImage,
        MCPPrompt,
        MCPText,
        User,
        create_message,
    )
    from mcp_agent.core.request_params import RequestParams

__all__ = [
    # Core fast-agent components
    "Core",
    "Context",
    "ContextDependent",
    "ServerRegistry",
    # Configuration and settings
    "Settings",
    "MCPSettings",
    "MCPServerSettings",
    "MCPServerAuthSettings",
    "MCPSamplingSettings",
    "MCPElicitationSettings",
    "MCPRootSettings",
    "AnthropicSettings",
    "OpenAISettings",
    "DeepSeekSettings",
    "GoogleSettings",
    "XAISettings",
    "GenericSettings",
    "OpenRouterSettings",
    "AzureSettings",
    "GroqSettings",
    "OpenTelemetrySettings",
    "TensorZeroSettings",
    "BedrockSettings",
    "HuggingFaceSettings",
    "LoggerSettings",
    # Progress and event tracking
    "ProgressAction",
    "ProgressEvent",
    # Agents
    "ToolAgentSynchronous",
    # MCP types (re-exported for convenience)
    "Prompt",
    "Tool",
    "CallToolResult",
    "TextContent",
    "ImageContent",
    "PromptMessage",
    "GetPromptResult",
    "ReadResourceResult",
    "EmbeddedResource",
    "Role",
]
