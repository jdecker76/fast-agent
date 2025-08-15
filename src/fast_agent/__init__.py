"""fast-agent - (fast-agent-mcp) An MCP native agent application framework"""

# Import important MCP types
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

# Core agent components
# Note: AgentApp removed from here to avoid circular imports
# Workflow decorators - removed to avoid circular imports
# Users should import these directly from mcp_agent.core.direct_decorators
# FastAgent components - removed to avoid circular imports
# Users should import FastAgent directly from mcp_agent.core.fastagent
# MCP content creation utilities
from mcp_agent.core.mcp_content import (
    Assistant,
    MCPFile,
    MCPImage,
    MCPPrompt,
    MCPText,
    User,
    create_message,
)

# Request configuration
from mcp_agent.core.request_params import RequestParams

# MCP content helpers
from mcp_agent.mcp.helpers import (
    get_image_data,
    get_resource_text,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_resource_link,
    is_text_content,
)

# Core protocol interfaces - removed to avoid circular imports

__all__ = [
    # MCP types
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
    # Request configuration
    "RequestParams",
    # MCP content helpers
    "get_text",
    "get_image_data",
    "get_resource_uri",
    "is_text_content",
    "is_image_content",
    "is_resource_content",
    "is_resource_link",
    "get_resource_text",
    # MCP content creation utilities
    "MCPText",
    "MCPImage",
    "MCPFile",
    "MCPPrompt",
    "User",
    "Assistant",
    "create_message",
]
