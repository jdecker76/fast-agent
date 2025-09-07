import warnings

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class PromptMessageMultipart(PromptMessageExtended):
    """Deprecated. Use fast_agent PromptMessageExtended instead.
    A class representing a multipart prompt message."""


# Emit a deprecation warning at import time so callers notice during migration.
warnings.warn(
    "PromptMessageMultipart is deprecated and will be removed. "
    "Use fast_agent.mcp.PromptMessageExtended instead.",
    DeprecationWarning,
    stacklevel=2,
)
