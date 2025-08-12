"""
Utilities for setting per-request MCP metadata.

This module provides a context-local (async-safe) store for metadata that
should be attached to outgoing MCP messages for the current request/turn.

Typical usage (from an API handler or CLI entrypoint):

    from mcp_agent.mcp.request_metadata import set_mcp_metadata, with_mcp_metadata

    set_mcp_metadata({"X-Auth0Id": auth0_id, "session_id": session_id})
    # Call agents; all MCP tool calls will include this metadata

Or as a context manager:

    with with_mcp_metadata({"X-Auth0Id": auth0_id}):
        await agent.generate(...)
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Dict, Optional


_MCP_REQUEST_METADATA: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "mcp_request_metadata", default={}
)


def set_mcp_metadata(metadata: Optional[Dict[str, Any]]) -> None:
    """Set (replace) the current request's MCP metadata.

    Passing None clears existing metadata. Passing a dict replaces the metadata
    for the current async context only.
    """
    _MCP_REQUEST_METADATA.set({} if metadata is None else dict(metadata))


def get_mcp_metadata() -> Dict[str, Any]:
    """Get the current request's MCP metadata (copy).

    Returns a copy to prevent accidental mutation of the stored state.
    """
    current = _MCP_REQUEST_METADATA.get()
    # Return a shallow copy to avoid callers mutating our internal store
    return dict(current) if current else {}


@contextmanager
def with_mcp_metadata(metadata: Optional[Dict[str, Any]]):
    """Temporarily set metadata within a context block.

    The previous metadata is restored on exit.
    """
    token = _MCP_REQUEST_METADATA.set({} if metadata is None else dict(metadata))
    try:
        yield
    finally:
        _MCP_REQUEST_METADATA.reset(token)


