"""
Direct FastAgent implementation that uses the simplified Agent architecture.
This replaces the traditional FastAgent with a more streamlined approach that
directly creates Agent instances without proxies.
"""

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from importlib.metadata import version as get_version
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

import yaml
from opentelemetry import trace

from mcp_agent import config
from mcp_agent.app import MCPApp
from mcp_agent.context import Context
from mcp_agent.core.agent_app import AgentApp
from mcp_agent.core.direct_decorators import (
    agent as agent_decorator,
)
from mcp_agent.core.direct_decorators import (
    chain as chain_decorator,
)
from mcp_agent.core.direct_decorators import (
    custom as custom_decorator,
)
from mcp_agent.core.direct_decorators import (
    evaluator_optimizer as evaluator_optimizer_decorator,
)
from mcp_agent.core.direct_decorators import (
    orchestrator as orchestrator_decorator,
)
from mcp_agent.core.direct_decorators import (
    parallel as parallel_decorator,
)
from mcp_agent.core.direct_decorators import (
    router as router_decorator,
)
from mcp_agent.core.direct_factory import (
    create_agents_in_dependency_order,
    get_model_factory,
)
from mcp_agent.core.error_handling import handle_error
from mcp_agent.core.exceptions import (
    AgentConfigError,
    CircularDependencyError,
    ModelConfigError,
    PromptExitError,
    ProviderKeyError,
    ServerConfigError,
    ServerInitializationError,
)
from mcp_agent.core.usage_display import display_usage_report
from mcp_agent.core.validation import (
    validate_provider_keys_post_creation,
    validate_server_references,
    validate_workflow_references,
)
from mcp_agent.human_input.handler import console_input_callback
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompts.prompt_load import load_prompt_multipart

if TYPE_CHECKING:
    from mcp_agent.agents.agent import Agent
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

F = TypeVar("F", bound=Callable[..., Any])  # For decorated functions
logger = get_logger(__name__)


class FastAgent:
    """
    A simplified FastAgent implementation that directly creates Agent instances
    without using proxies.
    """

    def __init__(
        self,
        name: str,
        config_path: str | None = None,
        ignore_unknown_args: bool = False,
        parse_cli_args: bool = True,
        quiet: bool = False,  # Add quiet parameter
    ) -> None:
        """
        Initialize the fast-agent application.

        Args:
            name: Name of the application
            config_path: Optional path to config file
            ignore_unknown_args: Whether to ignore unknown command line arguments
                                 when parse_cli_args is True.
            parse_cli_args: If True, parse command line arguments using argparse.
                            Set to False when embedding FastAgent in another framework
                            (like FastAPI/Uvicorn) that handles its own arguments.
            quiet: If True, disable progress display, tool and message logging for cleaner output
        """
        self.args = argparse.Namespace()  # Initialize args always
        self._programmatic_quiet = quiet  # Store the programmatic quiet setting

        # --- Wrap argument parsing logic ---
        if parse_cli_args:
            # Setup command line argument parsing
            parser = argparse.ArgumentParser(description="DirectFastAgent Application")
            parser.add_argument(
                "--model",
                help="Override the default model for all agents",
            )
            parser.add_argument(
                "--agent",
                default="default",
                help="Specify the agent to send a message to (used with --message)",
            )
            parser.add_argument(
                "-m",
                "--message",
                help="Message to send to the specified agent",
            )
            parser.add_argument(
                "-p", "--prompt-file", help="Path to a prompt file to use (either text or JSON)"
            )
            parser.add_argument(
                "--quiet",
                action="store_true",
                help="Disable progress display, tool and message logging for cleaner output",
            )
            parser.add_argument(
                "--version",
                action="store_true",
                help="Show version and exit",
            )
            parser.add_argument(
                "--server",
                action="store_true",
                help="Run as an MCP server",
            )
            parser.add_argument(
                "--transport",
                choices=["sse", "http", "stdio"],
                default="http",
                help="Transport protocol to use when running as a server (sse or stdio)",
            )
            parser.add_argument(
                "--port",
                type=int,
                default=8000,
                help="Port to use when running as a server with SSE transport",
            )
            parser.add_argument(
                "--host",
                default="0.0.0.0",
                help="Host address to bind to when running as a server with SSE transport",
            )

            if ignore_unknown_args:
                known_args, _ = parser.parse_known_args()
                self.args = known_args
            else:
                # Use parse_known_args here too, to avoid crashing on uvicorn args etc.
                # even if ignore_unknown_args is False, we only care about *our* args.
                known_args, unknown = parser.parse_known_args()
                self.args = known_args
                # Optionally, warn about unknown args if not ignoring?
                # if unknown and not ignore_unknown_args:
                #     logger.warning(f"Ignoring unknown command line arguments: {unknown}")

            # Handle version flag
            if self.args.version:
                try:
                    app_version = get_version("fast-agent-mcp")
                except:  # noqa: E722
                    app_version = "unknown"
                print(f"fast-agent-mcp v{app_version}")
                sys.exit(0)
        # --- End of wrapped logic ---

        # Apply programmatic quiet setting (overrides CLI if both are set)
        if self._programmatic_quiet:
            self.args.quiet = True

        self.name = name
        self.config_path = config_path

        try:
            # Load configuration directly for this instance
            self._load_config()

            # Apply programmatic quiet mode to config before creating app
            if self._programmatic_quiet and hasattr(self, "config"):
                if "logger" not in self.config:
                    self.config["logger"] = {}
                self.config["logger"]["progress_display"] = False
                self.config["logger"]["show_chat"] = False
                self.config["logger"]["show_tools"] = False

            # Create the app with our local settings
            self.app = MCPApp(
                name=name,
                settings=config.Settings(**self.config) if hasattr(self, "config") else None,
            )
            self.app.fast_agent = self

            # Stop progress display immediately if quiet mode is requested
            if self._programmatic_quiet:
                from mcp_agent.progress_display import progress_display

                progress_display.stop()

        except yaml.parser.ParserError as e:
            handle_error(
                e,
                "YAML Parsing Error",
                "There was an error parsing the config or secrets YAML configuration file.",
            )
            raise SystemExit(1)

        # Dictionary to store agent configurations from decorators
        self.agents: Dict[str, Dict[str, Any]] = {}

        # Dictionary to store deactivated agents that failed to load
        self.deactivated_agents: Dict[str, Dict[str, Any]] = {}
        
        # Set to store unavailable server names
        self.unavailable_servers: set[str] = set()

    def _load_config(self) -> None:
        """Load configuration from YAML file including secrets using get_settings
        but without relying on the global cache."""

        # Import but make a local copy to avoid affecting the global state
        from mcp_agent.config import _settings, get_settings

        # Temporarily clear the global settings to ensure a fresh load
        old_settings = _settings
        _settings = None

        try:
            # Use get_settings to load config - this handles all paths and secrets merging
            settings = get_settings(self.config_path)

            # Convert to dict for backward compatibility
            self.config = settings.model_dump() if settings else {}
        finally:
            # Restore the original global settings
            _settings = old_settings

    @property
    def context(self) -> Context:
        """Access the application context"""
        return self.app.context

    # Decorator methods with type-safe implementations
    agent = agent_decorator
    custom = custom_decorator
    orchestrator = orchestrator_decorator
    router = router_decorator
    chain = chain_decorator
    parallel = parallel_decorator
    evaluator_optimizer = evaluator_optimizer_decorator

    @asynccontextmanager
    async def run(self):
        """
        Run the application context, creating agents and handling their lifecycle.
        This simplified version directly creates agent instances.
        """
        polling_task = None
        try:
            async with self.app.run() as running_app:
                # Store the running app instance so agents can access it
                self.app = running_app

                # Define a model factory function that can be passed to agent creation
                def model_factory_func(model=None, request_params=None):
                    return get_model_factory(
                        self.context,
                        model=model,
                        request_params=request_params,
                        default_model=self.config.get("default_model"),
                        cli_model=self.args.model if hasattr(self.args, "model") else None,
                    )

                # Create agents in dependency order
                active_agents = await create_agents_in_dependency_order(
                    app_instance=self.app,
                    agents_dict=self.agents,
                    model_factory_func=model_factory_func,
                )
                    
                # After attempting to load all agents, validate provider keys for active agents
                validate_provider_keys_post_creation(active_agents)

                # Create the agent app with the successfully created agents
                agent_app = AgentApp(agents=active_agents)

                # Start the background polling task to reactivate agents
                polling_task = asyncio.create_task(self._poll_and_reactivate_servers(agent_app))

                yield agent_app
        finally:
            if polling_task:
                polling_task.cancel()
                try:
                    await polling_task
                except asyncio.CancelledError:
                    logger.info("Agent reactivation polling task cancelled.")
            logger.info("FastAgent run context finished.")

    async def _poll_and_reactivate_servers(self, agent_app: AgentApp):
        """
        Periodically poll unavailable servers and reactivate agents if they come online.
        """
        while True:
            await asyncio.sleep(30)  # Poll every 30 seconds for faster testing

            if not self.unavailable_servers:
                continue

            logger.warning(f"Polling unavailable servers: {list(self.unavailable_servers)}")

            # Create a copy of the set to iterate over, as it may be modified
            for server_name in list(self.unavailable_servers):
                # Show server configuration for debugging
                server_config = self.context.server_registry.get_server_config(server_name)
                if server_config:
                    logger.info(f"Server '{server_name}' config: transport={server_config.transport}, command={getattr(server_config, 'command', None)}, url={getattr(server_config, 'url', None)}")
                try:
                    # Use the server_registry to attempt a connection with timeout
                    logger.info(f"Attempting to connect to server '{server_name}'...")
                    async with asyncio.timeout(10):  # 10 second timeout
                        async with self.context.server_registry.start_server(server_name):
                            logger.warning(f"Server '{server_name}' is now available!")
                            self.unavailable_servers.remove(server_name)

                        # Check for agents that can be reactivated
                        for agent_name, agent_config in list(self.deactivated_agents.items()):
                            agent_config_obj = agent_config.get("config")
                            required_servers = agent_config_obj.servers if agent_config_obj else []
                            if server_name in required_servers:
                                # Check if all required servers for this agent are now online
                                if all(s not in self.unavailable_servers for s in required_servers):
                                    logger.warning(f"All required servers available for agent '{agent_name}', attempting reactivation")
                                    await self._reactivate_agent(agent_name, agent_config, agent_app)

                except Exception as e:
                    # Only log every 5th attempt to reduce noise
                    if hasattr(self, '_poll_counter'):
                        self._poll_counter += 1
                    else:
                        self._poll_counter = 1
                    
                    if self._poll_counter % 5 == 0:
                        logger.warning(f"Server '{server_name}' still unavailable after {self._poll_counter} attempts: {str(e)[:100]}...")
                    else:
                        logger.debug(f"Server '{server_name}' still unavailable: {e}")

    async def _reactivate_agent(self, agent_name: str, agent_config: Dict, agent_app: AgentApp):
        """
        Reactivate a single agent that was previously offline.
        """
        logger.warning(f"Attempting to reactivate agent: {agent_name}")
        try:
            # Define a model factory function for reactivation
            def model_factory_func(model=None, request_params=None):
                return get_model_factory(
                    self.context,
                    model=model,
                    request_params=request_params,
                    default_model=self.config.get("default_model"),
                    cli_model=self.args.model if hasattr(self.args, "model") else None,
                )

            # Create the agent
            created_agents = await create_agents_in_dependency_order(
                app_instance=self.app,
                agents_dict={agent_name: agent_config},
                model_factory_func=model_factory_func,
            )

            if agent_name in created_agents:
                # Add to the running agent_app
                agent_app.add_agent(agent_name, created_agents[agent_name])
                # Remove from deactivated list
                del self.deactivated_agents[agent_name]
                
                logger.warning(f"Agent '{agent_name}' has been successfully reactivated!")
            else:
                logger.error(f"Failed to create agent '{agent_name}' during reactivation.")

        except ServerInitializationError as e:
            # This can happen if the server goes down again right as we try to reactivate
            match = re.search(r"MCP Server: '([^']*)'", str(e))
            if match:
                server_name = match.group(1)
                self.unavailable_servers.add(server_name)
                logger.info(
                    f"Server '{server_name}' became unavailable during reactivation of agent '{agent_name}'. "
                    "Reactivation will be re-attempted later."
                )
            else:
                logger.error(f"Could not determine server name from reactivation error: {e}")

        except Exception as e:
            logger.error(f"Error reactivating agent '{agent_name}': {e}")


    def _handle_error(self, e: Exception, error_type: Optional[str] = None) -> None:
        """
        Centralized error handling for the application.

        Args:
            e: The exception that was raised
            error_type: Optional explicit error type
        """
        if isinstance(e, ServerConfigError):
            handle_error(
                e,
                "Server Configuration Error",
                "Please check your 'fastagent.config.yaml' configuration file and add the missing server definitions.",
            )
        elif isinstance(e, ProviderKeyError):
            handle_error(
                e,
                "Provider Configuration Error",
                "Please check your 'fastagent.secrets.yaml' configuration file and ensure all required API keys are set.",
            )
        elif isinstance(e, AgentConfigError):
            handle_error(
                e,
                "Workflow or Agent Configuration Error",
                "Please check your agent definition and ensure names and references are correct.",
            )
        elif isinstance(e, ServerInitializationError):
            handle_error(
                e,
                "MCP Server Startup Error",
                "There was an error starting up the MCP Server.",
            )
        elif isinstance(e, ModelConfigError):
            handle_error(
                e,
                "Model Configuration Error",
                "Common models: gpt-4.1, o3-mini, sonnet, haiku. for o3, set reasoning effort with o3-mini.high",
            )
        elif isinstance(e, CircularDependencyError):
            handle_error(
                e,
                "Circular Dependency Detected",
                "Check your agent configuration for circular dependencies.",
            )
        elif isinstance(e, PromptExitError):
            handle_error(
                e,
                "User requested exit",
            )
        elif isinstance(e, asyncio.CancelledError):
            handle_error(
                e,
                "Cancelled",
                "The operation was cancelled.",
            )
        else:
            handle_error(e, error_type or "Error", "An unexpected error occurred.")

    def _print_usage_report(self, active_agents: dict) -> None:
        """Print a formatted table of token usage for all agents."""
        display_usage_report(active_agents, show_if_progress_disabled=False, subdued_colors=True)

    async def start_server(
        self,
        transport: str = "http",
        host: str = "0.0.0.0",
        port: int = 8000,
        server_name: Optional[str] = None,
        server_description: Optional[str] = None,
    ) -> None:
        """
        Start the application as an MCP server.
        This method initializes agents and exposes them through an MCP server.
        It is a blocking method that runs until the server is stopped.

        Args:
            transport: Transport protocol to use ("stdio" or "sse")
            host: Host address for the server when using SSE
            port: Port for the server when using SSE
            server_name: Optional custom name for the MCP server
            server_description: Optional description for the MCP server
        """
        # This method simply updates the command line arguments and uses run()
        # to ensure we follow the same initialization path for all operations

        # Store original args
        original_args = None
        if hasattr(self, "args"):
            original_args = self.args

        # Create our own args object with server settings
        from argparse import Namespace

        self.args = Namespace()
        self.args.server = True
        self.args.transport = transport
        self.args.host = host
        self.args.port = port
        self.args.quiet = (
            original_args.quiet if original_args and hasattr(original_args, "quiet") else False
        )
        self.args.model = None
        if hasattr(original_args, "model"):
            self.args.model = original_args.model

        # Run the application, which will detect the server flag and start server mode
        async with self.run():
            pass  # This won't be reached due to SystemExit in run()

        # Restore original args (if we get here)
        if original_args:
            self.args = original_args

    # Keep run_with_mcp_server for backward compatibility
    async def run_with_mcp_server(
        self,
        transport: str = "sse",
        host: str = "0.0.0.0",
        port: int = 8000,
        server_name: Optional[str] = None,
        server_description: Optional[str] = None,
    ) -> None:
        """
        Run the application and expose agents through an MCP server.
        This method is kept for backward compatibility.
        For new code, use start_server() instead.

        Args:
            transport: Transport protocol to use ("stdio" or "sse")
            host: Host address for the server when using SSE
            port: Port for the server when using SSE
            server_name: Optional custom name for the MCP server
            server_description: Optional description for the MCP server
        """
        await self.start_server(
            transport=transport,
            host=host,
            port=port,
            server_name=server_name,
            server_description=server_description,
        )

    async def main(self):
        """
        Helper method for checking if server mode was requested.

        Usage:
        ```python
        fast = FastAgent("My App")

        @fast.agent(...)
        async def app_main():
            # Check if server mode was requested
            # This doesn't actually do anything - the check happens in run()
            # But it provides a way for application code to know if server mode
            # was requested for conditionals
            is_server_mode = hasattr(self, "args") and self.args.server

            # Normal run - this will handle server mode automatically if requested
            async with fast.run() as agent:
                # This code only executes for normal mode
                # Server mode will exit before reaching here
                await agent.send("Hello")
        ```

        Returns:
            bool: True if --server flag is set, False otherwise
        """
        # Just check if the flag is set, no action here
        # The actual server code will be handled by run()
        return hasattr(self, "args") and self.args.server
