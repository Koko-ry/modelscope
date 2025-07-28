#!/usr/bin/env python3
"""
MCP Client - A concise implementation based on the official MCP Python SDK
"""

import asyncio
import time
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Dict, List, Optional

# Import official MCP SDK
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, Implementation

from modelscope.utils.logger import get_logger

# Constants
DEFAULT_CLIENT_INFO = Implementation(
    name='ModelScope-mcp-client', version='1.0.0')

DEFAULT_READ_TIMEOUT = timedelta(seconds=30)
DEFAULT_HTTP_TIMEOUT = timedelta(seconds=30)
DEFAULT_SSE_READ_TIMEOUT = timedelta(seconds=30)

# Logger
logger = get_logger(__name__)


# Exception classes
class MCPClientError(Exception):
    """Base MCP client exception"""
    pass


class MCPConnectionError(MCPClientError):
    """MCP connection exception"""
    pass


class MCPToolExecutionError(MCPClientError):
    """MCP tool execution exception"""
    pass


class MCPTimeoutError(MCPClientError):
    """MCP timeout exception"""
    pass


class MCPClient:
    """
    MCP Client - A comprehensive MCP (Model Context Protocol) client implementation

    This client provides a simple and robust interface for connecting to and interacting
    with MCP servers using various transport protocols (STDIO, SSE, Streamable HTTP).

    Supported Transport Types:
        - stdio: Standard input/output communication with local processes
        - sse: Server-Sent Events over HTTP
        - streamable_http: Streamable HTTP transport

    Basic Usage:
        >>> import asyncio
        >>> from modelscope.hub.mcp.client import MCPClient

        >>> # 1. STDIO Transport (local process)
        >>> stdio_config = {
        ...     "type": "stdio",
        ...     "command": ["python", "my_mcp_server.py"]
        ... }
        >>> client = MCPClient(stdio_config)

        >>> # 2. SSE Transport (Server-Sent Events)
        >>> sse_config = {
        ...     "type": "sse",
        ...     "url": "https://api.example.com/mcp/sse"
        ... }
        >>> client = MCPClient(sse_config)

        >>> # 3. Streamable HTTP Transport
        >>> http_config = {
        ...     "type": "streamable_http",
        ...     "url": "https://api.example.com/mcp/stream"
        ... }
        >>> client = MCPClient(http_config)

    Context Manager Usage (Recommended):
        >>> async def main():
        ...     config = {"type": "sse", "url": "https://api.example.com/mcp/sse"}
        ...
        ...     async with MCPClient(config) as client:
        ...         # Check connection status
        ...         if client.is_connected():
        ...             print(f"Connected to {client.get_server_name()}")
        ...
        ...         # List available tools
        ...         tools = await client.list_tools()
        ...         for tool in tools:
        ...             print(f"Tool: {tool.name} - {tool.description}")
        ...
        ...         # Call a tool
        ...         result = await client.call_tool("search", {"query": "Python"})
        ...         print(f"Result: {result}")
        ...
        ...     # Client automatically disconnected
        >>> asyncio.run(main())

    Manual Connection Management:
        >>> async def manual_connection():
        ...     config = {"type": "sse", "url": "https://api.example.com/mcp/sse"}
        ...     client = MCPClient(config)
        ...
        ...     try:
        ...         # Connect to server
        ...         await client.connect()
        ...
        ...         # Get server information
        ...         server_info = client.get_server_info()
        ...         transport_type = client.get_transport_type()
        ...         print(f"Connected to {server_info} via {transport_type}")
        ...
        ...         # List tools with timeout
        ...         from datetime import timedelta
        ...         tools = await client.list_tools(timeout=timedelta(seconds=10))
        ...
        ...         # Call tool with custom timeout
        ...         result = await client.call_tool(
        ...             "analyze_code",
        ...             {"code": "print('hello')", "language": "python"},
        ...             timeout=timedelta(seconds=30)
        ...         )
        ...         print(f"Analysis result: {result}")
        ...
        ...     finally:
        ...         # Always disconnect
        ...         await client.disconnect()
        >>> asyncio.run(manual_connection())

    Complex Configuration (mcpServers format):
        >>> complex_config = {
        ...     "mcpServers": {
        ...         "my-server": {
        ...             "type": "sse",
        ...             "url": "https://api.example.com/mcp/sse"
        ...         }
        ...     }
        ... }
        >>> client = MCPClient(complex_config)  # Will use first server

    Error Handling:
        >>> async def error_handling_example():
        ...     config = {"type": "sse", "url": "https://invalid-url.com"}
        ...     client = MCPClient(config)
        ...
        ...     try:
        ...         await client.connect()
        ...         result = await client.call_tool("nonexistent_tool", {})
        ...     except MCPConnectionError as e:
        ...         print(f"Connection failed: {e}")
        ...     except MCPToolExecutionError as e:
        ...         print(f"Tool execution failed: {e}")
        ...     except MCPTimeoutError as e:
        ...         print(f"Operation timed out: {e}")
        ...     except MCPClientError as e:
        ...         print(f"General MCP error: {e}")
        ...     finally:
        ...         await client.disconnect()
        >>> asyncio.run(error_handling_example())

    Batch Operations:
        >>> async def batch_operations():
        ...     config = {"type": "sse", "url": "https://api.example.com/mcp/sse"}
        ...
        ...     async with MCPClient(config) as client:
        ...         # Get all tools first
        ...         tools = await client.list_tools()
        ...         tool_names = [tool.name for tool in tools]
        ...
        ...         # Call multiple tools
        ...         results = []
        ...         for tool_name in tool_names[:3]:  # Limit to first 3 tools
        ...             try:
        ...                 result = await client.call_tool(tool_name, {})
        ...                 results.append((tool_name, result))
        ...             except Exception as e:
        ...                 results.append((tool_name, f"Error: {e}"))
        ...
        ...         return results
        >>> results = asyncio.run(batch_operations())

    Server Information Access:
        >>> async def server_info_example():
        ...     config = {"type": "sse", "url": "https://api.example.com/mcp/sse"}
        ...
        ...     async with MCPClient(config) as client:
        ...         # Access server details
        ...         print(f"Server name: {client.get_server_name()}")
        ...         print(f"Transport: {client.get_transport_type()}")
        ...         print(f"Connected: {client.is_connected()}")
        ...
        ...         server_info = client.get_server_info()
        ...         if server_info:
        ...             print(f"Server version: {server_info.get('version')}")
        ...             print(f"Server implementation: {server_info.get('name')}")
        >>> asyncio.run(server_info_example())

    Attributes:
        mcp_server (Dict[str, Any]): Server configuration
        session (Optional[ClientSession]): MCP session instance
        exit_stack (Optional[AsyncExitStack]): Resource management stack
        client_info (Implementation): Client implementation info
        connected (bool): Connection status
        read_timeout (timedelta): Default read timeout
        server_info (Optional[Dict[str, Any]]): Server information
        server_name (str): Server display name

    Raises:
        MCPClientError: Base exception for all MCP client errors
        MCPConnectionError: Connection-related errors
        MCPToolExecutionError: Tool execution errors
        MCPTimeoutError: Timeout-related errors
        ValueError: Invalid configuration or parameters
    """

    def __init__(self, mcp_server: Dict[str, Any]):
        """
        Initialize MCP client with server configuration.

        Args:
            mcp_server (Dict[str, Any]): MCP server configuration dictionary.
                Supported formats:
                1. Direct configuration:
                   {"type": "sse", "url": "https://api.example.com/mcp/sse"}
                   {"type": "stdio", "command": ["python", "server.py"]}
                   {"type": "streamable_http", "url": "https://api.example.com/mcp/stream"}

                2. mcpServers nested format:
                   {"mcpServers": {"server1": {"type": "sse", "url": "..."}}}

        Raises:
            ValueError: If configuration is invalid or missing required fields.

        Examples:
            >>> # SSE server configuration
            >>> sse_config = {
            ...     "type": "sse",
            ...     "url": "https://api.example.com/mcp/sse"
            ... }
            >>> client = MCPClient(sse_config)

            >>> # STDIO server configuration
            >>> stdio_config = {
            ...     "type": "stdio",
            ...     "command": ["python", "/path/to/mcp_server.py", "--port", "8080"]
            ... }
            >>> client = MCPClient(stdio_config)

            >>> # Nested mcpServers format
            >>> nested_config = {
            ...     "mcpServers": {
            ...         "my-server": {
            ...             "type": "sse",
            ...             "url": "https://api.example.com/mcp/sse"
            ...         }
            ...     }
            ... }
            >>> client = MCPClient(nested_config)
        """
        if not mcp_server:
            raise ValueError('MCP server configuration is required')

        self.mcp_server = mcp_server
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self.client_info = DEFAULT_CLIENT_INFO
        self.connected = False
        self.read_timeout = DEFAULT_READ_TIMEOUT
        self.server_info: Optional[Dict[str, Any]] = None  # Server information

        # Validate configuration
        self._validate_config()

        # Auto-generate server name (maybe updated after connection)
        self.server_name = self._generate_server_name()

    def _generate_server_name(self) -> str:
        """Auto-generate server name"""
        config = self.mcp_server

        # Extract meaningful name from configuration
        if 'type' in config:
            transport_type = config['type']

            if transport_type == 'stdio' and 'command' in config:
                # Extract name from command
                command = config['command']
                if isinstance(command, list) and command:
                    return f'stdio-{command[0]}'
                elif isinstance(command, str):
                    return f'stdio-{command}'

            elif (transport_type in ['sse', 'streamable_http']
                  and 'url' in config):
                # Extract domain from URL
                url = config['url']
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    domain = parsed.netloc.split('.')[
                        0]  # Get first domain part
                    return f'{transport_type}-{domain}'
                except Exception:
                    return f'{transport_type}-server'

        # Default name
        return f"mcp-{config.get('type', 'unknown')}-server"

    def _validate_config(self) -> None:
        """Validate MCP server configuration"""
        config = self.mcp_server

        # Check for mcpServers nested structure
        if 'mcpServers' in config:
            servers = config['mcpServers']
            if not servers:
                raise ValueError('No servers found in mcpServers')

            # Get first server configuration
            first_server_name = list(servers.keys())[0]
            first_server_config = servers[first_server_name]

            # Validate server configuration
            if not isinstance(first_server_config, dict):
                raise ValueError(
                    f'Server configuration for {first_server_name} must be a dictionary'
                )

            # Auto-detect transport type if not specified
            if 'type' not in first_server_config:
                if 'command' in first_server_config:
                    first_server_config['type'] = 'stdio'
                elif 'url' in first_server_config:
                    # Default to SSE for URL-based connections
                    first_server_config['type'] = 'sse'
                else:
                    raise ValueError(
                        f'Server type cannot be determined for {first_server_name}: '
                        'missing both command and url')

            # Validate required fields based on type
            server_type = first_server_config['type']
            if server_type == 'stdio':
                if 'command' not in first_server_config:
                    raise ValueError(
                        f'Command is required for stdio server {first_server_name}'
                    )
            elif server_type in ['sse', 'streamable_http']:
                if 'url' not in first_server_config:
                    raise ValueError(
                        f'URL is required for {server_type} server {first_server_name}'
                    )

            self.mcp_server = first_server_config
        else:
            # Direct configuration
            # Auto-detect transport type if not specified
            if 'type' not in config:
                if 'command' in config:
                    config['type'] = 'stdio'
                elif 'url' in config:
                    # Default to SSE for URL-based connections
                    config['type'] = 'sse'
                else:
                    raise ValueError(
                        'Server type cannot be determined: missing both command and url'
                    )

            # Validate required fields based on type
            server_type = config['type']
            if server_type == 'stdio':
                if 'command' not in config:
                    raise ValueError('Command is required for stdio server')
            elif server_type in ['sse', 'streamable_http']:
                if 'url' not in config:
                    raise ValueError(
                        f'URL is required for {server_type} server')

            # Validate transport type
            if server_type not in ['stdio', 'sse', 'streamable_http']:
                raise ValueError(f'Unsupported transport type: {server_type}')

    async def connect(self) -> None:
        """
        Establish connection to the MCP server.

        This method creates the appropriate transport (STDIO, SSE, or Streamable HTTP),
        establishes the session, and performs the MCP initialization handshake.

        Raises:
            MCPConnectionError: If connection fails for any reason.
            ValueError: If server configuration is invalid.

        Examples:
            >>> import asyncio
            >>> config = {"type": "sse", "url": "https://api.example.com/mcp/sse"}
            >>> client = MCPClient(config)
            >>>
            >>> async def connect_example():
            ...     try:
            ...         await client.connect()
            ...         print(f"Connected to {client.get_server_name()}")
            ...     except MCPConnectionError as e:
            ...         print(f"Connection failed: {e}")
            ...     finally:
            ...         await client.disconnect()
            >>> asyncio.run(connect_example())
        """
        if self.connected:
            logger.warning(f'Already connected to server {self.server_name}')
            return

        try:
            # Create new exit_stack
            self.exit_stack = AsyncExitStack()

            # Establish connection based on transport type
            if self.mcp_server['type'] == 'stdio':
                read, write = await self._establish_stdio_connection()
            elif self.mcp_server['type'] == 'sse':
                read, write = await self._establish_sse_connection()
            elif self.mcp_server['type'] == 'streamable_http':
                read, write = await self._establish_streamable_http_connection(
                )
            else:
                raise MCPConnectionError(
                    f'Unsupported transport type: {self.mcp_server["type"]}')

            # Create session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                    client_info=self.client_info,
                    read_timeout_seconds=self.read_timeout,
                ))

            # Initialize session
            init_result = await self.session.initialize()

            # Get server information and update server name
            self._update_server_info(init_result)

            self.connected = True
            logger.info(f'Connected to server {self.server_name}')

        except Exception as e:
            logger.error(
                f'Failed to connect to server {self.server_name}: {e}')
            await self._cleanup()
            raise MCPConnectionError(f'Connection failed: {e}') from e

    async def _establish_stdio_connection(self) -> tuple[Any, Any]:
        """Establish STDIO connection"""
        config = self.mcp_server
        command = config.get('command', [])

        if not command:
            raise ValueError('STDIO command is required')

        # Create STDIO transport
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(StdioServerParameters(command=command)))
        return stdio_transport[0], stdio_transport[1]  # read, write

    async def _establish_sse_connection(self) -> tuple[Any, Any]:
        """Establish SSE connection"""
        config = self.mcp_server
        url = config.get('url')

        if not url:
            raise ValueError('SSE URL is required')

        # Create SSE transport
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(
                url,
                timeout=DEFAULT_HTTP_TIMEOUT.total_seconds(),
                sse_read_timeout=DEFAULT_SSE_READ_TIMEOUT.total_seconds()))
        return sse_transport[0], sse_transport[1]  # read, write

    async def _establish_streamable_http_connection(self) -> tuple[Any, Any]:
        """Establish Streamable HTTP connection"""
        config = self.mcp_server
        url = config.get('url')

        if not url:
            raise ValueError('Streamable HTTP URL is required')

        # Create Streamable HTTP transport
        streamable_http_transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(
                url,
                timeout=DEFAULT_HTTP_TIMEOUT,
                sse_read_timeout=DEFAULT_SSE_READ_TIMEOUT))
        return (streamable_http_transport[0], streamable_http_transport[1]
                )  # read, write

    def _update_server_info(self, init_result) -> None:
        """Get server information from initialization result and update server name"""
        try:
            # Get server information from initialization result
            if hasattr(init_result, 'serverInfo') and init_result.serverInfo:
                self.server_info = {
                    'name': init_result.serverInfo.name,
                    'version': init_result.serverInfo.version
                }

                # If user didn't specify server name, use server's name
                if self.server_info.get('name'):
                    server_name = self.server_info['name']
                    if server_name != self.server_name:
                        logger.info(
                            f'Server name updated from "{self.server_name}" to "{server_name}"'
                        )
                        self.server_name = server_name

        except Exception as e:
            logger.warning(f'Failed to update server info: {e}')

    async def disconnect(self) -> None:
        """
        Disconnect from the MCP server and clean up resources.
        """
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Don't manually call session.close(), let AsyncExitStack handle it automatically
            if self.session:
                self.session = None

            if self.exit_stack:
                try:
                    await self.exit_stack.aclose()
                except Exception as e:
                    # Ignore cleanup errors, these are usually normal
                    logger.debug(f'Exit stack cleanup warning: {e}')
                finally:
                    self.exit_stack = None

        except Exception as e:
            logger.warning(f'Error during cleanup: {e}')
        finally:
            self.connected = False

    async def call_tool(self,
                        tool_name: str,
                        tool_args: Dict[str, Any],
                        timeout: Optional[timedelta] = None) -> str:
        """
        Execute a tool on the connected MCP server.

        Args:
            tool_name (str): Name of the tool to execute
            tool_args (Dict[str, Any]): Arguments to pass to the tool
            timeout (Optional[timedelta]): Custom timeout for this operation.
                Defaults to self.read_timeout (30 seconds).

        Returns:
            str: Tool execution result as text. If multiple text content blocks
                are returned, they are joined with double newlines.

        Raises:
            MCPConnectionError: If not connected to server or connection lost
            MCPToolExecutionError: If tool execution fails
            MCPTimeoutError: If operation times out

        Examples:
            >>> import asyncio
            >>> from datetime import timedelta
            >>> config = {"type": "sse", "url": "https://api.example.com/mcp/sse"}
            >>>
            >>> async def call_tool_examples():
            ...     async with MCPClient(config) as client:
            ...         # Simple tool call
            ...         result = await client.call_tool("search", {"query": "Python"})
            ...         print(f"Search result: {result}")
            ...
            ...         # Tool call with custom timeout
            ...         result = await client.call_tool(
            ...             "analyze_code",
            ...             {"code": "def hello(): print('world')", "language": "python"},
            ...             timeout=timedelta(seconds=60)
            ...         )
            ...         print(f"Analysis: {result}")
            ...
            ...         # Tool call with complex arguments
            ...         result = await client.call_tool(
            ...             "process_data",
            ...             {
            ...                 "data": [1, 2, 3, 4, 5],
            ...                 "operation": "sum",
            ...                 "options": {"precision": 2}
            ...             }
            ...         )
            ...         print(f"Process result: {result}")
            >>> asyncio.run(call_tool_examples())
        """
        if not self.connected or not self.session:
            raise MCPConnectionError(
                f'Not connected to server {self.server_name}')

        try:
            read_timeout = timeout or self.read_timeout

            result = await self.session.call_tool(
                tool_name, tool_args, read_timeout_seconds=read_timeout)

            # Extract text content
            texts = []
            for content in result.content:
                if content.type == 'text':
                    texts.append(content.text)

            if texts:
                return '\n\n'.join(texts)
            else:
                return 'execute error'

        except McpError as e:
            logger.error(
                f'MCP error calling tool {tool_name} on server {self.server_name}: {e}'
            )
            if e.error.code == CONNECTION_CLOSED:
                self.connected = False
                raise MCPConnectionError(
                    f'Connection lost while calling tool {tool_name}: {e.error.message}'
                ) from e
            else:
                raise MCPToolExecutionError(
                    f'Tool execution failed: {e.error.message}') from e

        except asyncio.TimeoutError:
            raise MCPTimeoutError(
                f'Tool call {tool_name} timed out after {timeout or self.read_timeout}'
            )

        except Exception as e:
            logger.error(
                f'Failed to call tool {tool_name} on server {self.server_name}: {e}'
            )
            raise MCPToolExecutionError(f'Tool execution failed: {e}') from e

    async def list_tools(self,
                         timeout: Optional[timedelta] = None) -> List[Tool]:
        """
        Retrieve list of available tools from the connected MCP server.

        Args:
            timeout (Optional[timedelta]): Custom timeout for this operation.
                Currently not used but reserved for future implementation.

        Returns:
            List[Tool]: List of Tool objects, each containing:
                - name (str): Tool name
                - description (str): Tool description
                - inputSchema (dict): JSON schema for tool arguments

        Raises:
            MCPConnectionError: If not connected to server
            Exception: If request fails for any other reason
        """
        if not self.connected:
            raise MCPConnectionError('Not connected to server')

        try:
            result = await self.session.list_tools()
            return result.tools

        except Exception as e:
            logger.error(f'Failed to get tools: {e}')
            raise

    def is_connected(self) -> bool:
        """Check if client is currently connected to the MCP server."""
        return self.connected

    def get_server_name(self) -> str:
        """Get the display name of the connected MCP server."""
        return self.server_name

    def get_transport_type(self) -> Optional[str]:
        """Get the transport type used for server communication."""
        return self.mcp_server.get('type')

    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed information about the connected MCP server."""
        return self.server_info

    async def __aenter__(self):
        """Async context manager entry point."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point."""
        await self.disconnect()

    def __del__(self):
        """Destructor"""
        try:
            # Only clean up references, don't perform async operations
            if hasattr(self, 'session'):
                self.session = None

            if hasattr(self, 'exit_stack'):
                self.exit_stack = None

            self.connected = False

        except Exception:
            # Cannot throw exceptions in destructor
            pass
