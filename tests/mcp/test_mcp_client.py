# Copyright (c) Alibaba, Inc. and its affiliates.
# !/usr/bin/env python3
"""
Unit tests for MCPClient
"""

import asyncio
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from hub.mcp.client import (MCPClient, MCPClientError, MCPConnectionError,
                            MCPTimeoutError, MCPToolExecutionError)

# Add modelscope module directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'modelscope'))


class TestMCPClient(unittest.TestCase):
    """Test cases for MCPClient class"""

    def setUp(self):
        """Set up test fixtures"""
        self.valid_sse_config = {
            'type': 'sse',
            'url': 'https://example.com/sse'
        }

        self.valid_stdio_config = {
            'type': 'stdio',
            'command': ['test-server', '--config', 'test.json']
        }

        self.valid_streamable_http_config = {
            'type': 'streamable_http',
            'url': 'https://example.com/stream'
        }

        self.nested_config = {
            'mcpServers': {
                'test-server': {
                    'type': 'sse',
                    'url': 'https://example.com/sse'
                }
            }
        }

    def test_init_with_valid_config(self):
        """Test client initialization with valid configuration"""
        client = MCPClient(self.valid_sse_config)

        self.assertEqual(client.mcp_server, self.valid_sse_config)
        self.assertIsNone(client.session)
        self.assertIsNone(client.exit_stack)
        self.assertFalse(client.connected)
        self.assertEqual(client.server_name, 'sse-example')

    def test_init_with_empty_config(self):
        """Test client initialization with empty configuration"""
        with self.assertRaises(ValueError) as context:
            MCPClient({})

        self.assertIn('MCP server configuration is required',
                      str(context.exception))

    def test_init_with_none_config(self):
        """Test client initialization with None configuration"""
        with self.assertRaises(ValueError) as context:
            MCPClient(None)  # type: ignore

        self.assertIn('MCP server configuration is required',
                      str(context.exception))

    def test_init_with_nested_config(self):
        """Test client initialization with nested mcpServers configuration"""
        client = MCPClient(self.nested_config)

        # Should extract the first server configuration
        expected_config = self.nested_config['mcpServers']['test-server']
        self.assertEqual(client.mcp_server, expected_config)

    def test_validate_config_missing_type(self):
        """Test configuration validation with missing type"""
        invalid_config = {'url': 'https://example.com'}

        with self.assertRaises(ValueError) as context:
            MCPClient(invalid_config)

        self.assertIn('Server type is required', str(context.exception))

    def test_validate_config_missing_url_and_command(self):
        """Test configuration validation with missing URL and command"""
        invalid_config = {'type': 'sse'}

        with self.assertRaises(ValueError) as context:
            MCPClient(invalid_config)

        self.assertIn('Server URL or command is required',
                      str(context.exception))

    def test_validate_config_unsupported_transport(self):
        """Test configuration validation with unsupported transport type"""
        invalid_config = {'type': 'unsupported', 'url': 'https://example.com'}

        with self.assertRaises(ValueError) as context:
            MCPClient(invalid_config)

        self.assertIn('Unsupported transport type', str(context.exception))

    def test_generate_server_name_stdio(self):
        """Test server name generation for STDIO transport"""
        client = MCPClient(self.valid_stdio_config)

        # Should extract command name from list
        self.assertEqual(client.server_name, 'stdio-test-server')

    def test_generate_server_name_stdio_string(self):
        """Test server name generation for STDIO transport with string command"""
        config = {'type': 'stdio', 'command': 'test-server'}
        client = MCPClient(config)

        self.assertEqual(client.server_name, 'stdio-test-server')

    def test_generate_server_name_sse(self):
        """Test server name generation for SSE transport"""
        client = MCPClient(self.valid_sse_config)

        # Should extract domain from URL
        self.assertEqual(client.server_name, 'sse-example')

    def test_generate_server_name_streamable_http(self):
        """Test server name generation for Streamable HTTP transport"""
        client = MCPClient(self.valid_streamable_http_config)

        self.assertEqual(client.server_name, 'streamable_http-example')

    def test_generate_server_name_fallback(self):
        """Test server name generation fallback"""
        config = {'type': 'sse', 'url': 'invalid-url'}
        client = MCPClient(config)

        self.assertEqual(client.server_name, 'sse-server')

    def test_get_transport_type(self):
        """Test getting transport type"""
        client = MCPClient(self.valid_sse_config)

        self.assertEqual(client.get_transport_type(), 'sse')

    def test_is_connected_initial_state(self):
        """Test initial connection state"""
        client = MCPClient(self.valid_sse_config)

        self.assertFalse(client.is_connected())

    def test_get_server_name(self):
        """Test getting server name"""
        client = MCPClient(self.valid_sse_config)

        self.assertEqual(client.get_server_name(), 'sse-example')

    def test_get_server_info_initial_state(self):
        """Test getting server info in initial state"""
        client = MCPClient(self.valid_sse_config)

        self.assertIsNone(client.get_server_info())


class TestMCPClientConnection(unittest.IsolatedAsyncioTestCase):
    """Test cases for MCPClient connection functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.valid_sse_config = {
            'type': 'sse',
            'url': 'https://example.com/sse'
        }

    @patch('hub.mcp.client.sse_client')
    @patch('hub.mcp.client.ClientSession')
    async def test_connect_success(self, mock_client_session, mock_sse_client):
        """Test successful connection"""
        # Mock the SSE client
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_sse_client.return_value.__aenter__.return_value = (mock_read,
                                                                mock_write)

        # Mock the client session
        mock_session = AsyncMock()
        mock_session.initialize.return_value = Mock()
        mock_client_session.return_value.__aenter__.return_value = mock_session

        client = MCPClient(self.valid_sse_config)

        await client.connect()

        self.assertTrue(client.is_connected())
        self.assertIsNotNone(client.session)

    @patch('hub.mcp.client.sse_client')
    async def test_connect_failure(self, mock_sse_client):
        """Test connection failure"""
        # Mock SSE client to raise exception
        mock_sse_client.side_effect = Exception('Connection failed')

        client = MCPClient(self.valid_sse_config)

        with self.assertRaises(MCPConnectionError):
            await client.connect()

        self.assertFalse(client.is_connected())

    async def test_connect_already_connected(self):
        """Test connecting when already connected"""
        client = MCPClient(self.valid_sse_config)
        client.connected = True

        # Should not raise exception, just return
        await client.connect()

        self.assertTrue(client.is_connected())

    @patch('hub.mcp.client.stdio_client')
    @patch('hub.mcp.client.ClientSession')
    async def test_connect_stdio(self, mock_client_session, mock_stdio_client):
        """Test STDIO connection"""
        config = {'type': 'stdio', 'command': 'test-server'}

        # Mock the STDIO client
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio_client.return_value.__aenter__.return_value = (mock_read,
                                                                  mock_write)

        # Mock the client session
        mock_session = AsyncMock()
        mock_session.initialize.return_value = Mock()
        mock_client_session.return_value.__aenter__.return_value = mock_session

        client = MCPClient(config)

        await client.connect()

        self.assertTrue(client.is_connected())

    async def test_disconnect(self):
        """Test disconnection"""
        client = MCPClient(self.valid_sse_config)
        client.connected = True
        client.session = AsyncMock()
        client.exit_stack = AsyncMock()

        await client.disconnect()

        self.assertFalse(client.is_connected())
        self.assertIsNone(client.session)

    async def test_context_manager(self):
        """Test context manager functionality"""
        with patch('hub.mcp.client.sse_client') as mock_sse_client, \
             patch('hub.mcp.client.ClientSession') as mock_client_session:

            # Mock the SSE client
            mock_read = AsyncMock()
            mock_write = AsyncMock()
            mock_sse_client.return_value.__aenter__.return_value = (mock_read,
                                                                    mock_write)

            # Mock the client session
            mock_session = AsyncMock()
            mock_session.initialize.return_value = Mock()
            mock_client_session.return_value.__aenter__.return_value = mock_session

            async with MCPClient(self.valid_sse_config) as client:
                self.assertTrue(client.is_connected())

            # Should be disconnected after context exit
            self.assertFalse(client.is_connected())


class TestMCPClientTools(unittest.IsolatedAsyncioTestCase):
    """Test cases for MCPClient tool functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.valid_sse_config = {
            'type': 'sse',
            'url': 'https://example.com/sse'
        }

    @patch('hub.mcp.client.sse_client')
    @patch('hub.mcp.client.ClientSession')
    async def test_list_tools_success(self, mock_client_session,
                                      mock_sse_client):
        """Test successful tool listing"""
        # Mock the SSE client
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_sse_client.return_value.__aenter__.return_value = (mock_read,
                                                                mock_write)

        # Mock the client session
        mock_session = AsyncMock()
        mock_session.initialize.return_value = Mock()
        mock_session.list_tools.return_value = Mock(tools=['tool1', 'tool2'])
        mock_client_session.return_value.__aenter__.return_value = mock_session

        client = MCPClient(self.valid_sse_config)
        await client.connect()

        tools = await client.list_tools()

        self.assertEqual(tools, ['tool1', 'tool2'])

    async def test_list_tools_not_connected(self):
        """Test tool listing when not connected"""
        client = MCPClient(self.valid_sse_config)

        with self.assertRaises(MCPConnectionError):
            await client.list_tools()

    @patch('hub.mcp.client.sse_client')
    @patch('hub.mcp.client.ClientSession')
    async def test_call_tool_success(self, mock_client_session,
                                     mock_sse_client):
        """Test successful tool execution"""
        # Mock the SSE client
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_sse_client.return_value.__aenter__.return_value = (mock_read,
                                                                mock_write)

        # Mock the client session
        mock_session = AsyncMock()
        mock_session.initialize.return_value = Mock()

        # Mock tool result
        mock_content = Mock()
        mock_content.type = 'text'
        mock_content.text = 'Tool execution result'
        mock_result = Mock()
        mock_result.content = [mock_content]
        mock_session.call_tool.return_value = mock_result
        mock_client_session.return_value.__aenter__.return_value = mock_session

        client = MCPClient(self.valid_sse_config)
        await client.connect()

        result = await client.call_tool('test_tool', {'param': 'value'})

        self.assertEqual(result, 'Tool execution result')

    @patch('hub.mcp.client.sse_client')
    @patch('hub.mcp.client.ClientSession')
    async def test_call_tool_no_text_content(self, mock_client_session,
                                             mock_sse_client):
        """Test tool execution with no text content"""
        # Mock the SSE client
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_sse_client.return_value.__aenter__.return_value = (mock_read,
                                                                mock_write)

        # Mock the client session
        mock_session = AsyncMock()
        mock_session.initialize.return_value = Mock()

        # Mock tool result with no text content
        mock_result = Mock()
        mock_result.content = []
        mock_session.call_tool.return_value = mock_result
        mock_client_session.return_value.__aenter__.return_value = mock_session

        client = MCPClient(self.valid_sse_config)
        await client.connect()

        result = await client.call_tool('test_tool', {'param': 'value'})

        self.assertEqual(result, 'execute error')

    async def test_call_tool_not_connected(self):
        """Test tool execution when not connected"""
        client = MCPClient(self.valid_sse_config)

        with self.assertRaises(MCPConnectionError):
            await client.call_tool('test_tool', {'param': 'value'})

    @patch('hub.mcp.client.sse_client')
    @patch('hub.mcp.client.ClientSession')
    async def test_call_tool_timeout(self, mock_client_session,
                                     mock_sse_client):
        """Test tool execution timeout"""
        # Mock the SSE client
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_sse_client.return_value.__aenter__.return_value = (mock_read,
                                                                mock_write)

        # Mock the client session
        mock_session = AsyncMock()
        mock_session.initialize.return_value = Mock()
        mock_session.call_tool.side_effect = asyncio.TimeoutError()
        mock_client_session.return_value.__aenter__.return_value = mock_session

        client = MCPClient(self.valid_sse_config)
        await client.connect()

        with self.assertRaises(MCPTimeoutError):
            await client.call_tool('test_tool', {'param': 'value'})


class TestMCPClientExceptions(unittest.TestCase):
    """Test cases for MCPClient exceptions"""

    def test_mcp_client_error_inheritance(self):
        """Test exception inheritance hierarchy"""
        self.assertTrue(issubclass(MCPConnectionError, MCPClientError))
        self.assertTrue(issubclass(MCPToolExecutionError, MCPClientError))
        self.assertTrue(issubclass(MCPTimeoutError, MCPClientError))

    def test_exception_messages(self):
        """Test exception message creation"""
        connection_error = MCPConnectionError('Connection failed')
        self.assertEqual(str(connection_error), 'Connection failed')

        tool_error = MCPToolExecutionError('Tool execution failed')
        self.assertEqual(str(tool_error), 'Tool execution failed')

        timeout_error = MCPTimeoutError('Operation timed out')
        self.assertEqual(str(timeout_error), 'Operation timed out')


class TestMCPClientIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration test cases for MCPClient"""

    def setUp(self):
        """Set up test fixtures"""
        self.valid_sse_config = {
            'type': 'sse',
            'url': 'https://example.com/sse'
        }

    @patch('hub.mcp.client.sse_client')
    @patch('hub.mcp.client.ClientSession')
    async def test_full_workflow(self, mock_client_session, mock_sse_client):
        """Test complete client workflow"""
        # Mock the SSE client
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_sse_client.return_value.__aenter__.return_value = (mock_read,
                                                                mock_write)

        # Mock the client session
        mock_session = AsyncMock()
        mock_session.initialize.return_value = Mock()

        # Mock tool listing
        mock_session.list_tools.return_value = Mock(tools=['tool1', 'tool2'])

        # Mock tool execution
        mock_content = Mock()
        mock_content.type = 'text'
        mock_content.text = 'Success'
        mock_result = Mock()
        mock_result.content = [mock_content]
        mock_session.call_tool.return_value = mock_result
        mock_client_session.return_value.__aenter__.return_value = mock_session

        # Test complete workflow
        async with MCPClient(self.valid_sse_config) as client:
            # Check connection
            self.assertTrue(client.is_connected())
            self.assertEqual(client.get_transport_type(), 'sse')

            # The server name should be updated after connection
            self.assertNotEqual(client.get_server_name(), 'sse-example')

            # List tools
            tools = await client.list_tools()
            self.assertEqual(len(tools), 2)

            # Call tool
            result = await client.call_tool('tool1', {'param': 'value'})
            self.assertEqual(result, 'Success')

        # Check disconnection
        self.assertFalse(client.is_connected())


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
