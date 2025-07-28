# Copyright (c) Alibaba, Inc. and its affiliates.
"""
ModelScope MCP (Model Control Protocol) Package

- MCPApi: API client for ModelScope Hub MCP services
- MCP: Unified interface for multi-server tool execution
"""

from .api import MCPApi
from .manager import MCP

__all__ = ['MCPApi', 'MCP']
