#!/usr/bin/env python3
"""
Example script demonstrating how to use the Graphiti MCP server with a custom Anthropic client.
This shows how to initialize the MCP server with an Anthropic client instead of the default OpenAI client.
"""

import asyncio
import os
from typing import Optional

from anthropic import AsyncAnthropic
from graphiti_mcp_server import initialize_graphiti
from mcp.server.fastmcp import FastMCP

from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig


async def setup_mcp_server_with_anthropic(
    anthropic_api_key: Optional[str] = None, model_name: Optional[str] = None
):
    """
    Initialize the Graphiti MCP server with an Anthropic client.

    Args:
        anthropic_api_key: API key for Anthropic. If not provided, will try to get from environment.
        model_name: Model name to use. If not provided, will use the default from AnthropicClient.
    """
    # Get API key from environment if not provided
    api_key = anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError(
            'Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable'
        )

    # Create LLM config
    llm_config = LLMConfig(api_key=api_key)
    if model_name:
        llm_config.model = model_name

    # Initialize Anthropic client
    anthropic_client = AnthropicClient(config=llm_config)

    # Initialize Graphiti with the Anthropic client
    await initialize_graphiti(anthropic_client=anthropic_client)

    # Create and run MCP server
    mcp = FastMCP('graphiti')
    mcp.run(transport='stdio')


async def main():
    """Main function to run the example."""
    try:
        # You can provide your API key directly or set it in the environment
        # os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

        # Optional: specify a model name (defaults to claude-3-5-sonnet-20240620 if not specified)
        # model_name = "claude-3-opus-20240229"

        await setup_mcp_server_with_anthropic()
    except Exception as e:
        print(f'Error initializing Graphiti MCP server with Anthropic: {str(e)}')
        raise


if __name__ == '__main__':
    asyncio.run(main())
