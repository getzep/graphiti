#!/usr/bin/env python3
"""
Example script demonstrating how to use the Graphiti MCP server with Anthropic client.
"""

import asyncio
import os
import sys

# Add parent directory to path to import from graphiti_mcp_server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphiti_mcp_server import initialize_graphiti, mcp

from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig


async def main():
    """Run the Graphiti MCP server with Anthropic client."""
    # Get API key from environment
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('Error: ANTHROPIC_API_KEY environment variable not set')
        print('Please set it with: export ANTHROPIC_API_KEY=your_api_key')
        sys.exit(1)

    # Create Anthropic client config
    config = LLMConfig(
        api_key=api_key,
        model='claude-3-5-sonnet-20240620',  # You can change this to any supported Anthropic model
    )

    # Create Anthropic client
    anthropic_client = AnthropicClient(config=config)

    print('Initializing Graphiti with Anthropic client...')
    # Initialize Graphiti with Anthropic client
    await initialize_graphiti(llm_client=anthropic_client)

    print('Starting MCP server with Anthropic client...')
    # Run the MCP server
    mcp.run(transport='stdio')


if __name__ == '__main__':
    asyncio.run(main())
