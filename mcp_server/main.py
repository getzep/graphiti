#!/usr/bin/env python3
"""
Main entry point for Graphiti MCP Server

This is a backwards-compatible wrapper around the original graphiti_mcp_server.py
to maintain compatibility with existing deployment scripts and documentation.

Usage:
    python main.py [args...]

All arguments are passed through to the original server implementation.
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
mcp_server_dir = Path(__file__).parent
src_path = mcp_server_dir / 'src'
repo_root = mcp_server_dir.parent
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(repo_root))

# Import and run the original server
if __name__ == '__main__':
    from graphiti_mcp_server import main

    # Pass all command line arguments to the original main function
    main()
