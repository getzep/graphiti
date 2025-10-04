#!/usr/bin/env python3
"""
Test script to verify the episode queue processing bug fix.

This script should be run after restarting the MCP server with the fix applied.
It tests that episodes are properly processed and stored in Neo4j.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path to import graphiti modules
sys.path.insert(0, str(Path(__file__).parent))

# Set up minimal environment variables for testing
os.environ.setdefault('NEO4J_URI', 'bolt://localhost:7687')
os.environ.setdefault('NEO4J_USER', 'neo4j')
os.environ.setdefault('NEO4J_PASSWORD', 'graphiti123!')
os.environ.setdefault('GOOGLE_API_KEY', os.environ.get('GOOGLE_API_KEY', ''))

from neo4j import GraphDatabase


async def check_neo4j_count():
    """Check the number of nodes in Neo4j."""
    uri = os.environ['NEO4J_URI']
    user = os.environ['NEO4J_USER']
    password = os.environ['NEO4J_PASSWORD']

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            record = result.single()
            count = record['count'] if record else 0
            return count
    finally:
        driver.close()


async def main():
    """Main test function."""
    print("=" * 60)
    print("Testing Episode Queue Processing Bug Fix")
    print("=" * 60)

    # Test 1: Check initial Neo4j state
    print("\n[Test 1] Checking initial Neo4j node count...")
    initial_count = await check_neo4j_count()
    print(f"Initial node count: {initial_count}")

    # Test 2: The actual episode addition should be done through MCP tools
    print("\n[Test 2] Add test episodes through MCP tools:")
    print("  Run these commands in your MCP client:")
    print("  mcp__graphiti__add_memory(")
    print("    name='Bug Fix Verification Test',")
    print("    episode_body='This is a test episode to verify the queue processing bug fix.',")
    print("    source='text'")
    print("  )")
    print("\n  Wait 5-10 seconds, then run:")
    print("  mcp__graphiti__search_memory_nodes(query='bug fix verification test')")

    # Test 3: Instructions for manual verification
    print("\n[Test 3] Manual verification steps:")
    print("  1. Restart the MCP server to load the fix")
    print("  2. Add a test episode using the MCP tool (see Test 2)")
    print("  3. Wait 5-10 seconds for processing")
    print("  4. Check Neo4j node count again - it should be > 0")
    print("  5. Search for the episode using search_memory_nodes")

    print("\n[Test 4] Check for queue worker logs:")
    print("  Look for this log message in the server output:")
    print("  'Starting episode queue worker for group_id: <group_id>'")
    print("  If you see this message, the worker is starting correctly.")

    print("\n[Test 5] Re-check Neo4j count after adding episode:")
    print("  Run this script again after adding test episodes to see if count increased.")
    current_count = await check_neo4j_count()
    print(f"Current node count: {current_count}")

    if current_count > initial_count:
        print("\n✓ SUCCESS: Node count increased! Episode processing is working.")
    else:
        print("\n✗ FAIL: Node count did not increase. Check:")
        print("  - Is the MCP server restarted with the fix?")
        print("  - Are episodes being added through MCP tools?")
        print("  - Check server logs for errors")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
