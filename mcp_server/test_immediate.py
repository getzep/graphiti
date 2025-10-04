#!/usr/bin/env python3
"""
Test graphiti with immediate processing to see errors
"""

import asyncio
import os
import sys
from pathlib import Path

# Set environment
os.environ['NEO4J_URI'] = 'bolt://localhost:7687'
os.environ['NEO4J_USER'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'graphiti123!'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC4YW25znj-zTc0BwEYmXf446XP8rNBFes'
os.environ['MODEL_NAME'] = 'gemini-2.5-pro'

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test():
    print("Testing immediate episode processing...")

    # Import after setting path
    from graphiti_core import Graphiti
    from graphiti_core.llm_client.config import LLMConfig
    from datetime import datetime, timezone

    try:
        # Create LLM config
        print("1. Creating LLM config...")
        llm_config = LLMConfig(
            api_type="google",
            api_key=os.environ['GOOGLE_API_KEY'],
            model_name=os.environ['MODEL_NAME']
        )

        # Create client
        print("2. Creating LLM client...")
        llm_client = llm_config.create_client()

        if llm_client is None:
            print("ERROR: LLM client is None!")
            return

        print(f"   LLM client type: {type(llm_client)}")

        # Initialize Graphiti
        print("3. Initializing Graphiti...")
        client = Graphiti(
            uri=os.environ['NEO4J_URI'],
            user=os.environ['NEO4J_USER'],
            password=os.environ['NEO4J_PASSWORD'],
            llm_client=llm_client,
        )

        print("4. Building indices...")
        await client.build_indices_and_constraints()

        # Add episode
        print("5. Adding episode (this will process immediately)...")
        await client.add_episode(
            name="Immediate Test",
            episode_body="Testing immediate processing to catch any errors",
            source='text',
            group_id='test_immediate',
            reference_time=datetime.now(timezone.utc)
        )

        print("6. SUCCESS! Episode processed without errors")

        # Check database
        print("7. Checking database...")
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ['NEO4J_URI'],
            auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
        )

        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()['count']
            print(f"   Total nodes in database: {count}")

        driver.close()
        await client.close()

    except Exception as e:
        print(f"\nERROR during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
