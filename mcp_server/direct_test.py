#!/usr/bin/env python3
"""
Direct test of graphiti functionality bypassing MCP layer
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Set environment variables
os.environ['NEO4J_URI'] = 'bolt://localhost:7687'
os.environ['NEO4J_USER'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'graphiti123!'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC4YW25znj-zTc0BwEYmXf446XP8rNBFes'
os.environ['MODEL_NAME'] = 'gemini-2.5-pro'
os.environ['EMBEDDER_MODEL'] = 'models/text-embedding-004'

# Add parent directory to path to import graphiti_core
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Direct Graphiti Test (Bypassing MCP)")
print("=" * 60)

async def test_graphiti():
    try:
        from graphiti_core import Graphiti
        from graphiti_core.llm_client.config import LLMConfig

        print("\n[1] Creating LLM config...")
        llm_config = LLMConfig(
            api_type="google",
            api_key=os.environ['GOOGLE_API_KEY'],
            model_name=os.environ['MODEL_NAME']
        )

        print("[2] Creating LLM client...")
        llm_client = llm_config.create_client()
        print(f"    LLM client created: {llm_client is not None}")

        print("[3] Initializing Graphiti...")
        client = Graphiti(
            uri=os.environ['NEO4J_URI'],
            user=os.environ['NEO4J_USER'],
            password=os.environ['NEO4J_PASSWORD'],
            llm_client=llm_client,
        )

        print("[4] Building indices and constraints...")
        await client.build_indices_and_constraints()
        print("    Indices and constraints built successfully")

        print("\n[5] Adding episode...")
        episode_name = "Direct Test Episode"
        episode_body = "This is a direct test bypassing MCP to verify graphiti core functionality works correctly."

        await client.add_episode(
            name=episode_name,
            episode_body=episode_body,
            source='text',
            group_id='direct_test',
            reference_time=datetime.now(timezone.utc)
        )

        print(f"    Episode '{episode_name}' added successfully!")

        print("\n[6] Waiting 3 seconds for processing...")
        await asyncio.sleep(3)

        print("[7] Searching for added data...")
        results = await client.search(query="direct test", group_ids=['direct_test'])

        if results and len(results) > 0:
            print(f"    SUCCESS! Found {len(results)} result(s):")
            for i, result in enumerate(results[:3], 1):
                print(f"      {i}. {result}")
        else:
            print("    WARNING: No results found. Data may still be processing.")

        print("\n[8] Checking Neo4j database...")
        # Query Neo4j directly
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.environ['NEO4J_URI'],
            auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
        )

        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()['count']
            print(f"    Total nodes in Neo4j: {count}")

            if count > 0:
                print("    SUCCESS! Data is stored in Neo4j!")
            else:
                print("    WARNING: Neo4j database is still empty")

        driver.close()
        await client.close()

        print("\n" + "=" * 60)
        print("Test Complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

# Run the test
if __name__ == "__main__":
    asyncio.run(test_graphiti())
