import asyncio
import json
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.search import search_config_recipes, search_filters

FRAMES_PATH = os.path.join(os.path.dirname(__file__), '../data/glasses_frames_lg.json')

# Load environment variables for Neo4j connection
load_dotenv()
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

# Add this constant to your frames_runner.py
GROUP_ID = "graph_f445637a"  # The group ID used by MCP tools

def load_frames(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def ingest_frames(graphiti, frames):
    for frame in frames:
        await graphiti.add_episode(
            name=f"Frame {frame['frame_id']}",
            episode_body=json.dumps(frame),
            source=EpisodeType.json,
            source_description='Frame extracted from live video stream',
            reference_time=datetime.fromisoformat(frame['timestamp'].replace('Z', '+00:00')),
            group_id=GROUP_ID
        )
        print(f"Ingested frame {frame['frame_id']}")

async def display_communities(graphiti):
    print("\nDisplaying Community Information:")
    # Search for communities
    results = await graphiti._search(
        query="community", 
        config=search_config_recipes.NODE_HYBRID_SEARCH_RRF,
        group_ids=[GROUP_ID],
        search_filter=search_filters.SearchFilters(node_labels=["Community"])
    )
    
    if not results.communities:
        print("No communities found.")
        return
    
    for i, community in enumerate(results.communities):
        print(f"Community {i+1}: {community.name}")
        print(f"UUID: {community.uuid}")
        print(f"Summary: {community.summary}")
        
        # Get community members
        members_query = await graphiti._search(
            query="", 
            config=search_config_recipes.NODE_HYBRID_SEARCH_NODE_DISTANCE,
            group_ids=[GROUP_ID],
            center_node_uuid=community.uuid
        )
        
        if members_query.nodes:
            print("Members:")
            for node in members_query.nodes:
                print(f"  - {node.name} (UUID: {node.uuid})")
        
        print("---")

async def query_loop(graphiti):
    print("\nYou can now query your graph. Type 'exit' to quit.")
    print("Type 'communities' to display community information.")
    while True:
        query = input("Query> ").strip()
        if query.lower() in {'exit', 'quit'}:
            break
        if query.lower() == 'communities':
            await display_communities(graphiti)
            continue
        results = await graphiti.search(query)
        if not results:
            print("No results found.")
            continue
        print("\nResults:")
        for result in results:
            print(f"UUID: {getattr(result, 'uuid', None)}")
            print(f"Fact: {getattr(result, 'fact', None)}")
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f"Valid from: {result.valid_at}")
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f"Valid until: {result.invalid_at}")
            print('---')

async def main():
    graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    # Clear the graph before ingesting frames    
    await clear_data(graphiti.driver)
    print("Cleared the graph")
    await graphiti.build_indices_and_constraints()
    frames = load_frames(FRAMES_PATH)
    print(f"Loaded {len(frames)} frames from {FRAMES_PATH}")
    await ingest_frames(graphiti, frames)
    
    # Build communities after ingesting all frames
    print("\nBuilding communities...")
    community_nodes = await graphiti.build_communities()
    print(f"Built {len(community_nodes)} communities")
    
    await query_loop(graphiti)

if __name__ == '__main__':
    asyncio.run(main()) 