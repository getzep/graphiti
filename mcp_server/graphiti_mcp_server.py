#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Type

from mcp.server.fastmcp import FastMCP

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP('graphiti')

# Environment variables
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

MODEL_NAME = os.environ.get('MODEL_NAME')

# Available LLM client types
LLM_CLIENT_TYPES: Dict[str, Type[LLMClient]] = {
    'openai': OpenAIClient,
    'openai_generic': OpenAIGenericClient,
    'anthropic': AnthropicClient,
}

# Initialize Graphiti client
graphiti_client = None


async def initialize_graphiti(llm_client: Optional[LLMClient] = None):
    """Initialize the Graphiti client with the provided settings.

    Args:
        llm_client: Optional LLMClient instance to use for LLM operations
    """
    global graphiti_client

    # If no client is provided, create a default client (preferring Anthropic)
    if not llm_client:
        if ANTHROPIC_API_KEY:
            # Use Anthropic as default if API key is available
            config = LLMConfig(api_key=ANTHROPIC_API_KEY)
            if MODEL_NAME:
                config.model = MODEL_NAME
            llm_client = AnthropicClient(config=config)
            logger.info('Using Anthropic as default LLM client')
        elif OPENAI_API_KEY:
            # Fall back to OpenAI if Anthropic API key is not available
            config = LLMConfig(api_key=OPENAI_API_KEY)
            if OPENAI_BASE_URL:
                config.base_url = OPENAI_BASE_URL
            if MODEL_NAME:
                config.model = MODEL_NAME
            llm_client = OpenAIClient(config=config)
            logger.info('Using OpenAI as fallback LLM client')
        else:
            raise ValueError(
                'Either ANTHROPIC_API_KEY or OPENAI_API_KEY must be set when not using a custom LLM client'
            )

    if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
        raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

    graphiti_client = Graphiti(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        llm_client=llm_client,
    )

    # Initialize the graph database with Graphiti's indices
    await graphiti_client.build_indices_and_constraints()
    logger.info('Graphiti client initialized successfully')


def format_fact_result(edge: EntityEdge) -> dict:
    """Format an entity edge into a readable result.

    Since EntityEdge is a Pydantic BaseModel, we can use its built-in serialization capabilities.
    """
    # Convert to dict using Pydantic's model_dump method with mode='json'
    # This automatically handles datetime serialization and other complex types
    return edge.model_dump(
        mode='json',  # Properly handle datetime serialization for JSON
        exclude={
            'fact_embedding',  # Exclude embedding data
        },
    )


@mcp.tool()
async def add_episode(
    group_id: str,
    name: str,
    episode_body: str,
    source: str = 'text',
    source_description: str = '',
    uuid: Optional[str] = None,
) -> dict:
    """Add an episode to the Graphiti knowledge graph. This is the primary way to add information to the graph.

    Args:
        name: Name of the episode
        episode_body: The content of the episode
        source: Source type (text, message, etc.)
        source_description: Description of the source
        group_id: Optional group ID to organize episodes
        uuid: Optional UUID for the episode
    """
    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    # Map string source to EpisodeType enum
    source_type = EpisodeType.text
    if source.lower() == 'message':
        source_type = EpisodeType.message
    elif source.lower() == 'json':
        source_type = EpisodeType.json

    try:
        await graphiti_client.add_episode(
            name=name,
            episode_body=episode_body,
            source=source_type,
            source_description=source_description,
            group_id=group_id,
            uuid=uuid,
            reference_time=datetime.now(timezone.utc),
        )
        return {'message': f"Episode '{name}' added successfully"}
    except Exception as e:
        logger.error(f'Error adding episode: {str(e)}')
        return {'error': f'Error adding episode: {str(e)}'}


@mcp.tool()
async def search_nodes(
    query: str,
    group_ids: Optional[List[str]] = None,
    max_nodes: int = 10,
    center_node_uuid: Optional[str] = None,
) -> dict:
    """Search the Graphiti knowledge graph for relevant node summaries.
    These contain a summary of all of a node's relationships with other nodes.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_nodes: Maximum number of nodes to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        # Configure the search
        search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_nodes

        # Perform the search using the _search method
        search_results = await graphiti_client._search(
            query=query,
            config=search_config,
            group_ids=group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=SearchFilters(),
        )

        if not search_results.nodes:
            return {'message': 'No relevant nodes found', 'nodes': []}

        # Format the node results
        formatted_nodes = [
            {
                'uuid': node.uuid,
                'name': node.name,
                'summary': node.summary if hasattr(node, 'summary') else '',
                'labels': node.labels if hasattr(node, 'labels') else [],
                'group_id': node.group_id,
                'created_at': node.created_at.isoformat(),
                'attributes': node.attributes if hasattr(node, 'attributes') else {},
            }
            for node in search_results.nodes
        ]

        return {'message': 'Nodes retrieved successfully', 'nodes': formatted_nodes}
    except Exception as e:
        logger.error(f'Error searching nodes: {str(e)}')
        return {'error': f'Error searching nodes: {str(e)}'}


@mcp.tool()
async def search_facts(
    query: str,
    group_ids: Optional[List[str]] = None,
    max_facts: int = 10,
    center_node_uuid: Optional[str] = None,
) -> dict:
    """Search the Graphiti knowledge graph for relevant facts.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        relevant_edges = await graphiti_client.search(
            group_ids=group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return {'message': 'No relevant facts found', 'facts': []}

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return {'message': 'Facts retrieved successfully', 'facts': facts}
    except Exception as e:
        logger.error(f'Error searching: {str(e)}')
        return {'error': f'Error searching: {str(e)}'}


@mcp.tool()
async def delete_entity_edge(uuid: str) -> dict:
    """Delete an entity edge from the Graphiti knowledge graph.

    Args:
        uuid: UUID of the entity edge to delete
    """
    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(graphiti_client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(graphiti_client.driver)
        return {'message': f'Entity edge with UUID {uuid} deleted successfully'}
    except Exception as e:
        logger.error(f'Error deleting entity edge: {str(e)}')
        return {'error': f'Error deleting entity edge: {str(e)}'}


@mcp.tool()
async def delete_episode(uuid: str) -> dict:
    """Delete an episode from the Graphiti knowledge graph.

    Args:
        uuid: UUID of the episode to delete
    """
    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        # Import EpisodicNode class
        from graphiti_core.nodes import EpisodicNode

        # Get the episodic node by UUID
        episodic_node = await EpisodicNode.get_by_uuid(graphiti_client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(graphiti_client.driver)
        return {'message': f'Episode with UUID {uuid} deleted successfully'}
    except Exception as e:
        logger.error(f'Error deleting episode: {str(e)}')
        return {'error': f'Error deleting episode: {str(e)}'}


@mcp.tool()
async def get_entity_edge(uuid: str) -> dict:
    """Get an entity edge from the Graphiti knowledge graph by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(graphiti_client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        logger.error(f'Error getting entity edge: {str(e)}')
        return {'error': f'Error getting entity edge: {str(e)}'}


@mcp.tool()
async def get_episodes(group_id: str, last_n: int = 10) -> list[dict] | dict:
    """Get the most recent episodes for a specific group.

    Args:
        group_id: ID of the group to retrieve episodes from
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        episodes = await graphiti_client.retrieve_episodes(
            group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
        )

        if not episodes:
            return {'message': f'No episodes found for group {group_id}', 'episodes': []}

        # Use Pydantic's model_dump method for EpisodicNode serialization
        formatted_episodes = [
            # Use mode='json' to handle datetime serialization
            episode.model_dump(mode='json')
            for episode in episodes
        ]

        # Return the Python list directly - MCP will handle serialization
        return formatted_episodes
    except Exception as e:
        logger.error(f'Error getting episodes: {str(e)}')
        return {'error': f'Error getting episodes: {str(e)}'}


@mcp.tool()
async def clear_graph() -> dict:
    """Clear all data from the Graphiti knowledge graph and rebuild indices."""
    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data

        await clear_data(graphiti_client.driver)
        await graphiti_client.build_indices_and_constraints()
        return {'message': 'Graph cleared successfully and indices rebuilt'}
    except Exception as e:
        logger.error(f'Error clearing graph: {str(e)}')
        return {'error': f'Error clearing graph: {str(e)}'}


@mcp.resource('graphiti/status')
async def get_status() -> dict:
    """Get the status of the Graphiti MCP server and Neo4j connection."""
    if graphiti_client is None:
        return {'status': 'error', 'message': 'Graphiti client not initialized'}

    try:
        # Test Neo4j connection
        await graphiti_client.driver.verify_connectivity()
        return {'status': 'ok', 'message': 'Graphiti MCP server is running and connected to Neo4j'}
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Graphiti MCP server is running but Neo4j connection failed: {str(e)}',
        }


def create_llm_client(
    client_type: str, api_key: Optional[str] = None, model: Optional[str] = None
) -> LLMClient:
    """Create an LLM client of the specified type.

    Args:
        client_type: Type of LLM client to create ('openai',  'anthropic')
        api_key: API key for the LLM service
        model: Model name to use

    Returns:
        An instance of the specified LLM client
    """
    if client_type not in LLM_CLIENT_TYPES:
        raise ValueError(
            f"Unknown LLM client type: {client_type}. Available types: {', '.join(LLM_CLIENT_TYPES.keys())}"
        )

    # Create config with provided API key and model
    config = LLMConfig(api_key=api_key)

    # Set model if provided
    if model:
        config.model = model

    # Set base URL for OpenAI client if available
    if client_type == 'openai' and OPENAI_BASE_URL:
        config.base_url = OPENAI_BASE_URL

    # Create and return the client
    return LLM_CLIENT_TYPES[client_type](config=config)


async def main():
    """Main function to parse arguments and initialize the Graphiti MCP server."""
    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with optional LLM client'
    )
    parser.add_argument(
        '--llm-client',
        choices=list(LLM_CLIENT_TYPES.keys()),
        help='Type of LLM client to use (default: anthropic if available, otherwise openai)',
    )
    parser.add_argument('--model', help='Model name to use with the LLM client')

    args = parser.parse_args()

    try:
        llm_client = None

        # Create LLM client if specified
        if args.llm_client:
            # Get API key based on client type from environment variables
            api_key = None
            if args.llm_client == 'openai':
                api_key = OPENAI_API_KEY
            elif args.llm_client == 'anthropic':
                api_key = ANTHROPIC_API_KEY

            # Create the client
            llm_client = create_llm_client(
                client_type=args.llm_client, api_key=api_key, model=args.model
            )

        # Initialize Graphiti with the specified LLM client
        await initialize_graphiti(llm_client)

        # Run the server with stdio transport for MCP
        mcp.run(transport='stdio')
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise


if __name__ == '__main__':
    asyncio.run(main())
