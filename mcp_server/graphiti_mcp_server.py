#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()


class MCPConfig(BaseModel):
    transport: str


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Welcome to Graphiti MCP - a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to 
capture relationships between concepts, entities, and information. The system organizes data as episodes 
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic, 
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including 
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid 
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_episode tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations. 
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality. 
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid 
API keys are provided for any language model operations.
"""

# Initialize FastMCP server
mcp = FastMCP(
    'graphiti',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)

# Environment variables
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL')

MODEL_NAME = os.environ.get('MODEL_NAME')

# Available LLM client types
LLM_CLIENT_TYPES: Dict[str, Type[LLMClient]] = {
    'openai': OpenAIClient,
    'openai_generic': OpenAIGenericClient,
}

# Initialize Graphiti client and default group_id
graphiti_client = None
default_group_id = None


async def initialize_graphiti(llm_client: Optional[LLMClient] = None, destroy_graph: bool = False):
    """Initialize the Graphiti client with the provided settings.

    Args:
        llm_client: Optional LLMClient instance to use for LLM operations
        destroy_graph: Optional boolean to destroy all Graphiti graphs
    """
    global graphiti_client

    # If no client is provided, create a default OpenAI client
    if not llm_client:
        if OPENAI_API_KEY:
            config = LLMConfig(api_key=OPENAI_API_KEY)
            if OPENAI_BASE_URL:
                config.base_url = OPENAI_BASE_URL
            if MODEL_NAME:
                config.model = MODEL_NAME
            llm_client = OpenAIClient(config=config)
            logger.info('Using OpenAI as LLM client')
        else:
            raise ValueError('OPENAI_API_KEY must be set when not using a custom LLM client')

    if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
        raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

    graphiti_client = Graphiti(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        llm_client=llm_client,
    )

    if destroy_graph:
        logger.info('Destroying graph...')
        await clear_data(graphiti_client.driver)

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
    name: str,
    episode_body: str,
    group_id: Optional[str] = None,
    source: str = 'text',
    source_description: str = '',
    uuid: Optional[str] = None,
) -> dict:
    """Add an episode to the Graphiti knowledge graph. This is the primary way to add information to the graph.

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode. When source='json', this must be a properly escaped JSON string,
                           not a raw Python dictionary. The JSON data will be automatically processed
                           to extract entities and relationships.
        group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default)
                               - 'json': For structured data
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode

    Examples:
        # Adding plain text content
        add_episode(
            name="Company News",
            episode_body="Acme Corp announced a new product line today.",
            source="text",
            source_description="news article",
            group_id="some_arbitrary_string"
        )

        # Adding structured JSON data
        # NOTE: episode_body must be a properly escaped JSON string
        add_episode(
            name="Customer Profile",
            episode_body='\{"name": "John Smith", "company": "Acme Corp", "role": "CEO"\}',
            source="json",
            source_description="CRM data"
        )

        # Adding more complex JSON with arrays and nested objects
        add_episode(
            name="Product Catalog",
            episode_body='\{"catalog": \{"company": "Tech Solutions Inc.", "products": [\{"id": "P001", "name": "Product X", "features": ["Feature A", "Feature B"]\}]\}\}',
            source="json",
            source_description="Product catalog"
        )

        # Adding message-style content
        add_episode(
            name="Customer Conversation",
            episode_body="user: What's your return policy?\nassistant: You can return items within 30 days.",
            source="message",
            source_description="chat transcript",
            group_id="some_arbitrary_string"
        )

    Notes:
        When using source='json':
        - The JSON must be a properly escaped string, not a raw Python dictionary
        - The JSON will be automatically processed to extract entities and relationships
        - Complex nested structures are supported (arrays, nested objects, mixed data types), but keep nesting to a minimum
        - Entities will be created from appropriate JSON properties
        - Relationships between entities will be established based on the JSON structure
    """
    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    # Map string source to EpisodeType enum
    source_type = EpisodeType.text
    if source.lower() == 'message':
        source_type = EpisodeType.message
    elif source.lower() == 'json':
        source_type = EpisodeType.json

    # Use the provided group_id or fall back to the default
    effective_group_id = group_id if group_id is not None else default_group_id

    try:
        await graphiti_client.add_episode(
            name=name,
            episode_body=episode_body,
            source=source_type,
            source_description=source_description,
            group_id=effective_group_id,
            uuid=uuid,
            reference_time=datetime.now(timezone.utc),
        )
        return {'message': f"Episode '{name}' added successfully"}
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error adding episode: {error_msg}')

        return {'error': f'Error adding episode: {error_msg}'}


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

    # Use the provided group_ids or fall back to the default if none provided
    effective_group_ids = group_ids if group_ids is not None else [default_group_id]

    try:
        # Configure the search
        search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_nodes

        # Perform the search using the _search method
        search_results = await graphiti_client._search(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
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

    # Use the provided group_ids or fall back to the default if none provided
    effective_group_ids = group_ids if group_ids is not None else [default_group_id]

    try:
        relevant_edges = await graphiti_client.search(
            group_ids=effective_group_ids,
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
async def get_episodes(group_id: Optional[str] = None, last_n: int = 10) -> list[dict] | dict:
    """Get the most recent episodes for a specific group.

    Args:
        group_id: ID of the group to retrieve episodes from. If not provided, uses the default group_id.
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    # Use the provided group_id or fall back to the default
    effective_group_id = group_id if group_id is not None else default_group_id

    if not isinstance(effective_group_id, str):
        return {'error': 'Group ID must be a string'}

    try:
        episodes = await graphiti_client.retrieve_episodes(
            group_ids=[effective_group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
        )

        if not episodes:
            return {'message': f'No episodes found for group {effective_group_id}', 'episodes': []}

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


@mcp.resource('http://graphiti/status')
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
    client_type: str = 'openai', api_key: Optional[str] = None, model: Optional[str] = None
) -> LLMClient:
    """Create an OpenAI LLM client.

    Args:
        client_type: Type of LLM client to create (only 'openai' or 'openai_generic' supported)
        api_key: API key for the OpenAI service
        model: Model name to use

    Returns:
        An instance of the OpenAI LLM client
    """
    if client_type not in LLM_CLIENT_TYPES:
        raise ValueError(
            f"Unsupported LLM client type: {client_type}. Only OpenAI clients are supported: {', '.join(LLM_CLIENT_TYPES.keys())}"
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


async def initialize_server() -> MCPConfig:
    """Initialize the Graphiti server with the specified LLM client."""
    global default_group_id

    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with optional LLM client'
    )
    parser.add_argument(
        '--group-id',
        help='Namespace for the graph. This is an arbitrary string used to organize related data. '
        'If not provided, a random UUID will be generated.',
    )
    parser.add_argument(
        '--transport',
        choices=['sse', 'stdio'],
        default='sse',
        help='Transport to use for communication with the client. (default: sse)',
    )
    # OpenAI is the only supported LLM client
    parser.add_argument('--model', help='Model name to use with the LLM client')
    parser.add_argument('--destroy-graph', action='store_true', help='Destroy all Graphiti graphs')

    args = parser.parse_args()

    # Set the default group_id from CLI argument or generate a random one
    if args.group_id:
        default_group_id = args.group_id
        logger.info(f'Using provided group_id: {default_group_id}')
    else:
        default_group_id = f'graph_{uuid.uuid4().hex[:8]}'
        logger.info(f'Generated random group_id: {default_group_id}')

    llm_client = None

    # Create OpenAI client if model is specified or if OPENAI_API_KEY is available
    if args.model or OPENAI_API_KEY:
        # Create the OpenAI client
        llm_client = create_llm_client(
            client_type='openai', api_key=OPENAI_API_KEY, model=args.model
        )

    # Initialize Graphiti with the specified LLM client
    await initialize_graphiti(llm_client, destroy_graph=args.destroy_graph)

    return MCPConfig(transport=args.transport)


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with stdio transport for MCP in the same event loop
    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'sse':
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        await mcp.run_sse_async()


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise


if __name__ == '__main__':
    main()
