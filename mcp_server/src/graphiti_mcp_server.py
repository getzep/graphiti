#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from starlette.responses import JSONResponse

from .config.schema import GraphitiConfig, ServerConfig
from .models.response_types import (
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeResult,
    NodeSearchResponse,
    StatusResponse,
    SuccessResponse,
)
from .services.factories import DatabaseDriverFactory, EmbedderFactory, LLMClientFactory
from .services.queue_service import QueueService
from .utils.formatting import format_fact_result

# Load .env file from mcp_server directory
mcp_server_dir = Path(__file__).parent.parent
env_file = mcp_server_dir / '.env'
if env_file.exists():
    load_dotenv(env_file)
else:
    # Try current working directory as fallback
    load_dotenv()


# Semaphore limit for concurrent Graphiti operations.
#
# This controls how many episodes can be processed simultaneously. Each episode
# processing involves multiple LLM calls (entity extraction, deduplication, etc.),
# so the actual number of concurrent LLM requests will be higher.
#
# TUNING GUIDELINES:
#
# LLM Provider Rate Limits (requests per minute):
# - OpenAI Tier 1 (free):     3 RPM   -> SEMAPHORE_LIMIT=1-2
# - OpenAI Tier 2:            60 RPM   -> SEMAPHORE_LIMIT=5-8
# - OpenAI Tier 3:           500 RPM   -> SEMAPHORE_LIMIT=10-15
# - OpenAI Tier 4:         5,000 RPM   -> SEMAPHORE_LIMIT=20-50
# - Anthropic (default):     50 RPM   -> SEMAPHORE_LIMIT=5-8
# - Anthropic (high tier): 1,000 RPM   -> SEMAPHORE_LIMIT=15-30
# - Azure OpenAI (varies):  Consult your quota -> adjust accordingly
#
# SYMPTOMS:
# - Too high: 429 rate limit errors, increased costs from parallel processing
# - Too low: Slow throughput, underutilized API quota
#
# MONITORING:
# - Watch logs for rate limit errors (429)
# - Monitor episode processing times
# - Check LLM provider dashboard for actual request rates
#
# DEFAULT: 10 (suitable for OpenAI Tier 3, mid-tier Anthropic)
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 10))


# Configure structured logging with timestamps
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stderr,
)

# Configure specific loggers
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)  # Reduce access log noise
logging.getLogger('mcp.server.streamable_http_manager').setLevel(
    logging.WARNING
)  # Reduce MCP noise


# Patch uvicorn's logging config to use our format
def configure_uvicorn_logging():
    """Configure uvicorn loggers to match our format after they're created."""
    for logger_name in ['uvicorn', 'uvicorn.error', 'uvicorn.access']:
        uvicorn_logger = logging.getLogger(logger_name)
        # Remove existing handlers and add our own with proper formatting
        uvicorn_logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.propagate = False


logger = logging.getLogger(__name__)

# Create global config instance - will be properly initialized later
config: GraphitiConfig

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to 
capture relationships between concepts, entities, and information. The system organizes data as episodes 
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic, 
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including 
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid 
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
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

# MCP server instance
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)

# Global services
graphiti_service: Optional['GraphitiService'] = None
queue_service: QueueService | None = None

# Global client for backward compatibility
graphiti_client: Graphiti | None = None
semaphore: asyncio.Semaphore


class GraphitiService:
    """Graphiti service using the unified configuration system."""

    def __init__(self, config: GraphitiConfig, semaphore_limit: int = 10):
        self.config = config
        self.semaphore_limit = semaphore_limit
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.client: Graphiti | None = None
        self.entity_types = None

    async def initialize(self) -> None:
        """Initialize the Graphiti client with factory-created components."""
        try:
            # Create clients using factories
            llm_client = None
            embedder_client = None

            # Create LLM client based on configured provider
            try:
                llm_client = LLMClientFactory.create(self.config.llm)
            except Exception as e:
                logger.warning(f'Failed to create LLM client: {e}')

            # Create embedder client based on configured provider
            try:
                embedder_client = EmbedderFactory.create(self.config.embedder)
            except Exception as e:
                logger.warning(f'Failed to create embedder client: {e}')

            # Get database configuration
            db_config = DatabaseDriverFactory.create_config(self.config.database)

            # Build entity types from configuration
            custom_types = None
            if self.config.graphiti.entity_types:
                custom_types = {}
                for entity_type in self.config.graphiti.entity_types:
                    # Create a dynamic Pydantic model for each entity type
                    # Note: Don't use 'name' as it's a protected Pydantic attribute
                    entity_model = type(
                        entity_type.name,
                        (BaseModel,),
                        {
                            '__doc__': entity_type.description,
                        },
                    )
                    custom_types[entity_type.name] = entity_model

            # Store entity types for later use
            self.entity_types = custom_types

            # Initialize Graphiti client with appropriate driver
            try:
                if self.config.database.provider.lower() == 'falkordb':
                    # For FalkorDB, create a FalkorDriver instance directly
                    from graphiti_core.driver.falkordb_driver import FalkorDriver

                    falkor_driver = FalkorDriver(
                        host=db_config['host'],
                        port=db_config['port'],
                        password=db_config['password'],
                        database=db_config['database'],
                    )

                    self.client = Graphiti(
                        graph_driver=falkor_driver,
                        llm_client=llm_client,
                        embedder=embedder_client,
                        max_coroutines=self.semaphore_limit,
                    )
                else:
                    # For Neo4j (default), create a Neo4jDriver instance with database parameter
                    neo4j_driver = Neo4jDriver(
                        uri=db_config['uri'],
                        user=db_config['user'],
                        password=db_config['password'],
                        database=db_config.get('database', 'neo4j'),
                    )

                    self.client = Graphiti(
                        graph_driver=neo4j_driver,
                        llm_client=llm_client,
                        embedder=embedder_client,
                        max_coroutines=self.semaphore_limit,
                    )
            except Exception as db_error:
                # Check for connection errors
                error_msg = str(db_error).lower()
                if 'connection refused' in error_msg or 'could not connect' in error_msg:
                    db_provider = self.config.database.provider
                    if db_provider.lower() == 'falkordb':
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: FalkorDB is not running\n'
                            f'{"=" * 70}\n\n'
                            f'FalkorDB at {db_config["host"]}:{db_config["port"]} is not accessible.\n\n'
                            f'To start FalkorDB:\n'
                            f'  - Using Docker Compose: cd mcp_server && docker compose up\n'
                            f'  - Or run FalkorDB manually: docker run -p 6379:6379 falkordb/falkordb\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                    elif db_provider.lower() == 'neo4j':
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: Neo4j is not running\n'
                            f'{"=" * 70}\n\n'
                            f'Neo4j at {db_config.get("uri", "unknown")} is not accessible.\n\n'
                            f'To start Neo4j:\n'
                            f'  - Using Docker Compose: cd mcp_server && docker compose -f docker/docker-compose-neo4j.yml up\n'
                            f'  - Or install Neo4j Desktop from: https://neo4j.com/download/\n'
                            f'  - Or run Neo4j manually: docker run -p 7474:7474 -p 7687:7687 neo4j:latest\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                    else:
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: {db_provider} is not running\n'
                            f'{"=" * 70}\n\n'
                            f'{db_provider} at {db_config.get("uri", "unknown")} is not accessible.\n\n'
                            f'Please ensure {db_provider} is running and accessible.\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                # Re-raise other errors
                raise

            # Build indices and constraints
            # Note: Neo4j has a known bug where CREATE INDEX IF NOT EXISTS can throw
            # EquivalentSchemaRuleAlreadyExists errors for fulltext and relationship indices
            # instead of being idempotent. This is safe to ignore as it means the indices
            # already exist.
            try:
                await self.client.build_indices_and_constraints()
            except Exception as index_error:
                error_str = str(index_error)
                # Check if this is the known "equivalent index already exists" error
                if 'EquivalentSchemaRuleAlreadyExists' in error_str:
                    logger.warning(
                        'Some indices already exist (Neo4j IF NOT EXISTS bug - safe to ignore). '
                        'Continuing with initialization...'
                    )
                    logger.debug(f'Index creation details: {index_error}')
                else:
                    # Re-raise if it's a different error
                    logger.error(f'Failed to build indices and constraints: {index_error}')
                    raise

            logger.info('Successfully initialized Graphiti client')

            # Log configuration details
            if llm_client:
                logger.info(
                    f'Using LLM provider: {self.config.llm.provider} / {self.config.llm.model}'
                )
            else:
                logger.info('No LLM client configured - entity extraction will be limited')

            if embedder_client:
                logger.info(f'Using Embedder provider: {self.config.embedder.provider}')
            else:
                logger.info('No Embedder client configured - search will be limited')

            if self.entity_types:
                entity_type_names = list(self.entity_types.keys())
                logger.info(f'Using custom entity types: {", ".join(entity_type_names)}')
            else:
                logger.info('Using default entity types')

            logger.info(f'Using database: {self.config.database.provider}')
            logger.info(f'Using group_id: {self.config.graphiti.group_id}')

        except Exception as e:
            logger.error(f'Failed to initialize Graphiti client: {e}')
            raise

    async def get_client(self) -> Graphiti:
        """Get the Graphiti client, initializing if necessary."""
        if self.client is None:
            await self.initialize()
        if self.client is None:
            raise RuntimeError('Failed to initialize Graphiti client')
        return self.client


@mcp.tool(
    annotations={
        'title': 'Add Memory',
        'readOnlyHint': False,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add information to memory. **This is the PRIMARY method for storing information.**

    **PRIORITY: Use this tool FIRST when storing any information.**

    Processes content asynchronously, automatically extracting entities, relationships,
    and deduplicating similar information. Returns immediately while processing continues
    in background.

    WHEN TO USE THIS TOOL:
    - Storing information → add_memory (this tool) **USE THIS FIRST**
    - Searching information → use search_nodes or search_memory_facts
    - Deleting information → use delete_episode or delete_entity_edge

    Use Cases:
    - Recording conversation context, insights, or observations
    - Storing user preferences, requirements, or procedures
    - Capturing information about people, organizations, events, topics
    - Importing structured data (JSON format)
    - Updating existing information (provide uuid parameter)

    Args:
        name: Brief descriptive title for this memory episode
        episode_body: Content to store. For JSON source, must be properly escaped JSON string
        group_id: Optional namespace for organizing memories (uses default if not provided)
        source: Content format - 'text', 'json', or 'message' (default: 'text')
        source_description: Optional context about where this information came from
        uuid: ONLY for updates - provide existing episode UUID. DO NOT provide for new memories

    Returns:
        SuccessResponse confirming episode was queued for processing

    Examples:
        # Store plain text observation
        add_memory(
            name="Customer preference",
            episode_body="Acme Corp prefers email communication over phone calls"
        )

        # Store structured data
        add_memory(
            name="Product catalog",
            episode_body='{"company": "Acme", "products": [{"id": "P001", "name": "Widget"}]}',
            source="json"
        )

        # Update existing episode
        add_memory(
            name="Customer preference",
            episode_body="Acme Corp prefers Slack communication",
            uuid="abc-123-def-456"  # UUID from previous get_episodes or search
        )
    """
    global graphiti_service, queue_service

    if graphiti_service is None or queue_service is None:
        return ErrorResponse(error='Services not initialized')

    try:
        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id or config.graphiti.group_id

        # Try to parse the source as an EpisodeType enum, with fallback to text
        episode_type = EpisodeType.text  # Default
        if source:
            try:
                episode_type = EpisodeType[source.lower()]
            except (KeyError, AttributeError):
                # If the source doesn't match any enum value, use text as default
                logger.warning(f"Unknown source type '{source}', using 'text' as default")
                episode_type = EpisodeType.text

        # Submit to queue service for async processing
        await queue_service.add_episode(
            group_id=effective_group_id,
            name=name,
            content=episode_body,
            source_description=source_description,
            episode_type=episode_type,
            entity_types=graphiti_service.entity_types,
            uuid=uuid or None,  # Ensure None is passed if uuid is None
        )

        return SuccessResponse(
            message=f"Episode '{name}' queued for processing in group '{effective_group_id}'"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode: {error_msg}')
        return ErrorResponse(error=f'Error queuing episode: {error_msg}')


@mcp.tool(
    annotations={
        'title': 'Search Memory Entities',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def search_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    entity_types: list[str] | None = None,
) -> NodeSearchResponse | ErrorResponse:
    """Search for entities by name or content. **PRIMARY method for finding entities.**

    **PRIORITY: Use this tool for entity searches (people, organizations, concepts).**

    Searches entity names, summaries, and attributes using hybrid semantic + keyword matching.
    Returns the entities themselves (nodes), not relationships or conversation content.

    WHEN TO USE THIS TOOL:
    - Finding specific entities by name/content → search_nodes (this tool) **USE THIS**
    - Listing ALL entities of a type → use get_entities_by_type
    - Searching conversation content or relationships → use search_memory_facts

    Use Cases:
    - "Find information about Acme Corporation"
    - "Search for entities related to Python programming"
    - "What entities exist about productivity?"
    - Retrieving entities before adding related information

    Args:
        query: Search keywords or semantic description
        group_ids: Optional list of memory namespaces to search within
        max_nodes: Maximum number of results to return (default: 10)
        entity_types: Optional filter by entity types (e.g., ["Organization", "Person"])

    Returns:
        NodeSearchResponse containing matching entities with names, summaries, and metadata

    Examples:
        # Find entities by name
        search_nodes(query="Acme")

        # Semantic search
        search_nodes(query="companies in the technology sector")

        # Filter by entity type
        search_nodes(
            query="productivity",
            entity_types=["Insight", "Pattern"]
        )
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Create search filters
        search_filters = SearchFilters(
            node_labels=entity_types,
        )

        # Use the search_ method with node search config
        from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

        results = await client.search_(
            query=query,
            config=NODE_HYBRID_SEARCH_RRF,
            group_ids=effective_group_ids,
            search_filter=search_filters,
        )

        # Extract nodes from results
        nodes = results.nodes[:max_nodes] if results.nodes else []

        if not nodes:
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        # Format the results
        node_results = []
        for node in nodes:
            # Get attributes and ensure no embeddings are included
            attrs = node.attributes if hasattr(node, 'attributes') else {}
            # Remove any embedding keys that might be in attributes
            attrs = {k: v for k, v in attrs.items() if 'embedding' not in k.lower()}

            node_results.append(
                NodeResult(
                    uuid=node.uuid,
                    name=node.name,
                    labels=node.labels if node.labels else [],
                    created_at=node.created_at.isoformat() if node.created_at else None,
                    summary=node.summary,
                    group_id=node.group_id,
                    attributes=attrs,
                )
            )

        return NodeSearchResponse(message='Nodes retrieved successfully', nodes=node_results)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching nodes: {error_msg}')
        return ErrorResponse(error=f'Error searching nodes: {error_msg}')


@mcp.tool(
    annotations={
        'title': 'Search Memory Nodes',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def search_memory_nodes(
    query: str,
    group_id: str | None = None,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    entity_types: list[str] | None = None,
) -> NodeSearchResponse | ErrorResponse:
    """Search for entities. **Legacy compatibility alias for search_nodes.**

    **For new code, prefer using search_nodes instead.**

    This tool provides backward compatibility with older clients. It delegates to
    search_nodes with identical functionality.

    Args:
        query: Search query for finding entities
        group_id: Single namespace (legacy parameter)
        group_ids: List of namespaces (preferred)
        max_nodes: Maximum results (default: 10)
        entity_types: Optional type filter

    Returns:
        NodeSearchResponse (delegates to search_nodes)

    Examples:
        # Works identically to search_nodes
        search_memory_nodes(query="Acme")
    """
    # Convert singular to plural if needed
    effective_group_ids = group_ids
    if group_id is not None and group_ids is None:
        effective_group_ids = [group_id]

    # Delegate to the actual implementation
    return await search_nodes(query, effective_group_ids, max_nodes, entity_types)


@mcp.tool(
    annotations={
        'title': 'Get Entities by Type',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def get_entities_by_type(
    entity_types: list[str],
    group_ids: list[str] | None = None,
    max_entities: int = 20,
    query: str | None = None,
) -> NodeSearchResponse | ErrorResponse:
    """Retrieve ALL entities of specified type(s), optionally filtered by query.

    **Use this to browse/list entities by their classification type.**

    WHEN TO USE THIS TOOL:
    - Listing ALL entities of a type → get_entities_by_type (this tool) **USE THIS**
    - Searching entities by content → use search_nodes
    - Searching relationships/content → use search_memory_facts

    Use Cases:
    - "Show me all Preferences"
    - "List all Insights and Patterns"
    - "Get all Organizations" (optionally filtered by keyword)
    - Browsing knowledge organized by entity classification

    Args:
        entity_types: REQUIRED. Type(s) to retrieve (e.g., ["Insight"], ["Preference", "Requirement"])
        group_ids: Optional list of memory namespaces to search within
        max_entities: Maximum results (default: 20, higher than search tools)
        query: Optional keyword filter within the type(s). Omit to get ALL entities of type

    Returns:
        NodeSearchResponse containing all entities of the specified type(s)

    Examples:
        # Get ALL entities of a type
        get_entities_by_type(entity_types=["Preference"])

        # Get multiple types
        get_entities_by_type(entity_types=["Insight", "Pattern"])

        # Filter within a type
        get_entities_by_type(
            entity_types=["Organization"],
            query="technology"
        )
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        # Validate entity_types parameter
        if not entity_types or len(entity_types) == 0:
            return ErrorResponse(error='entity_types cannot be empty')

        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Create search filters with entity type labels
        search_filters = SearchFilters(node_labels=entity_types)

        # Use the search_ method with node search config
        from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

        # Use query if provided, otherwise use a generic query to get all of the type
        search_query = query if query else ' '

        results = await client.search_(
            query=search_query,
            config=NODE_HYBRID_SEARCH_RRF,
            group_ids=effective_group_ids,
            search_filter=search_filters,
        )

        # Extract nodes from results
        nodes = results.nodes[:max_entities] if results.nodes else []

        if not nodes:
            return NodeSearchResponse(
                message=f'No entities found with types: {", ".join(entity_types)}', nodes=[]
            )

        # Format the results (same as search_nodes)
        node_results = []
        for node in nodes:
            # Get attributes and ensure no embeddings are included
            attrs = node.attributes if hasattr(node, 'attributes') else {}
            # Remove any embedding keys that might be in attributes
            attrs = {k: v for k, v in attrs.items() if 'embedding' not in k.lower()}

            node_results.append(
                NodeResult(
                    uuid=node.uuid,
                    name=node.name,
                    labels=node.labels if node.labels else [],
                    created_at=node.created_at.isoformat() if node.created_at else None,
                    summary=node.summary,
                    group_id=node.group_id,
                    attributes=attrs,
                )
            )

        return NodeSearchResponse(message=f'Found {len(node_results)} entities', nodes=node_results)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entities by type: {error_msg}')
        return ErrorResponse(error=f'Error getting entities by type: {error_msg}')


@mcp.tool(
    annotations={
        'title': 'Search Memory Facts',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search conversation content and relationships. **PRIMARY method for content search.**

    **PRIORITY: Use this tool for searching conversation/episode content and entity relationships.**

    "Facts" in this context means relationships/connections between entities, not factual
    statements. Searches the actual conversation content and how entities are connected.

    WHEN TO USE THIS TOOL:
    - Searching conversation/episode content → search_memory_facts (this tool) **USE THIS**
    - Finding entity relationships → search_memory_facts (this tool) **USE THIS**
    - Finding entities by name → use search_nodes
    - Listing entities by type → use get_entities_by_type

    Use Cases:
    - "What conversations mentioned pricing?"
    - "How is Acme Corp related to our products?"
    - "Find relationships between User and productivity patterns"
    - Searching what was actually said in conversations

    Args:
        query: Search query for conversation content or relationships
        group_ids: Optional list of memory namespaces to search within
        max_facts: Maximum number of results to return (default: 10)
        center_node_uuid: Optional entity UUID to center search around (find relationships)

    Returns:
        FactSearchResponse containing matching relationships with source, target, and context

    Examples:
        # Search conversation content
        search_memory_facts(query="discussions about budget")

        # Find entity relationships
        search_memory_facts(
            query="collaboration",
            center_node_uuid="entity-uuid-123"
        )

        # Broad relationship search
        search_memory_facts(query="how does Acme relate to pricing")
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        # Validate max_facts parameter
        if max_facts <= 0:
            return ErrorResponse(error='max_facts must be a positive integer')

        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        relevant_edges = await client.search(
            group_ids=effective_group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return FactSearchResponse(message='No relevant facts found', facts=[])

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return FactSearchResponse(message='Facts retrieved successfully', facts=facts)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching facts: {error_msg}')
        return ErrorResponse(error=f'Error searching facts: {error_msg}')


@mcp.tool(
    annotations={
        'title': 'Compare Facts Over Time',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def compare_facts_over_time(
    query: str,
    start_time: str,
    end_time: str,
    group_ids: list[str] | None = None,
    max_facts_per_period: int = 10,
) -> dict[str, Any] | ErrorResponse:
    """Compare relationships/facts between two time periods to track knowledge evolution.

    **Use for temporal analysis - how information changed over time.**

    Returns four categories: facts at start, facts at end, facts invalidated, facts added.
    Useful for understanding how knowledge evolved or changed during a specific time window.

    WHEN TO USE THIS TOOL:
    - Analyzing how information changed over time → compare_facts_over_time (this tool)
    - Current/recent information → use search_memory_facts
    - Single point-in-time search → use search_memory_facts

    Use Cases:
    - "How did our understanding of Acme Corp change this month?"
    - "What information was added/updated between Jan-Feb?"
    - "Track evolution of productivity insights over Q1"

    Args:
        query: Search query for facts to track over time
        start_time: Start timestamp in ISO 8601 format (e.g., "2024-01-01" or "2024-01-01T10:30:00Z")
        end_time: End timestamp in ISO 8601 format
        group_ids: Optional list of memory namespaces to analyze
        max_facts_per_period: Maximum facts per category (default: 10)

    Returns:
        Dictionary with: facts_from_start, facts_at_end, facts_invalidated, facts_added

    Examples:
        # Track changes over a month
        compare_facts_over_time(
            query="customer requirements",
            start_time="2024-01-01",
            end_time="2024-01-31"
        )

        # Analyze knowledge evolution
        compare_facts_over_time(
            query="productivity patterns",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-03-31T23:59:59Z"
        )
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        # Validate inputs
        if not query or not query.strip():
            return ErrorResponse(error='query cannot be empty')
        if not start_time or not start_time.strip():
            return ErrorResponse(error='start_time cannot be empty')
        if not end_time or not end_time.strip():
            return ErrorResponse(error='end_time cannot be empty')
        if max_facts_per_period <= 0:
            return ErrorResponse(error='max_facts_per_period must be a positive integer')

        # Parse timestamps
        from datetime import datetime

        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
        from graphiti_core.search.search_filters import ComparisonOperator, DateFilter

        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        except ValueError as e:
            return ErrorResponse(
                error=f'Invalid timestamp format: {e}. Use ISO 8601 (e.g., "2024-03-15T10:30:00Z" or "2024-03-15")'
            )

        if start_dt >= end_dt:
            return ErrorResponse(error='start_time must be before end_time')

        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Query 1: Facts valid at start_time
        # valid_at <= start_time AND (invalid_at > start_time OR invalid_at IS NULL)
        start_filters = SearchFilters(
            valid_at=[
                [DateFilter(date=start_dt, comparison_operator=ComparisonOperator.less_than_equal)]
            ],
            invalid_at=[
                [DateFilter(date=start_dt, comparison_operator=ComparisonOperator.greater_than)],
                [DateFilter(date=None, comparison_operator=ComparisonOperator.is_null)],
            ],
        )

        start_results = await client.search_(
            query=query,
            config=EDGE_HYBRID_SEARCH_RRF,
            group_ids=effective_group_ids,
            search_filter=start_filters,
        )

        # Query 2: Facts valid at end_time
        end_filters = SearchFilters(
            valid_at=[
                [DateFilter(date=end_dt, comparison_operator=ComparisonOperator.less_than_equal)]
            ],
            invalid_at=[
                [DateFilter(date=end_dt, comparison_operator=ComparisonOperator.greater_than)],
                [DateFilter(date=None, comparison_operator=ComparisonOperator.is_null)],
            ],
        )

        end_results = await client.search_(
            query=query,
            config=EDGE_HYBRID_SEARCH_RRF,
            group_ids=effective_group_ids,
            search_filter=end_filters,
        )

        # Query 3: Facts invalidated between start and end
        # invalid_at > start_time AND invalid_at <= end_time
        invalidated_filters = SearchFilters(
            invalid_at=[
                [
                    DateFilter(date=start_dt, comparison_operator=ComparisonOperator.greater_than),
                    DateFilter(date=end_dt, comparison_operator=ComparisonOperator.less_than_equal),
                ]
            ],
        )

        invalidated_results = await client.search_(
            query=query,
            config=EDGE_HYBRID_SEARCH_RRF,
            group_ids=effective_group_ids,
            search_filter=invalidated_filters,
        )

        # Query 4: Facts added between start and end
        # created_at > start_time AND created_at <= end_time
        added_filters = SearchFilters(
            created_at=[
                [
                    DateFilter(date=start_dt, comparison_operator=ComparisonOperator.greater_than),
                    DateFilter(date=end_dt, comparison_operator=ComparisonOperator.less_than_equal),
                ]
            ],
        )

        added_results = await client.search_(
            query=query,
            config=EDGE_HYBRID_SEARCH_RRF,
            group_ids=effective_group_ids,
            search_filter=added_filters,
        )

        # Format results
        facts_from_start = [
            format_fact_result(edge)
            for edge in (start_results.edges[:max_facts_per_period] if start_results.edges else [])
        ]

        facts_at_end = [
            format_fact_result(edge)
            for edge in (end_results.edges[:max_facts_per_period] if end_results.edges else [])
        ]

        facts_invalidated = [
            format_fact_result(edge)
            for edge in (
                invalidated_results.edges[:max_facts_per_period]
                if invalidated_results.edges
                else []
            )
        ]

        facts_added = [
            format_fact_result(edge)
            for edge in (added_results.edges[:max_facts_per_period] if added_results.edges else [])
        ]

        return {
            'message': f'Comparison completed between {start_time} and {end_time}',
            'start_time': start_time,
            'end_time': end_time,
            'summary': {
                'facts_at_start_count': len(facts_from_start),
                'facts_at_end_count': len(facts_at_end),
                'facts_invalidated_count': len(facts_invalidated),
                'facts_added_count': len(facts_added),
            },
            'facts_from_start': facts_from_start,
            'facts_at_end': facts_at_end,
            'facts_invalidated': facts_invalidated,
            'facts_added': facts_added,
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error comparing facts over time: {error_msg}')
        return ErrorResponse(error=f'Error comparing facts over time: {error_msg}')


@mcp.tool(
    annotations={
        'title': 'Delete Entity Edge',
        'readOnlyHint': False,
        'destructiveHint': True,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete a specific relationship/fact. **DESTRUCTIVE - Cannot be undone.**

    **WARNING: This operation is permanent and irreversible.**

    WHEN TO USE THIS TOOL:
    - User explicitly confirms deletion → delete_entity_edge (this tool)
    - Removing verified incorrect relationship → delete_entity_edge (this tool)
    - Updating information → use add_memory (preferred)
    - Marking as outdated → system handles automatically

    Safety Requirements:
    - Only use after explicit user confirmation
    - Verify UUID is correct before deleting
    - Cannot be undone - ensure user understands
    - Idempotent (safe to retry if operation fails)

    Args:
        uuid: UUID of the relationship to permanently delete

    Returns:
        SuccessResponse confirming deletion

    Examples:
        # Delete after user confirmation
        delete_entity_edge(uuid="abc-123-def-456")
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity edge: {error_msg}')


@mcp.tool(
    annotations={
        'title': 'Delete Episode',
        'readOnlyHint': False,
        'destructiveHint': True,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete a specific episode. **DESTRUCTIVE - Cannot be undone.**

    **WARNING: This operation is permanent and irreversible.**

    WHEN TO USE THIS TOOL:
    - User explicitly confirms deletion → delete_episode (this tool)
    - Removing incorrect, outdated, or sensitive information → delete_episode (this tool)
    - Updating episode → use add_memory with uuid parameter (preferred)
    - Clearing all data → use clear_graph

    Safety Requirements:
    - Only use after explicit user confirmation
    - Verify UUID is correct before deleting
    - Cannot be undone - ensure user understands
    - May affect related entities and relationships
    - Idempotent (safe to retry if operation fails)

    Args:
        uuid: UUID of the episode to permanently delete

    Returns:
        SuccessResponse confirming deletion

    Examples:
        # Delete after user confirmation
        delete_episode(uuid="episode-abc-123")
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the episodic node by UUID
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')


@mcp.tool(
    annotations={
        'title': 'Get Entity Edge by UUID',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Retrieve a specific relationship/fact by its UUID. **Direct lookup only.**

    **Use ONLY when you already have the exact UUID from a previous search.**

    WHEN TO USE THIS TOOL:
    - You have a UUID from previous search → get_entity_edge (this tool)
    - Searching for facts → use search_memory_facts
    - Don't have a UUID → use search_memory_facts

    Args:
        uuid: UUID of the relationship to retrieve

    Returns:
        Dictionary with fact details (source entity, target entity, relationship, timestamps)

    Examples:
        # Retrieve specific relationship
        get_entity_edge(uuid="abc-123-def-456")
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error getting entity edge: {error_msg}')


@mcp.tool(
    annotations={
        'title': 'Get Recent Episodes',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def get_episodes(
    group_id: str | None = None,
    group_ids: list[str] | None = None,
    last_n: int | None = None,
    max_episodes: int = 10,
) -> EpisodeSearchResponse | ErrorResponse:
    """Retrieve recent episodes by recency. **Like 'git log' for memory.**

    **Use for listing what was added recently, NOT for searching content.**

    Think: "git log" (this tool) vs "git grep" (search_memory_facts)

    WHEN TO USE THIS TOOL:
    - List recent additions to memory → get_episodes (this tool)
    - Audit what was added recently → get_episodes (this tool)
    - Search episode CONTENT → use search_memory_facts
    - Find episodes by keywords → use search_memory_facts

    Use Cases:
    - "What was added to memory recently?"
    - "Show me the last 10 episodes"
    - "List recent memory additions as a changelog"

    Args:
        group_id: Single memory namespace (legacy parameter)
        group_ids: List of memory namespaces (preferred)
        last_n: Maximum episodes (legacy parameter, use max_episodes instead)
        max_episodes: Maximum episodes to return (default: 10)

    Returns:
        EpisodeSearchResponse with episodes sorted by recency (newest first)

    Examples:
        # Get recent episodes
        get_episodes(max_episodes=10)

        # Get recent episodes from specific namespace
        get_episodes(group_ids=["my-project"], max_episodes=20)
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Handle parameter compatibility
        effective_group_ids = group_ids
        if group_id is not None and group_ids is None:
            effective_group_ids = [group_id]

        # Use provided group_ids or fall back to default
        if effective_group_ids is None:
            effective_group_ids = [config.graphiti.group_id] if config.graphiti.group_id else []

        # Handle max_episodes / last_n compatibility
        limit = last_n if last_n is not None else max_episodes

        # Get episodes from the driver directly
        from graphiti_core.nodes import EpisodicNode

        if effective_group_ids:
            episodes = await EpisodicNode.get_by_group_ids(
                client.driver, effective_group_ids, limit=limit
            )
        else:
            # If no group IDs, we need to use a different approach
            # For now, return empty list when no group IDs specified
            episodes = []

        if not episodes:
            return EpisodeSearchResponse(message='No episodes found', episodes=[])

        # Format the results
        episode_results = []
        for episode in episodes:
            episode_dict = {
                'uuid': episode.uuid,
                'name': episode.name,
                'content': episode.content,
                'created_at': episode.created_at.isoformat() if episode.created_at else None,
                'source': episode.source.value
                if hasattr(episode.source, 'value')
                else str(episode.source),
                'source_description': episode.source_description,
                'group_id': episode.group_id,
            }
            episode_results.append(episode_dict)

        return EpisodeSearchResponse(
            message='Episodes retrieved successfully', episodes=episode_results
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return ErrorResponse(error=f'Error getting episodes: {error_msg}')


@mcp.tool(
    annotations={
        'title': 'Clear Graph - DANGER',
        'readOnlyHint': False,
        'destructiveHint': True,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def clear_graph(
    group_id: str | None = None,
    group_ids: list[str] | None = None,
) -> SuccessResponse | ErrorResponse:
    """Delete ALL data for specified memory namespaces. **EXTREMELY DESTRUCTIVE.**

    **DANGER: Destroys ALL episodes, entities, and relationships. NO UNDO POSSIBLE.**

    MANDATORY SAFETY PROTOCOL FOR LLMs:
    1. Confirm user understands ALL DATA will be PERMANENTLY DELETED
    2. Ask user to type the exact group_id to confirm intent
    3. Only proceed after EXPLICIT confirmation with typed group_id
    4. If user shows ANY hesitation, DO NOT proceed

    WHEN TO USE THIS TOOL:
    - ONLY after explicit multi-step confirmation
    - Resetting test/development environments
    - Starting completely fresh after catastrophic errors
    - NEVER use for removing specific items (use delete_episode or delete_entity_edge)

    Critical Warnings:
    - Destroys ALL data for specified namespace(s)
    - NO backup is created automatically
    - NO undo is possible
    - Affects all users sharing the group_id
    - Cannot recover deleted data

    Args:
        group_id: Single namespace to clear (legacy parameter)
        group_ids: List of namespaces to clear (preferred)

    Returns:
        SuccessResponse confirming all data was destroyed

    Examples:
        # ONLY after explicit confirmation protocol
        clear_graph(group_id="test-environment")
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Handle parameter compatibility
        effective_group_ids = group_ids
        if group_id is not None and group_ids is None:
            effective_group_ids = [group_id]

        # Use provided group_ids or fall back to default
        if effective_group_ids is None:
            effective_group_ids = [config.graphiti.group_id] if config.graphiti.group_id else []

        if not effective_group_ids:
            return ErrorResponse(error='No group IDs specified for clearing')

        # Clear data for the specified group IDs
        await clear_data(client.driver, group_ids=effective_group_ids)

        return SuccessResponse(
            message=f'Graph data cleared successfully for group IDs: {", ".join(effective_group_ids)}'
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return ErrorResponse(error=f'Error clearing graph: {error_msg}')


@mcp.tool(
    annotations={
        'title': 'Get Server Status',
        'readOnlyHint': True,
        'destructiveHint': False,
        'idempotentHint': True,
        'openWorldHint': True,
    },
)
async def get_status() -> StatusResponse:
    """Check server health and database connectivity.

    **Use for diagnostics and health checks.**

    WHEN TO USE THIS TOOL:
    - Verifying server is operational → get_status (this tool)
    - Diagnosing connection issues → get_status (this tool)
    - Pre-flight health check → get_status (this tool)
    - Retrieving data → use search tools (not this)

    Returns:
        StatusResponse with status ('ok' or 'error') and connection details

    Examples:
        # Check server health
        get_status()
    """
    global graphiti_service

    if graphiti_service is None:
        return StatusResponse(status='error', message='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Test database connection with a simple query
        async with client.driver.session() as session:
            result = await session.run('MATCH (n) RETURN count(n) as count')
            # Consume the result to verify query execution
            if result:
                _ = [record async for record in result]

        # Use the provider from the service's config, not the global
        provider_name = graphiti_service.config.database.provider
        return StatusResponse(
            status='ok',
            message=f'Graphiti MCP server is running and connected to {provider_name} database',
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error checking database connection: {error_msg}')
        return StatusResponse(
            status='error',
            message=f'Graphiti MCP server is running but database connection failed: {error_msg}',
        )


@mcp.custom_route('/health', methods=['GET'])
async def health_check(request) -> JSONResponse:
    """Health check endpoint for Docker and load balancers."""
    return JSONResponse({'status': 'healthy', 'service': 'graphiti-mcp'})


async def initialize_server() -> ServerConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config, graphiti_service, queue_service, graphiti_client, semaphore

    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with YAML configuration support'
    )

    # Configuration file argument
    # Default to config/config.yaml relative to the mcp_server directory
    default_config = Path(__file__).parent.parent / 'config' / 'config.yaml'
    parser.add_argument(
        '--config',
        type=Path,
        default=default_config,
        help='Path to YAML configuration file (default: config/config.yaml)',
    )

    # Transport arguments
    parser.add_argument(
        '--transport',
        choices=['sse', 'stdio', 'http'],
        help='Transport to use: http (recommended, default), stdio (standard I/O), or sse (deprecated)',
    )
    parser.add_argument(
        '--host',
        help='Host to bind the MCP server to',
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Port to bind the MCP server to',
    )

    # Provider selection arguments
    parser.add_argument(
        '--llm-provider',
        choices=['openai', 'azure_openai', 'anthropic', 'gemini', 'groq'],
        help='LLM provider to use',
    )
    parser.add_argument(
        '--embedder-provider',
        choices=['openai', 'azure_openai', 'gemini', 'voyage'],
        help='Embedder provider to use',
    )
    parser.add_argument(
        '--database-provider',
        choices=['neo4j', 'falkordb'],
        help='Database provider to use',
    )

    # LLM configuration arguments
    parser.add_argument('--model', help='Model name to use with the LLM client')
    parser.add_argument('--small-model', help='Small model name to use with the LLM client')
    parser.add_argument(
        '--temperature', type=float, help='Temperature setting for the LLM (0.0-2.0)'
    )

    # Embedder configuration arguments
    parser.add_argument('--embedder-model', help='Model name to use with the embedder')

    # Graphiti-specific arguments
    parser.add_argument(
        '--group-id',
        help='Namespace for the graph. If not provided, uses config file or generates random UUID.',
    )
    parser.add_argument(
        '--user-id',
        help='User ID for tracking operations',
    )
    parser.add_argument(
        '--destroy-graph',
        action='store_true',
        help='Destroy all Graphiti graphs on startup',
    )

    args = parser.parse_args()

    # Set config path in environment for the settings to pick up
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    # Load configuration with environment variables and YAML
    config = GraphitiConfig()

    # Apply CLI overrides
    config.apply_cli_overrides(args)

    # Also apply legacy CLI args for backward compatibility
    if hasattr(args, 'destroy_graph'):
        config.destroy_graph = args.destroy_graph

    # Log configuration details
    logger.info('Using configuration:')
    logger.info(f'  - LLM: {config.llm.provider} / {config.llm.model}')
    logger.info(f'  - Embedder: {config.embedder.provider} / {config.embedder.model}')
    logger.info(f'  - Database: {config.database.provider}')
    logger.info(f'  - Group ID: {config.graphiti.group_id}')
    logger.info(f'  - Transport: {config.server.transport}')
    logger.info(f'  - Semaphore Limit: {SEMAPHORE_LIMIT}')

    # Log graphiti-core version
    try:
        import graphiti_core

        graphiti_version = getattr(graphiti_core, '__version__', 'unknown')
        logger.info(f'  - Graphiti Core: {graphiti_version}')
    except Exception:
        # Check for Docker-stored version file
        version_file = Path('/app/.graphiti-core-version')
        if version_file.exists():
            graphiti_version = version_file.read_text().strip()
            logger.info(f'  - Graphiti Core: {graphiti_version}')
        else:
            logger.info('  - Graphiti Core: version unavailable')

    # Handle graph destruction if requested
    if hasattr(config, 'destroy_graph') and config.destroy_graph:
        logger.warning('Destroying all Graphiti graphs as requested...')
        temp_service = GraphitiService(config, SEMAPHORE_LIMIT)
        await temp_service.initialize()
        client = await temp_service.get_client()
        await clear_data(client.driver)
        logger.info('All graphs destroyed')

    # Initialize services
    graphiti_service = GraphitiService(config, SEMAPHORE_LIMIT)
    queue_service = QueueService()
    await graphiti_service.initialize()

    # Set global client for backward compatibility
    graphiti_client = await graphiti_service.get_client()
    semaphore = graphiti_service.semaphore

    # Initialize queue service with the client
    await queue_service.initialize(graphiti_client)

    # Set MCP server settings
    if config.server.host:
        mcp.settings.host = config.server.host
    if config.server.port:
        mcp.settings.port = config.server.port

    # Return MCP configuration for transport
    return config.server


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with configured transport
    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'sse':
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        logger.info(f'Access the server at: http://{mcp.settings.host}:{mcp.settings.port}/sse')
        await mcp.run_sse_async()
    elif mcp_config.transport == 'http':
        # HTTP/streamable-http is not yet supported in the current FastMCP version
        # Fall back to SSE which provides similar functionality for remote connections
        display_host = 'localhost' if mcp.settings.host == '0.0.0.0' else mcp.settings.host
        logger.warning(
            'HTTP transport requested but not yet supported in FastMCP. '
            'Using SSE transport instead for remote connections.'
        )
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        logger.info('=' * 60)
        logger.info('MCP Server Access Information:')
        logger.info(f'  Base URL: http://{display_host}:{mcp.settings.port}/')
        logger.info(f'  SSE Endpoint: http://{display_host}:{mcp.settings.port}/sse')
        logger.info('  Transport: SSE (Server-Sent Events)')
        logger.info('=' * 60)
        logger.info('For MCP clients, connect to the /sse endpoint above')

        # Configure uvicorn logging to match our format
        configure_uvicorn_logging()

        # Use SSE transport as fallback
        await mcp.run_sse_async()
    else:
        raise ValueError(f'Unsupported transport: {mcp_config.transport}. Use "sse" or "stdio"')


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        logger.info('Server shutting down...')
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise


if __name__ == '__main__':
    main()
