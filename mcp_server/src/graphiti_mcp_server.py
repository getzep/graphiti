#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodeType, SagaNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from starlette.responses import JSONResponse

from config.schema import GraphitiConfig, ServerConfig
from models.response_types import (
    BuildCommunitiesResponse,
    CommunityResult,
    EpisodeEntitiesResponse,
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeSearchResponse,
    SagaSummaryResponse,
    StatusResponse,
    SuccessResponse,
    TripletResponse,
)
from services.factories import DatabaseDriverFactory, EmbedderFactory, LLMClientFactory
from services.queue_service import QueueService
from utils.formatting import format_fact_result, to_edge_result, to_node_result
from utils.type_config import (
    build_edge_type_map,
    build_edge_types,
    build_entity_types,
    build_fact_search_filters,
    coerce_group_ids,
    parse_reference_time,
)

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
Graphiti is a memory service for AI agents built on a temporally-aware knowledge graph. It performs
well with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti organizes data as episodes (content snippets), nodes (entities), and facts (relationships
between entities). Each piece of information is partitioned by group_id, so you can maintain separate
knowledge domains in one graph.

Bi-temporal model: every episode records both when it was ingested and when the events it describes
actually occurred. Pass reference_time to add_memory to set the event-occurrence time; otherwise the
current time is used. Facts carry valid_at / invalid_at metadata, so a fact can be superseded by newer
information while its history is preserved.

Core tools:
- add_memory: add an episode (text, message, or JSON). Supports reference_time (bi-temporal),
  excluded_entity_types and custom_extraction_instructions to steer extraction,
  previous_episode_uuids to supply explicit context, update_communities to refresh community summaries,
  and saga / saga_previous_episode_uuid to associate the episode with an ordered saga.
- add_triplet: write a single fact (source entity -> fact -> target entity) directly, bypassing
  extraction. graphiti-core resolves/deduplicates the endpoint entities and generates embeddings.
- search_nodes: semantic + keyword + graph search over entities, optionally filtered by entity type
  (node label) and re-ranked around a center_node_uuid.
- search_memory_facts: search over facts (edges), optionally filtered by edge (fact) type and by
  valid_at / invalid_at date ranges.
- summarize_saga: generate or refresh the running summary of a saga's episodes.
- build_communities: detect entity communities and produce higher-level community summaries.
- get_episode_entities: trace provenance — the entities and facts created by specific episode UUIDs.
- get_entity_edge / get_episodes: retrieve specific facts or episodes.
- delete_episode: remove an episode and cascade-delete the entities/facts it solely created.
- delete_entity_edge / clear_graph: remove a fact, or clear a group's data.

Custom types: the server can register rich entity-type and edge-type models (with attributes and
extraction instructions) from configuration, and constrain which edge types are allowed between which
entity types via an edge_type_map. With no such configuration, default extraction behavior applies.

When adding information, provide descriptive names and detailed content to improve search quality.
When searching, use specific queries and consider filtering by group_id, type, or date range. The
server requires a configured database and valid API keys for language-model operations.
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
        self.entity_types: dict[str, type[BaseModel]] | None = None
        self.edge_types: dict[str, type[BaseModel]] | None = None
        self.edge_type_map: dict[tuple[str, str], list[str]] | None = None

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

            # Build entity / edge types from configuration. Rich Pydantic models
            # registered in models.entity_types / models.edge_types are preferred;
            # otherwise documentation-only models are built from the description.
            self.entity_types = build_entity_types(self.config.graphiti.entity_types)
            self.edge_types = build_edge_types(self.config.graphiti.edge_types)
            self.edge_type_map = build_edge_type_map(self.config.graphiti.edge_type_map)

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
                    # For Neo4j (default), use the original approach
                    self.client = Graphiti(
                        uri=db_config['uri'],
                        user=db_config['user'],
                        password=db_config['password'],
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

            # Build indices
            await self.client.build_indices_and_constraints()

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

            if self.edge_types:
                edge_type_names = list(self.edge_types.keys())
                logger.info(f'Using custom edge types: {", ".join(edge_type_names)}')
            else:
                logger.info('Using default edge types')

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


@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
    reference_time: str | None = None,
    excluded_entity_types: list[str] | None = None,
    custom_extraction_instructions: str | None = None,
    previous_episode_uuids: list[str] | None = None,
    update_communities: bool = False,
    saga: str | None = None,
    saga_previous_episode_uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

    Graphiti uses a bi-temporal model: each episode records both when it was ingested and
    when the described events actually occurred (its reference_time). Pass reference_time to
    set the event-occurrence time explicitly; otherwise the current time is used.

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode to persist to memory. When source='json', this must be a
                           properly escaped JSON string, not a raw Python dictionary. The JSON data will be
                           automatically processed to extract entities and relationships.
        group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default)
                               - 'json': For structured data
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode
        reference_time (str, optional): ISO-8601 timestamp for when the described events occurred
                                 (e.g. "2025-01-15T10:30:00Z" or "2025-01-15T10:30:00+00:00"). A
                                 timezone-naive value is interpreted as UTC. Defaults to the
                                 current time when omitted.
        excluded_entity_types (list[str], optional): Names of configured entity types to exclude
                                 from extraction for this episode.
        custom_extraction_instructions (str, optional): Additional natural-language guidance passed
                                 to the extraction model for this episode only.
        previous_episode_uuids (list[str], optional): Explicit list of prior episode UUIDs to use
                                 as conversational/contextual history. Overrides automatic retrieval.
        update_communities (bool, optional): When True, incrementally update community summaries to
                                 reflect entities affected by this episode. Defaults to False.
        saga (str, optional): Name/id of a saga to associate this episode with. Sagas group related
                                 episodes so their evolving narrative can be summarized via summarize_saga.
        saga_previous_episode_uuid (str, optional): UUID of the preceding episode in the saga, used to
                                 order episodes within the saga.

    Examples:
        # Adding plain text content
        add_memory(
            name="Company News",
            episode_body="Acme Corp announced a new product line today.",
            source="text",
            source_description="news article",
            group_id="some_arbitrary_string"
        )

        # Adding structured JSON data
        # NOTE: episode_body should be a JSON string (standard JSON escaping)
        add_memory(
            name="Customer Profile",
            episode_body='{"company": {"name": "Acme Technologies"}, "products": [{"id": "P001", "name": "CloudSync"}, {"id": "P002", "name": "DataMiner"}]}',
            source="json",
            source_description="CRM data"
        )

        # Recording an event that occurred in the past (bi-temporal)
        add_memory(
            name="Historical Note",
            episode_body="The merger closed in early 2020.",
            reference_time="2020-03-01T00:00:00Z"
        )
    """
    global graphiti_service, queue_service

    if graphiti_service is None or queue_service is None:
        return ErrorResponse(error='Services not initialized')

    # Parse the optional reference_time before queuing so callers get an immediate
    # error on a malformed timestamp rather than a silent background failure.
    try:
        parsed_reference_time = parse_reference_time(reference_time)
    except ValueError as e:
        return ErrorResponse(error=f'Invalid reference_time: {e}')

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
            reference_time=parsed_reference_time,
            edge_types=graphiti_service.edge_types,
            edge_type_map=graphiti_service.edge_type_map,
            excluded_entity_types=excluded_entity_types,
            previous_episode_uuids=previous_episode_uuids,
            custom_extraction_instructions=custom_extraction_instructions,
            update_communities=update_communities,
            saga=saga,
            saga_previous_episode_uuid=saga_previous_episode_uuid,
        )

        return SuccessResponse(
            message=f"Episode '{name}' queued for processing in group '{effective_group_id}'"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode: {error_msg}')
        return ErrorResponse(error=f'Error queuing episode: {error_msg}')


@mcp.tool()
async def search_nodes(
    query: str,
    group_ids: str | list[str] | None = None,
    max_nodes: int = 10,
    entity_types: list[str] | None = None,
    center_node_uuid: str | None = None,
) -> NodeSearchResponse | ErrorResponse:
    """Search for nodes (entities) in the graph memory.

    Args:
        query: The search query
        group_ids: Optional group ID, or list of group IDs, to filter results (a single
            string is accepted and treated as a one-element list)
        max_nodes: Maximum number of nodes to return (default: 10)
        entity_types: Optional list of entity type names (node labels) to filter by
        center_node_uuid: Optional UUID of a node to center the search around. Results
            closer to this node in the graph are ranked higher.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Accept a scalar group_id or a list; fall back to the default when omitted.
        group_ids = coerce_group_ids(group_ids)
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

        # center_node_uuid is only honored by the node_distance reranker, so select
        # that recipe when a center node is given (mirroring core's Graphiti.search);
        # otherwise use RRF.
        from graphiti_core.search.search_config_recipes import (
            NODE_HYBRID_SEARCH_NODE_DISTANCE,
            NODE_HYBRID_SEARCH_RRF,
        )

        node_config = (
            NODE_HYBRID_SEARCH_NODE_DISTANCE if center_node_uuid else NODE_HYBRID_SEARCH_RRF
        )
        results = await client.search_(
            query=query,
            config=node_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=search_filters,
        )

        # Extract nodes from results
        nodes = results.nodes[:max_nodes] if results.nodes else []

        if not nodes:
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        # Format the results (embeddings stripped by to_node_result)
        node_results = [to_node_result(node) for node in nodes]

        return NodeSearchResponse(message='Nodes retrieved successfully', nodes=node_results)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching nodes: {error_msg}')
        return ErrorResponse(error=f'Error searching nodes: {error_msg}')


@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: str | list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
    edge_types: list[str] | None = None,
    valid_at_after: str | None = None,
    valid_at_before: str | None = None,
    invalid_at_after: str | None = None,
    invalid_at_before: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the graph memory for relevant facts (entity edges).

    Args:
        query: The search query
        group_ids: Optional group ID, or list of group IDs, to filter results (a single
            string is accepted and treated as a one-element list)
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
        edge_types: Optional list of edge (fact) type names to filter by
        valid_at_after: Optional ISO-8601 lower bound; only facts whose valid_at is at or
            after this time are returned (timezone-naive is treated as UTC)
        valid_at_before: Optional ISO-8601 upper bound on a fact's valid_at
        invalid_at_after: Optional ISO-8601 lower bound on a fact's invalid_at
        invalid_at_before: Optional ISO-8601 upper bound on a fact's invalid_at
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        # Validate max_facts parameter
        if max_facts <= 0:
            return ErrorResponse(error='max_facts must be a positive integer')

        # Build search filters from the optional edge-type / date-range params.
        try:
            search_filter = build_fact_search_filters(
                edge_types=edge_types,
                valid_at_after=valid_at_after,
                valid_at_before=valid_at_before,
                invalid_at_after=invalid_at_after,
                invalid_at_before=invalid_at_before,
            )
        except ValueError as e:
            return ErrorResponse(error=f'Invalid date filter: {e}')

        client = await graphiti_service.get_client()

        # Accept a scalar group_id or a list; fall back to the default when omitted.
        group_ids = coerce_group_ids(group_ids)
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
            search_filter=search_filter,
        )

        if not relevant_edges:
            return FactSearchResponse(message='No relevant facts found', facts=[])

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return FactSearchResponse(message='Facts retrieved successfully', facts=facts)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching facts: {error_msg}')
        return ErrorResponse(error=f'Error searching facts: {error_msg}')


@mcp.tool()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory.

    Args:
        uuid: UUID of the entity edge to delete
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


@mcp.tool()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory.

    Uses Graphiti.remove_episode, which cascades the deletion: entities and facts
    that were created solely by this episode are removed along with it, while
    entities and facts also supported by other episodes are preserved.

    Args:
        uuid: UUID of the episode to delete
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # remove_episode cascades cleanup of episode-created entities/edges,
        # unlike EpisodicNode.delete which would orphan them.
        await client.remove_episode(uuid)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')


@mcp.tool()
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
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


@mcp.tool()
async def get_episodes(
    group_ids: str | list[str] | None = None,
    max_episodes: int = 10,
) -> EpisodeSearchResponse | ErrorResponse:
    """Get episodes from the graph memory.

    Args:
        group_ids: Optional group ID, or list of group IDs, to filter results (a single
            string is accepted and treated as a one-element list)
        max_episodes: Maximum number of episodes to return (default: 10)
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Accept a scalar group_id or a list; fall back to the default when omitted.
        group_ids = coerce_group_ids(group_ids)
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Get episodes from the driver directly
        from graphiti_core.nodes import EpisodicNode

        if effective_group_ids:
            episodes = await EpisodicNode.get_by_group_ids(
                client.driver, effective_group_ids, limit=max_episodes
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


@mcp.tool()
async def summarize_saga(
    saga_name: str, group_id: str | None = None
) -> SagaSummaryResponse | ErrorResponse:
    """Summarize a saga: an ordered group of related episodes.

    Generates (or refreshes) a running summary of the saga's narrative across its
    episodes and returns the saga's name and summary text.

    Sagas are keyed by (name, group_id): pass the same ``saga`` name you used with
    add_memory. This tool resolves that name to the saga within the group and
    summarizes it.

    Args:
        saga_name: The saga name — the same value passed as ``saga`` to add_memory.
        group_id: Optional group ID the saga belongs to. Falls back to the default group.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        effective_group_id = group_id or config.graphiti.group_id
        if not effective_group_id:
            return ErrorResponse(error='No group_id provided and no default group_id is configured')

        # add_memory takes a saga *name*; core keys sagas by (name, group_id) and
        # assigns its own UUID, while summarize_saga requires that UUID. Resolve the
        # name to its UUID within the group before delegating to core.
        sagas = await SagaNode.get_by_group_ids(client.driver, [effective_group_id])
        match = next((saga for saga in sagas if saga.name == saga_name), None)
        if match is None:
            return ErrorResponse(
                error=f"No saga named '{saga_name}' found in group '{effective_group_id}'"
            )

        saga_node = await client.summarize_saga(match.uuid)

        return SagaSummaryResponse(
            message=f"Saga '{saga_name}' summarized successfully",
            uuid=saga_node.uuid,
            name=saga_node.name,
            summary=saga_node.summary,
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error summarizing saga: {error_msg}')
        return ErrorResponse(error=f'Error summarizing saga: {error_msg}')


@mcp.tool()
async def build_communities(
    group_ids: str | list[str] | None = None,
) -> BuildCommunitiesResponse | ErrorResponse:
    """Detect and build community summaries over the graph's entities.

    Communities group densely-connected entities and produce higher-level summaries
    that can then be searched. This is a relatively expensive operation that
    processes the full set of entities for the given group(s).

    Args:
        group_ids: Optional group ID, or list of group IDs, to build communities for.
            Falls back to the default group when omitted. Pass an explicit list to scope
            community detection across multiple graphs.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Accept a scalar group_id or a list; fall back to the default when omitted.
        normalized_group_ids = coerce_group_ids(group_ids)
        if normalized_group_ids is None and config.graphiti.group_id:
            normalized_group_ids = [config.graphiti.group_id]

        communities, community_edges = await client.build_communities(
            group_ids=normalized_group_ids
        )

        community_results: list[CommunityResult] = [
            CommunityResult(
                uuid=community.uuid,
                name=community.name,
                group_id=community.group_id,
                summary=getattr(community, 'summary', None),
            )
            for community in communities
        ]

        return BuildCommunitiesResponse(
            message=f'Built {len(communities)} communities',
            community_count=len(communities),
            edge_count=len(community_edges),
            communities=community_results,
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error building communities: {error_msg}')
        return ErrorResponse(error=f'Error building communities: {error_msg}')


@mcp.tool()
async def add_triplet(
    source_node_name: str,
    edge_name: str,
    fact: str,
    target_node_name: str,
    group_id: str | None = None,
    source_node_uuid: str | None = None,
    target_node_uuid: str | None = None,
) -> TripletResponse | ErrorResponse:
    """Directly add a single fact triplet (source entity -> fact -> target entity).

    Unlike add_memory, this bypasses episode extraction and writes the relationship
    directly. graphiti-core resolves the source/target nodes against existing entities
    (deduplicating by name) and generates the required embeddings internally.

    Args:
        source_node_name: Name of the source entity
        edge_name: Relationship/edge type name (e.g. "WORKS_FOR")
        fact: Natural-language statement describing the relationship
        target_node_name: Name of the target entity
        group_id: Optional group ID. Falls back to the default group when omitted.
        source_node_uuid: Optional UUID to reuse an existing source entity. A new UUID is
            generated when omitted.
        target_node_uuid: Optional UUID to reuse an existing target entity. A new UUID is
            generated when omitted.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        effective_group_id = group_id or config.graphiti.group_id
        if not effective_group_id:
            return ErrorResponse(error='No group_id provided and no default group_id is configured')
        now = datetime.now(timezone.utc)

        source_node = EntityNode(
            uuid=source_node_uuid or str(uuid4()),
            name=source_node_name,
            group_id=effective_group_id,
            created_at=now,
        )
        target_node = EntityNode(
            uuid=target_node_uuid or str(uuid4()),
            name=target_node_name,
            group_id=effective_group_id,
            created_at=now,
        )
        edge = EntityEdge(
            name=edge_name,
            fact=fact,
            group_id=effective_group_id,
            source_node_uuid=source_node.uuid,
            target_node_uuid=target_node.uuid,
            created_at=now,
        )

        result = await client.add_triplet(source_node, edge, target_node)

        return TripletResponse(
            message=f"Triplet '{source_node_name} -[{edge_name}]-> {target_node_name}' added",
            nodes=[to_node_result(node) for node in result.nodes],
            edges=[to_edge_result(e) for e in result.edges],
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error adding triplet: {error_msg}')
        return ErrorResponse(error=f'Error adding triplet: {error_msg}')


@mcp.tool()
async def get_episode_entities(
    episode_uuids: list[str],
) -> EpisodeEntitiesResponse | ErrorResponse:
    """Get the entities (nodes) and facts (edges) created by specific episodes.

    Use this to trace provenance: given one or more episode UUIDs, return the graph
    elements that those episodes produced.

    Args:
        episode_uuids: List of episode UUIDs to look up provenance for
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    if not episode_uuids:
        return ErrorResponse(error='episode_uuids must contain at least one UUID')

    try:
        client = await graphiti_service.get_client()

        results = await client.get_nodes_and_edges_by_episode(episode_uuids)

        return EpisodeEntitiesResponse(
            message=f'Retrieved provenance for {len(episode_uuids)} episode(s)',
            nodes=[to_node_result(node) for node in results.nodes],
            edges=[to_edge_result(e) for e in results.edges],
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episode entities: {error_msg}')
        return ErrorResponse(error=f'Error getting episode entities: {error_msg}')


@mcp.tool()
async def clear_graph(
    group_ids: str | list[str] | None = None,
) -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph for specified group IDs.

    Args:
        group_ids: Optional group ID, or list of group IDs, to clear (a single string is
            accepted). If not provided, clears the default group.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Accept a scalar group_id or a list; fall back to the default when omitted.
        # (Parenthesized so an explicit group_ids isn't dropped when the configured
        # default group_id is unset — `or` binds tighter than the ternary.)
        group_ids = coerce_group_ids(group_ids)
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else ([config.graphiti.group_id] if config.graphiti.group_id else [])
        )

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


@mcp.tool()
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and database connection."""
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
        # Use localhost for display if binding to 0.0.0.0
        display_host = 'localhost' if mcp.settings.host == '0.0.0.0' else mcp.settings.host
        logger.info(
            f'Running MCP server with streamable HTTP transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        logger.info('=' * 60)
        logger.info('MCP Server Access Information:')
        logger.info(f'  Base URL: http://{display_host}:{mcp.settings.port}/')
        logger.info(f'  MCP Endpoint: http://{display_host}:{mcp.settings.port}/mcp/')
        logger.info('  Transport: HTTP (streamable)')

        # Show FalkorDB Browser UI access if enabled
        if os.environ.get('BROWSER', '1') == '1':
            logger.info(f'  FalkorDB Browser UI: http://{display_host}:3000/')

        logger.info('=' * 60)
        logger.info('For MCP clients, connect to the /mcp/ endpoint above')

        # Configure uvicorn logging to match our format
        configure_uvicorn_logging()

        await mcp.run_streamable_http_async()
    else:
        raise ValueError(
            f'Unsupported transport: {mcp_config.transport}. Use "sse", "stdio", or "http"'
        )


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
