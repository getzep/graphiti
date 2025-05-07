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
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from openai import AsyncAzureOpenAI, APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

from .utils import (
    CircuitBreakerError,
    PermanentError,
    RetryableError,
    RETRYABLE_NETWORK_ERRORS,
    with_circuit_breaker,
    with_logging,
    with_retry,
)

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()

DEFAULT_LLM_MODEL = 'gpt-4.1-mini'
DEFAULT_EMBEDDER_MODEL = 'text-embedding-3-small'


class Requirement(BaseModel):
    """A Requirement represents a specific need, feature, or functionality that a product or service must fulfill.

    Always ensure an edge is created between the requirement and the project it belongs to, and clearly indicate on the
    edge that the requirement is a requirement.

    Instructions for identifying and extracting requirements:
    1. Look for explicit statements of needs or necessities ("We need X", "X is required", "X must have Y")
    2. Identify functional specifications that describe what the system should do
    3. Pay attention to non-functional requirements like performance, security, or usability criteria
    4. Extract constraints or limitations that must be adhered to
    5. Focus on clear, specific, and measurable requirements rather than vague wishes
    6. Capture the priority or importance if mentioned ("critical", "high priority", etc.)
    7. Include any dependencies between requirements when explicitly stated
    8. Preserve the original intent and scope of the requirement
    9. Categorize requirements appropriately based on their domain or function
    """

    project_name: str = Field(
        ...,
        description='The name of the project to which the requirement belongs.',
    )
    description: str = Field(
        ...,
        description='Description of the requirement. Only use information mentioned in the context to write this description.',
    )


class Preference(BaseModel):
    """A Preference represents a user's expressed like, dislike, or preference for something.

    Instructions for identifying and extracting preferences:
    1. Look for explicit statements of preference such as "I like/love/enjoy/prefer X" or "I don't like/hate/dislike X"
    2. Pay attention to comparative statements ("I prefer X over Y")
    3. Consider the emotional tone when users mention certain topics
    4. Extract only preferences that are clearly expressed, not assumptions
    5. Categorize the preference appropriately based on its domain (food, music, brands, etc.)
    6. Include relevant qualifiers (e.g., "likes spicy food" rather than just "likes food")
    7. Only extract preferences directly stated by the user, not preferences of others they mention
    8. Provide a concise but specific description that captures the nature of the preference
    """

    category: str = Field(
        ...,
        description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')",
    )
    description: str = Field(
        ...,
        description='Brief description of the preference. Only use information mentioned in the context to write this description.',
    )


class Procedure(BaseModel):
    """A Procedure informing the agent what actions to take or how to perform in certain scenarios. Procedures are typically composed of several steps.

    Instructions for identifying and extracting procedures:
    1. Look for sequential instructions or steps ("First do X, then do Y")
    2. Identify explicit directives or commands ("Always do X when Y happens")
    3. Pay attention to conditional statements ("If X occurs, then do Y")
    4. Extract procedures that have clear beginning and end points
    5. Focus on actionable instructions rather than general information
    6. Preserve the original sequence and dependencies between steps
    7. Include any specified conditions or triggers for the procedure
    8. Capture any stated purpose or goal of the procedure
    9. Summarize complex procedures while maintaining critical details
    """

    description: str = Field(
        ...,
        description='Brief description of the procedure. Only use information mentioned in the context to write this description.',
    )


ENTITY_TYPES: dict[str, BaseModel] = {
    'Requirement': Requirement,  # type: ignore
    'Preference': Preference,  # type: ignore
    'Procedure': Procedure,  # type: ignore
}


# Type definitions for API responses
class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


class NodeResult(TypedDict):
    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    status: str
    message: str


def create_azure_credential_token_provider() -> Callable[[], str]:
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, 'https://cognitiveservices.azure.com/.default'
    )
    return token_provider


# Server configuration classes
# The configuration system has a hierarchy:
# - GraphitiConfig is the top-level configuration
#   - LLMConfig handles all OpenAI/LLM related settings
#   - EmbedderConfig manages embedding settings
#   - Neo4jConfig manages database connection details
#   - Various other settings like group_id and feature flags
# Configuration values are loaded from:
# 1. Default values in the class definitions
# 2. Environment variables (loaded via load_dotenv())
# 3. Command line arguments (which override environment variables)
class GraphitiLLMConfig(BaseModel):
    """Configuration for the LLM client.

    Centralizes all LLM-specific configuration parameters including API keys and model selection.
    """

    api_key: str | None = None
    model: str = DEFAULT_LLM_MODEL
    temperature: float = 0.0
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiLLMConfig':
        """Create LLM configuration from environment variables."""
        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_LLM_MODEL

        azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', None)
        azure_openai_api_version = os.environ.get('AZURE_OPENAI_API_VERSION', None)
        azure_openai_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', None)
        azure_openai_use_managed_identity = (
            os.environ.get('AZURE_OPENAI_USE_MANAGED_IDENTITY', 'false').lower() == 'true'
        )

        if azure_openai_endpoint is None:
            # Setup for OpenAI API
            # Log if empty model was provided
            if model_env == '':
                logger.debug(
                    f'MODEL_NAME environment variable not set, using default: {DEFAULT_LLM_MODEL}'
                )
            elif not model_env.strip():
                logger.warning(
                    f'Empty MODEL_NAME environment variable, using default: {DEFAULT_LLM_MODEL}'
                )

            return cls(
                api_key=os.environ.get('OPENAI_API_KEY'),
                model=model,
                temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
            )
        else:
            # Setup for Azure OpenAI API
            # Log if empty deployment name was provided
            if azure_openai_deployment_name is None:
                logger.error('AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set')

                raise ValueError('AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set')
            if not azure_openai_use_managed_identity:
                # api key
                api_key = os.environ.get('OPENAI_API_KEY', None)
            else:
                # Managed identity
                api_key = None

            return cls(
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                azure_openai_endpoint=azure_openai_endpoint,
                api_key=api_key,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
                temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
            )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiLLMConfig':
        """Create LLM configuration from CLI arguments, falling back to environment variables."""
        # Start with environment-based config
        config = cls.from_env()

        # CLI arguments override environment variables when provided
        if hasattr(args, 'model') and args.model:
            # Only use CLI model if it's not empty
            if args.model.strip():
                config.model = args.model
            else:
                # Log that empty model was provided and default is used
                logger.warning(f'Empty model name provided, using default: {DEFAULT_LLM_MODEL}')

        if hasattr(args, 'temperature') and args.temperature is not None:
            config.temperature = args.temperature

        return config

    def create_client(self) -> LLMClient | None:
        """Create an LLM client based on this configuration.

        Returns:
            LLMClient instance if able, None otherwise
        """

        if self.azure_openai_endpoint is not None:
            # Azure OpenAI API setup
            if self.azure_openai_use_managed_identity:
                # Use managed identity for authentication
                token_provider = create_azure_credential_token_provider()
                return AsyncAzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    azure_deployment=self.azure_openai_deployment_name,
                    api_version=self.azure_openai_api_version,
                    azure_ad_token_provider=token_provider,
                )
            elif self.api_key:
                # Use API key for authentication
                return AsyncAzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    azure_deployment=self.azure_openai_deployment_name,
                    api_version=self.azure_openai_api_version,
                    api_key=self.api_key,
                )
            else:
                logger.error('OPENAI_API_KEY must be set when using Azure OpenAI API')
                return None

        if not self.api_key:
            return None

        llm_client_config = LLMConfig(api_key=self.api_key, model=self.model)

        # Set temperature
        llm_client_config.temperature = self.temperature

        return OpenAIClient(config=llm_client_config)

    def create_cross_encoder_client(self) -> CrossEncoderClient | None:
        """Create a cross-encoder client based on this configuration."""
        if self.azure_openai_endpoint is not None:
            client = self.create_client()
            return OpenAIRerankerClient(client=client)
        else:
            llm_client_config = LLMConfig(api_key=self.api_key, model=self.model)
            return OpenAIRerankerClient(config=llm_client_config)


class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the embedder client.

    Centralizes all embedding-related configuration parameters.
    """

    model: str = DEFAULT_EMBEDDER_MODEL
    api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiEmbedderConfig':
        """Create embedder configuration from environment variables."""

        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('EMBEDDER_MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_EMBEDDER_MODEL

        azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', None)
        azure_openai_api_version = os.environ.get('AZURE_OPENAI_EMBEDDING_API_VERSION', None)
        azure_openai_deployment_name = os.environ.get(
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', None
        )
        azure_openai_use_managed_identity = (
            os.environ.get('AZURE_OPENAI_USE_MANAGED_IDENTITY', 'false').lower() == 'true'
        )
        if azure_openai_endpoint is not None:
            # Setup for Azure OpenAI API
            # Log if empty deployment name was provided
            azure_openai_deployment_name = os.environ.get(
                'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', None
            )
            if azure_openai_deployment_name is None:
                logger.error('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set')

                raise ValueError(
                    'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set'
                )

            if not azure_openai_use_managed_identity:
                # api key
                api_key = os.environ.get('OPENAI_API_KEY', None)
            else:
                # Managed identity
                api_key = None

            return cls(
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                azure_openai_endpoint=azure_openai_endpoint,
                api_key=api_key,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
            )
        else:
            return cls(
                model=model,
                api_key=os.environ.get('OPENAI_API_KEY'),
            )

    def create_client(self) -> EmbedderClient | None:
        if self.azure_openai_endpoint is not None:
            # Azure OpenAI API setup
            if self.azure_openai_use_managed_identity:
                # Use managed identity for authentication
                token_provider = create_azure_credential_token_provider()
                return AsyncAzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    azure_deployment=self.azure_openai_deployment_name,
                    api_version=self.azure_openai_api_version,
                    azure_ad_token_provider=token_provider,
                )
            elif self.api_key:
                # Use API key for authentication
                return AsyncAzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    azure_deployment=self.azure_openai_deployment_name,
                    api_version=self.azure_openai_api_version,
                    api_key=self.api_key,
                )
            else:
                logger.error('OPENAI_API_KEY must be set when using Azure OpenAI API')
                return None
        else:
            # OpenAI API setup
            if not self.api_key:
                return None

            embedder_config = OpenAIEmbedderConfig(api_key=self.api_key, model=self.model)

            return OpenAIEmbedder(config=embedder_config)


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j database connection."""

    uri: str = 'bolt://localhost:7687'
    user: str = 'neo4j'
    password: str = 'password'

    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create Neo4j configuration from environment variables."""
        return cls(
            uri=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.environ.get('NEO4J_USER', 'neo4j'),
            password=os.environ.get('NEO4J_PASSWORD', 'password'),
        )


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti client.

    Centralizes all configuration parameters for the Graphiti client.
    """

    llm: GraphitiLLMConfig = Field(default_factory=GraphitiLLMConfig)
    embedder: GraphitiEmbedderConfig = Field(default_factory=GraphitiEmbedderConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    group_id: str | None = None
    use_custom_entities: bool = False
    destroy_graph: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiConfig':
        """Create a configuration instance from environment variables."""
        return cls(
            llm=GraphitiLLMConfig.from_env(),
            embedder=GraphitiEmbedderConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiConfig':
        """Create configuration from CLI arguments, falling back to environment variables."""
        # Start with environment configuration
        config = cls.from_env()

        # Apply CLI overrides
        if args.group_id:
            config.group_id = args.group_id
        else:
            config.group_id = f'graph_{uuid.uuid4().hex[:8]}'

        config.use_custom_entities = args.use_custom_entities
        config.destroy_graph = args.destroy_graph

        # Update LLM config using CLI args
        config.llm = GraphitiLLMConfig.from_cli_and_env(args)

        return config


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str = 'sse'  # Default to SSE transport

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> 'MCPConfig':
        """Create MCP configuration from CLI arguments."""
        return cls(transport=args.transport)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create global config instance - will be properly initialized later
config = GraphitiConfig()

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

# MCP server instance
mcp = FastMCP(
    'graphiti',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)

# Initialize Graphiti client
graphiti_client: Graphiti | None = None


class GraphitiInitError(Exception):
    """Base class for Graphiti initialization errors."""
    pass

class ConfigurationError(GraphitiInitError, PermanentError):
    """Error in configuration values."""
    pass

class DatabaseConnectionError(GraphitiInitError, RetryableError):
    """Error connecting to Neo4j database."""
    pass

class LLMServiceError(GraphitiInitError, RetryableError):
    """Error initializing LLM service."""
    pass

@with_retry(
    max_attempts=5,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(
        DatabaseConnectionError,
        LLMServiceError,
        *RETRYABLE_NETWORK_ERRORS,
    ),
)
@with_logging(include_args=False)  # Don't log args as they contain credentials
async def initialize_graphiti():
    """Initialize the Graphiti client with the configured settings.
    
    This function implements retry logic for transient failures and
    graceful degradation for non-critical service failures.
    
    Raises:
        ConfigurationError: If required configuration is missing
        DatabaseConnectionError: If Neo4j connection fails
        LLMServiceError: If LLM service initialization fails
        GraphitiInitError: For other initialization errors
    """
    global graphiti_client, config

    try:
        # Validate Neo4j configuration first
        if not config.neo4j.uri or not config.neo4j.user or not config.neo4j.password:
            raise ConfigurationError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

        # Initialize services with graceful degradation
        llm_client = None
        embedder_client = None
        cross_encoder_client = None

        # Try to initialize LLM client
        try:
            llm_client = config.llm.create_client()
            if llm_client:
                logger.info(f'LLM client initialized with model: {config.llm.model}')
            elif config.use_custom_entities:
                # Only raise error if custom entities are required
                raise ConfigurationError('OPENAI_API_KEY must be set when custom entities are enabled')
            else:
                logger.warning('No LLM client configured - entity extraction will be limited')
        except (APIError, APITimeoutError, RateLimitError, AzureError) as e:
            if config.use_custom_entities:
                raise LLMServiceError(f'Failed to initialize LLM client: {str(e)}') from e
            logger.warning(f'Failed to initialize LLM client, continuing with limited functionality: {str(e)}')

        # Try to initialize embedder client
        try:
            embedder_client = config.embedder.create_client()
            if embedder_client:
                logger.info('Embedder client initialized successfully')
            else:
                logger.warning('No embedder client configured - search functionality will be limited')
        except (APIError, APITimeoutError, RateLimitError, AzureError) as e:
            logger.warning(f'Failed to initialize embedder client, continuing with limited functionality: {str(e)}')

        # Try to initialize cross-encoder client
        try:
            cross_encoder_client = config.llm.create_cross_encoder_client()
            if cross_encoder_client:
                logger.info('Cross-encoder client initialized successfully')
            else:
                logger.warning('No cross-encoder client configured - search reranking will be disabled')
        except (APIError, APITimeoutError, RateLimitError, AzureError) as e:
            logger.warning(f'Failed to initialize cross-encoder client, continuing without reranking: {str(e)}')

        # Initialize Graphiti client with Neo4j connection
        try:
            graphiti_client = Graphiti(
                uri=config.neo4j.uri,
                user=config.neo4j.user,
                password=config.neo4j.password,
                llm_client=llm_client,
                embedder=embedder_client,
                cross_encoder=cross_encoder_client,
            )
        except (ServiceUnavailable, SessionExpired) as e:
            raise DatabaseConnectionError(f'Failed to connect to Neo4j: {str(e)}') from e
        except Exception as e:
            raise GraphitiInitError(f'Failed to initialize Graphiti client: {str(e)}') from e

        # Destroy graph if requested
        if config.destroy_graph:
            logger.info('Destroying graph...')
            try:
                await clear_data(graphiti_client.driver)
            except Exception as e:
                logger.error(f'Failed to clear graph data: {str(e)}')
                # Continue with initialization even if clear fails

        # Initialize the graph database with Graphiti's indices
        try:
            await graphiti_client.build_indices_and_constraints()
            logger.info('Graph indices and constraints initialized successfully')
        except Exception as e:
            raise DatabaseConnectionError(f'Failed to initialize graph indices: {str(e)}') from e

        # Log final configuration state
        logger.info(f'Using group_id: {config.group_id}')
        logger.info(
            f'Custom entity extraction: {"enabled" if config.use_custom_entities else "disabled"}'
        )
        logger.info('Graphiti client initialized successfully')

        # Return success to signal retry decorator
        return True

    except (ConfigurationError, GraphitiInitError) as e:
        # Log with full context and re-raise
        logger.error('Graphiti initialization failed', exc_info=True)
        raise


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.

    Since EntityEdge is a Pydantic BaseModel, we can use its built-in serialization capabilities.

    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    return edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',
        },
    )


# Dictionary to store queues for each group_id
# Each queue is a list of tasks to be processed sequentially
episode_queues: dict[str, asyncio.Queue] = {}
# Dictionary to track if a worker is running for each group_id
queue_workers: dict[str, bool] = {}


class EpisodeProcessingError(RetryableError):
    """Error processing an episode that may be retryable."""
    pass

class EpisodeValidationError(PermanentError):
    """Error validating episode data that cannot be retried."""
    pass

@with_circuit_breaker(failure_threshold=5, reset_timeout=300.0)
@with_retry(
    max_attempts=3,
    base_delay=2.0,
    max_delay=10.0,
    retryable_exceptions=(EpisodeProcessingError, *RETRYABLE_NETWORK_ERRORS),
)
@with_logging()
async def process_episode_queue(group_id: str):
    """Process episodes for a specific group_id sequentially.

    This function runs as a long-lived task that processes episodes
    from the queue one at a time. It implements:
    - Circuit breaker to prevent overwhelming failing systems
    - Retry logic for transient failures
    - Detailed logging of operations
    - Error classification and handling
    """
    global queue_workers

    logger.info(f'Starting episode queue worker for group_id: {group_id}')
    queue_workers[group_id] = True
    
    # Track queue metrics
    queue_size = episode_queues[group_id].qsize()
    logger.info(f'Current queue size for group_id {group_id}: {queue_size}')

    try:
        while True:
            # Get the next episode processing function from the queue
            # This will wait if the queue is empty
            process_func = await episode_queues[group_id].get()
            
            # Log queue metrics after getting item
            new_size = episode_queues[group_id].qsize()
            if new_size != queue_size:
                queue_size = new_size
                logger.info(f'Queue size for group_id {group_id} changed to: {queue_size}')

            try:
                start_time = time.time()
                
                # Process the episode with timeout
                try:
                    async with asyncio.timeout(30):  # 30 second timeout
                        await process_func()
                except asyncio.TimeoutError:
                    raise EpisodeProcessingError("Episode processing timed out")
                
                # Log processing time
                duration = time.time() - start_time
                logger.info(
                    f'Episode processed successfully',
                    extra={
                        'group_id': group_id,
                        'duration': f'{duration:.3f}s',
                        'queue_size': queue_size
                    }
                )
                
            except (ServiceUnavailable, SessionExpired) as e:
                # Database connection issues - retryable
                logger.warning(
                    f'Database error processing episode',
                    extra={
                        'group_id': group_id,
                        'error': str(e),
                        'error_type': e.__class__.__name__
                    }
                )
                raise EpisodeProcessingError(f"Database error: {str(e)}") from e
                
            except (APIError, APITimeoutError, RateLimitError) as e:
                # LLM service issues - retryable
                logger.warning(
                    f'LLM service error processing episode',
                    extra={
                        'group_id': group_id,
                        'error': str(e),
                        'error_type': e.__class__.__name__
                    }
                )
                raise EpisodeProcessingError(f"LLM service error: {str(e)}") from e
                
            except ValueError as e:
                # Validation errors - not retryable
                logger.error(
                    f'Validation error processing episode',
                    extra={
                        'group_id': group_id,
                        'error': str(e),
                        'error_type': e.__class__.__name__
                    }
                )
                # Don't retry validation errors
                
            except Exception as e:
                # Unexpected errors - log with full context
                logger.error(
                    f'Unexpected error processing episode',
                    extra={
                        'group_id': group_id,
                        'error': str(e),
                        'error_type': e.__class__.__name__
                    },
                    exc_info=True
                )
                # Treat unexpected errors as retryable
                raise EpisodeProcessingError(f"Unexpected error: {str(e)}") from e
                
            finally:
                # Mark the task as done regardless of success/failure
                episode_queues[group_id].task_done()
                
    except asyncio.CancelledError:
        logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        
    except CircuitBreakerError as e:
        logger.error(
            f'Circuit breaker opened for group_id {group_id}',
            extra={'error': str(e)}
        )
        # Allow circuit breaker errors to propagate
        raise
        
    except Exception as e:
        logger.error(
            f'Fatal error in queue worker',
            extra={
                'group_id': group_id,
                'error': str(e),
                'error_type': e.__class__.__name__
            },
            exc_info=True
        )
        raise
        
    finally:
        queue_workers[group_id] = False
        logger.info(f'Stopped episode queue worker for group_id: {group_id}')


@mcp.tool()
@with_logging(truncate_length=2000)  # Truncate long episode bodies in logs
async def add_episode(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to the Graphiti knowledge graph. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

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
        # NOTE: episode_body must be a properly escaped JSON string. Note the triple backslashes
        add_episode(
            name="Customer Profile",
            episode_body="{\\\"company\\\": {\\\"name\\\": \\\"Acme Technologies\\\"}, \\\"products\\\": [{\\\"id\\\": \\\"P001\\\", \\\"name\\\": \\\"CloudSync\\\"}, {\\\"id\\\": \\\"P002\\\", \\\"name\\\": \\\"DataMiner\\\"}]}",
            source="json",
            source_description="CRM data"
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
    global graphiti_client, episode_queues, queue_workers

    # Generate a request ID for tracking this operation
    request_id = str(uuid.uuid4())
    logger.info(
        f"Received add_episode request",
        extra={
            'request_id': request_id,
            'name': name,
            'source': source,
            'group_id': group_id,
            'body_length': len(episode_body),
        }
    )

    if graphiti_client is None:
        logger.error(
            "Add episode failed - Graphiti client not initialized",
            extra={'request_id': request_id}
        )
        return {'error': 'Graphiti client not initialized'}

    try:
        # Input validation
        if not name or not name.strip():
            raise EpisodeValidationError("Episode name cannot be empty")
        
        if not episode_body or not episode_body.strip():
            raise EpisodeValidationError("Episode body cannot be empty")
            
        # Validate source type
        source = source.lower()
        if source not in ('text', 'json', 'message'):
            raise EpisodeValidationError(
                f"Invalid source type '{source}'. Must be one of: text, json, message"
            )

        # Map string source to EpisodeType enum
        source_type = EpisodeType.text
        if source == 'message':
            source_type = EpisodeType.message
        elif source == 'json':
            source_type = EpisodeType.json
            # Validate JSON if source type is json
            try:
                import json
                json.loads(episode_body)
            except json.JSONDecodeError as e:
                raise EpisodeValidationError(f"Invalid JSON in episode_body: {str(e)}")

        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        # Cast group_id to str to satisfy type checker
        # The Graphiti client expects a str for group_id, not Optional[str]
        group_id_str = str(effective_group_id) if effective_group_id is not None else ''

        # We've already checked that graphiti_client is not None above
        # This assert statement helps type checkers understand that graphiti_client is defined
        assert graphiti_client is not None, 'graphiti_client should not be None here'

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Define the episode processing function with request tracking
        async def process_episode():
            try:
                logger.info(
                    f"Processing queued episode",
                    extra={
                        'request_id': request_id,
                        'name': name,
                        'group_id': group_id_str
                    }
                )
                
                # Use all entity types if use_custom_entities is enabled, otherwise use empty dict
                entity_types = ENTITY_TYPES if config.use_custom_entities else {}

                # Process with timeout
                async with asyncio.timeout(30):  # 30 second timeout
                    await client.add_episode(
                        name=name,
                        episode_body=episode_body,
                        source=source_type,
                        source_description=source_description,
                        group_id=group_id_str,  # Using the string version of group_id
                        uuid=uuid,
                        reference_time=datetime.now(timezone.utc),
                        entity_types=entity_types,
                    )
                    
                logger.info(
                    f"Episode processed successfully",
                    extra={
                        'request_id': request_id,
                        'name': name,
                        'group_id': group_id_str
                    }
                )

                # Build communities after successful episode addition
                try:
                    logger.info(
                        f"Building communities",
                        extra={
                            'request_id': request_id,
                            'name': name,
                            'group_id': group_id_str
                        }
                    )
                    await client.build_communities()
                except Exception as e:
                    # Log but don't fail the operation if community building fails
                    logger.warning(
                        f"Failed to build communities",
                        extra={
                            'request_id': request_id,
                            'name': name,
                            'group_id': group_id_str,
                            'error': str(e),
                            'error_type': e.__class__.__name__
                        }
                    )
                
            except asyncio.TimeoutError:
                logger.error(
                    f"Episode processing timed out",
                    extra={
                        'request_id': request_id,
                        'name': name,
                        'group_id': group_id_str
                    }
                )
                raise EpisodeProcessingError("Episode processing timed out")
                
            except Exception as e:
                logger.error(
                    f"Error processing episode",
                    extra={
                        'request_id': request_id,
                        'name': name,
                        'group_id': group_id_str,
                        'error': str(e),
                        'error_type': e.__class__.__name__
                    },
                    exc_info=True
                )
                raise

        # Initialize queue for this group_id if it doesn't exist
        if group_id_str not in episode_queues:
            episode_queues[group_id_str] = asyncio.Queue()
            logger.info(
                f"Created new episode queue",
                extra={
                    'request_id': request_id,
                    'group_id': group_id_str
                }
            )

        # Add the processing function to the queue
        await episode_queues[group_id_str].put(process_episode)
        queue_size = episode_queues[group_id_str].qsize()
        logger.info(
            f"Added episode to queue",
            extra={
                'request_id': request_id,
                'group_id': group_id_str,
                'queue_size': queue_size
            }
        )

        # Start queue worker if not already running
        if not queue_workers.get(group_id_str, False):
            logger.info(
                f"Starting new queue worker",
                extra={
                    'request_id': request_id,
                    'group_id': group_id_str
                }
            )
            asyncio.create_task(process_episode_queue(group_id_str))

        return {
            'message': (
                f"Episode '{name}' queued successfully. "
                f"Current queue size: {queue_size}"
            )
        }

    except EpisodeValidationError as e:
        error_msg = f"Episode validation failed: {str(e)}"
        logger.error(
            error_msg,
            extra={
                'request_id': request_id,
                'name': name,
                'source': source,
                'error_type': 'EpisodeValidationError'
            }
        )
        return {'error': error_msg}

    except Exception as e:
        error_msg = f"Failed to queue episode: {str(e)}"
        logger.error(
            error_msg,
            extra={
                'request_id': request_id,
                'name': name,
                'source': source,
                'error_type': e.__class__.__name__
            },
            exc_info=True
        )
        return {'error': error_msg}


        # Start a worker for this queue if one isn't already running
        if not queue_workers.get(group_id_str, False):
            asyncio.create_task(process_episode_queue(group_id_str))

        # Return immediately with a success message
        return {
            'message': f"Episode '{name}' queued for processing (position: {episode_queues[group_id_str].qsize()})"
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode task: {error_msg}')
        return {'error': f'Error queuing episode task: {error_msg}'}


class SearchError(RetryableError):
    """Error during search operations that may be retryable."""
    pass

@mcp.tool()
@with_retry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=5.0,
    retryable_exceptions=(SearchError, *RETRYABLE_NETWORK_ERRORS),
)
@with_logging()
async def search_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    center_node_uuid: str | None = None,
    entity: str = '',  # cursor seems to break with None
) -> NodeSearchResponse | ErrorResponse:
    """Search the Graphiti knowledge graph for relevant node summaries.
    These contain a summary of all of a node's relationships with other nodes.

    Note: entity is a single entity type to filter results (permitted: "Preference", "Procedure").

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_nodes: Maximum number of nodes to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
        entity: Optional single entity type to filter results (permitted: "Preference", "Procedure")
    """
    global graphiti_client

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    if graphiti_client is None:
        logger.error(
            "Search nodes failed - Graphiti client not initialized",
            extra={'request_id': request_id}
        )
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Input validation
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if max_nodes < 1:
            raise ValueError("max_nodes must be greater than 0")

        if entity and entity not in ('Preference', 'Procedure'):
            raise ValueError("entity must be either 'Preference' or 'Procedure' if provided")

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        # Configure the search
        if center_node_uuid is not None:
            search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
        else:
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_nodes

        filters = SearchFilters()
        if entity != '':
            filters.node_labels = [entity]

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        logger.info(
            f"Performing node search",
            extra={
                'request_id': request_id,
                'query': query,
                'group_ids': effective_group_ids,
                'max_nodes': max_nodes,
                'center_node_uuid': center_node_uuid,
                'entity': entity
            }
        )

        try:
            # Perform the search using the _search method with timeout
            async with asyncio.timeout(10):  # 10 second timeout
                search_results = await client._search(
                    query=query,
                    config=search_config,
                    group_ids=effective_group_ids,
                    center_node_uuid=center_node_uuid,
                    search_filter=filters,
                )
        except asyncio.TimeoutError:
            logger.error(
                f"Search operation timed out",
                extra={
                    'request_id': request_id,
                    'query': query
                }
            )
            raise SearchError("Search operation timed out")
        except (ServiceUnavailable, SessionExpired) as e:
            logger.error(
                f"Database error during search",
                extra={
                    'request_id': request_id,
                    'error': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            raise SearchError(f"Database error: {str(e)}") from e

        if not search_results.nodes:
            logger.info(
                f"No nodes found matching search criteria",
                extra={
                    'request_id': request_id,
                    'query': query
                }
            )
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        # Format the node results
        formatted_nodes: list[NodeResult] = [
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

        logger.info(
            f"Search completed successfully",
            extra={
                'request_id': request_id,
                'query': query,
                'nodes_found': len(formatted_nodes)
            }
        )

        return NodeSearchResponse(
            message=f'Found {len(formatted_nodes)} relevant nodes',
            nodes=formatted_nodes
        )

    except ValueError as e:
        # Input validation errors - not retryable
        error_msg = str(e)
        logger.error(
            f"Invalid search parameters",
            extra={
                'request_id': request_id,
                'error': error_msg,
                'error_type': 'ValueError'
            }
        )
        return ErrorResponse(error=f'Invalid search parameters: {error_msg}')

    except Exception as e:
        # Unexpected errors - log with full context
        error_msg = str(e)
        logger.error(
            f"Unexpected error during search",
            extra={
                'request_id': request_id,
                'query': query,
                'error': error_msg,
                'error_type': e.__class__.__name__
            },
            exc_info=True
        )
        # Treat unexpected errors as retryable
        raise SearchError(f"Unexpected error: {error_msg}") from e


@mcp.tool()
@with_retry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=5.0,
    retryable_exceptions=(SearchError, *RETRYABLE_NETWORK_ERRORS),
)
@with_logging()
async def search_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the Graphiti knowledge graph for relevant facts.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    global graphiti_client

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    if graphiti_client is None:
        logger.error(
            "Search facts failed - Graphiti client not initialized",
            extra={'request_id': request_id}
        )
        return {'error': 'Graphiti client not initialized'}

    try:
        # Input validation
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if max_facts < 1:
            raise ValueError("max_facts must be greater than 0")

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        logger.info(
            f"Performing fact search",
            extra={
                'request_id': request_id,
                'query': query,
                'group_ids': effective_group_ids,
                'max_facts': max_facts,
                'center_node_uuid': center_node_uuid
            }
        )

        try:
            # Perform the search with timeout
            async with asyncio.timeout(10):  # 10 second timeout
                relevant_edges = await client.search(
                    group_ids=effective_group_ids,
                    query=query,
                    num_results=max_facts,
                    center_node_uuid=center_node_uuid,
                )
        except asyncio.TimeoutError:
            logger.error(
                f"Search operation timed out",
                extra={
                    'request_id': request_id,
                    'query': query
                }
            )
            raise SearchError("Search operation timed out")
        except (ServiceUnavailable, SessionExpired) as e:
            logger.error(
                f"Database error during search",
                extra={
                    'request_id': request_id,
                    'error': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            raise SearchError(f"Database error: {str(e)}") from e

        if not relevant_edges:
            logger.info(
                f"No facts found matching search criteria",
                extra={
                    'request_id': request_id,
                    'query': query
                }
            )
            return {'message': 'No relevant facts found', 'facts': []}

        # Format the results
        try:
            facts = [format_fact_result(edge) for edge in relevant_edges]
        except Exception as e:
            logger.error(
                f"Error formatting search results",
                extra={
                    'request_id': request_id,
                    'error': str(e),
                    'error_type': e.__class__.__name__
                },
                exc_info=True
            )
            raise SearchError(f"Error formatting results: {str(e)}") from e

        logger.info(
            f"Search completed successfully",
            extra={
                'request_id': request_id,
                'query': query,
                'facts_found': len(facts)
            }
        )

        return {
            'message': f'Found {len(facts)} relevant facts',
            'facts': facts
        }

    except ValueError as e:
        # Input validation errors - not retryable
        error_msg = str(e)
        logger.error(
            f"Invalid search parameters",
            extra={
                'request_id': request_id,
                'error': error_msg,
                'error_type': 'ValueError'
            }
        )
        return {'error': f'Invalid search parameters: {error_msg}'}

    except Exception as e:
        # Unexpected errors - log with full context
        error_msg = str(e)
        logger.error(
            f"Unexpected error during search",
            extra={
                'request_id': request_id,
                'query': query,
                'error': error_msg,
                'error_type': e.__class__.__name__
            },
            exc_info=True
        )
        # Treat unexpected errors as retryable
        raise SearchError(f"Unexpected error: {error_msg}") from e


class DeleteError(RetryableError):
    """Error during delete operations that may be retryable."""
    pass

@mcp.tool()
@with_retry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=5.0,
    retryable_exceptions=(DeleteError, *RETRYABLE_NETWORK_ERRORS),
)
@with_logging()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the Graphiti knowledge graph.

    Args:
        uuid: UUID of the entity edge to delete
    """
    global graphiti_client

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    if graphiti_client is None:
        logger.error(
            "Delete entity edge failed - Graphiti client not initialized",
            extra={'request_id': request_id}
        )
        return {'error': 'Graphiti client not initialized'}

    try:
        # Input validation
        if not uuid or not uuid.strip():
            raise ValueError("UUID cannot be empty")

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        logger.info(
            f"Deleting entity edge",
            extra={
                'request_id': request_id,
                'uuid': uuid
            }
        )

        try:
            # Get and delete the entity edge with timeout
            async with asyncio.timeout(10):  # 10 second timeout
                # Get the entity edge by UUID
                entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
                if entity_edge is None:
                    logger.warning(
                        f"Entity edge not found",
                        extra={
                            'request_id': request_id,
                            'uuid': uuid
                        }
                    )
                    return {'error': f'Entity edge with UUID {uuid} not found'}

                # Delete the edge using its delete method
                await entity_edge.delete(client.driver)

        except asyncio.TimeoutError:
            logger.error(
                f"Delete operation timed out",
                extra={
                    'request_id': request_id,
                    'uuid': uuid
                }
            )
            raise DeleteError("Delete operation timed out")
        except (ServiceUnavailable, SessionExpired) as e:
            logger.error(
                f"Database error during delete",
                extra={
                    'request_id': request_id,
                    'uuid': uuid,
                    'error': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            raise DeleteError(f"Database error: {str(e)}") from e

        logger.info(
            f"Entity edge deleted successfully",
            extra={
                'request_id': request_id,
                'uuid': uuid
            }
        )

        return {'message': f'Entity edge with UUID {uuid} deleted successfully'}

    except ValueError as e:
        # Input validation errors - not retryable
        error_msg = str(e)
        logger.error(
            f"Invalid delete parameters",
            extra={
                'request_id': request_id,
                'error': error_msg,
                'error_type': 'ValueError'
            }
        )
        return {'error': f'Invalid delete parameters: {error_msg}'}

    except Exception as e:
        # Unexpected errors - log with full context
        error_msg = str(e)
        logger.error(
            f"Unexpected error during delete",
            extra={
                'request_id': request_id,
                'uuid': uuid,
                'error': error_msg,
                'error_type': e.__class__.__name__
            },
            exc_info=True
        )
        # Treat unexpected errors as retryable
        raise DeleteError(f"Unexpected error: {error_msg}") from e


@mcp.tool()
@with_retry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=5.0,
    retryable_exceptions=(DeleteError, *RETRYABLE_NETWORK_ERRORS),
)
@with_logging()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the Graphiti knowledge graph.

    Args:
        uuid: UUID of the episode to delete
    """
    global graphiti_client

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    if graphiti_client is None:
        logger.error(
            "Delete episode failed - Graphiti client not initialized",
            extra={'request_id': request_id}
        )
        return {'error': 'Graphiti client not initialized'}

    try:
        # Input validation
        if not uuid or not uuid.strip():
            raise ValueError("UUID cannot be empty")

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        logger.info(
            f"Deleting episode",
            extra={
                'request_id': request_id,
                'uuid': uuid
            }
        )

        try:
            # Get and delete the episode with timeout
            async with asyncio.timeout(10):  # 10 second timeout
                # Get the episodic node by UUID
                episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
                if episodic_node is None:
                    logger.warning(
                        f"Episode not found",
                        extra={
                            'request_id': request_id,
                            'uuid': uuid
                        }
                    )
                    return {'error': f'Episode with UUID {uuid} not found'}

                # Delete the node using its delete method
                await episodic_node.delete(client.driver)

        except asyncio.TimeoutError:
            logger.error(
                f"Delete operation timed out",
                extra={
                    'request_id': request_id,
                    'uuid': uuid
                }
            )
            raise DeleteError("Delete operation timed out")
        except (ServiceUnavailable, SessionExpired) as e:
            logger.error(
                f"Database error during delete",
                extra={
                    'request_id': request_id,
                    'uuid': uuid,
                    'error': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            raise DeleteError(f"Database error: {str(e)}") from e

        logger.info(
            f"Episode deleted successfully",
            extra={
                'request_id': request_id,
                'uuid': uuid
            }
        )

        return {'message': f'Episode with UUID {uuid} deleted successfully'}

    except ValueError as e:
        # Input validation errors - not retryable
        error_msg = str(e)
        logger.error(
            f"Invalid delete parameters",
            extra={
                'request_id': request_id,
                'error': error_msg,
                'error_type': 'ValueError'
            }
        )
        return {'error': f'Invalid delete parameters: {error_msg}'}

    except Exception as e:
        # Unexpected errors - log with full context
        error_msg = str(e)
        logger.error(
            f"Unexpected error during delete",
            extra={
                'request_id': request_id,
                'uuid': uuid,
                'error': error_msg,
                'error_type': e.__class__.__name__
            },
            exc_info=True
        )
        # Treat unexpected errors as retryable
        raise DeleteError(f"Unexpected error: {error_msg}") from e


class GetError(RetryableError):
    """Error during get operations that may be retryable."""
    pass

@mcp.tool()
@with_retry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=5.0,
    retryable_exceptions=(GetError, *RETRYABLE_NETWORK_ERRORS),
)
@with_logging()
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the Graphiti knowledge graph by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    global graphiti_client

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    if graphiti_client is None:
        logger.error(
            "Get entity edge failed - Graphiti client not initialized",
            extra={'request_id': request_id}
        )
        return {'error': 'Graphiti client not initialized'}

    try:
        # Input validation
        if not uuid or not uuid.strip():
            raise ValueError("UUID cannot be empty")

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        logger.info(
            f"Getting entity edge",
            extra={
                'request_id': request_id,
                'uuid': uuid
            }
        )

        try:
            # Get the entity edge with timeout
            async with asyncio.timeout(10):  # 10 second timeout
                # Get the entity edge directly using the EntityEdge class method
                entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
                if entity_edge is None:
                    logger.warning(
                        f"Entity edge not found",
                        extra={
                            'request_id': request_id,
                            'uuid': uuid
                        }
                    )
                    return {'error': f'Entity edge with UUID {uuid} not found'}

                # Use the format_fact_result function to serialize the edge
                result = format_fact_result(entity_edge)

        except asyncio.TimeoutError:
            logger.error(
                f"Get operation timed out",
                extra={
                    'request_id': request_id,
                    'uuid': uuid
                }
            )
            raise GetError("Get operation timed out")
        except (ServiceUnavailable, SessionExpired) as e:
            logger.error(
                f"Database error during get",
                extra={
                    'request_id': request_id,
                    'uuid': uuid,
                    'error': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            raise GetError(f"Database error: {str(e)}") from e

        logger.info(
            f"Entity edge retrieved successfully",
            extra={
                'request_id': request_id,
                'uuid': uuid
            }
        )

        # Return the Python dict directly - MCP will handle serialization
        return result

    except ValueError as e:
        # Input validation errors - not retryable
        error_msg = str(e)
        logger.error(
            f"Invalid get parameters",
            extra={
                'request_id': request_id,
                'error': error_msg,
                'error_type': 'ValueError'
            }
        )
        return {'error': f'Invalid get parameters: {error_msg}'}

    except Exception as e:
        # Unexpected errors - log with full context
        error_msg = str(e)
        logger.error(
            f"Unexpected error during get",
            extra={
                'request_id': request_id,
                'uuid': uuid,
                'error': error_msg,
                'error_type': e.__class__.__name__
            },
            exc_info=True
        )
        # Treat unexpected errors as retryable
        raise GetError(f"Unexpected error: {error_msg}") from e


@mcp.tool()
@with_retry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=5.0,
    retryable_exceptions=(GetError, *RETRYABLE_NETWORK_ERRORS),
)
@with_logging()
async def get_episodes(
    group_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    """Get the most recent episodes for a specific group.

    Args:
        group_id: ID of the group to retrieve episodes from. If not provided, uses the default group_id.
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    global graphiti_client

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    if graphiti_client is None:
        logger.error(
            "Get episodes failed - Graphiti client not initialized",
            extra={'request_id': request_id}
        )
        return {'error': 'Graphiti client not initialized'}

    try:
        # Input validation
        if last_n < 1:
            raise ValueError("last_n must be greater than 0")

        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        if not isinstance(effective_group_id, str):
            raise ValueError('Group ID must be a string')

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        logger.info(
            f"Getting episodes",
            extra={
                'request_id': request_id,
                'group_id': effective_group_id,
                'last_n': last_n
            }
        )

        try:
            # Get episodes with timeout
            async with asyncio.timeout(10):  # 10 second timeout
                episodes = await client.retrieve_episodes(
                    group_ids=[effective_group_id],
                    last_n=last_n,
                    reference_time=datetime.now(timezone.utc)
                )

        except asyncio.TimeoutError:
            logger.error(
                f"Get operation timed out",
                extra={
                    'request_id': request_id,
                    'group_id': effective_group_id
                }
            )
            raise GetError("Get operation timed out")
        except (ServiceUnavailable, SessionExpired) as e:
            logger.error(
                f"Database error during get",
                extra={
                    'request_id': request_id,
                    'group_id': effective_group_id,
                    'error': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            raise GetError(f"Database error: {str(e)}") from e

        if not episodes:
            logger.info(
                f"No episodes found",
                extra={
                    'request_id': request_id,
                    'group_id': effective_group_id
                }
            )
            return {
                'message': f'No episodes found for group {effective_group_id}',
                'episodes': []
            }

        try:
            # Use Pydantic's model_dump method for EpisodicNode serialization
            formatted_episodes = [
                # Use mode='json' to handle datetime serialization
                episode.model_dump(mode='json')
                for episode in episodes
            ]
        except Exception as e:
            logger.error(
                f"Error formatting episodes",
                extra={
                    'request_id': request_id,
                    'group_id': effective_group_id,
                    'error': str(e),
                    'error_type': e.__class__.__name__
                },
                exc_info=True
            )
            raise GetError(f"Error formatting episodes: {str(e)}") from e

        logger.info(
            f"Episodes retrieved successfully",
            extra={
                'request_id': request_id,
                'group_id': effective_group_id,
                'episodes_found': len(formatted_episodes)
            }
        )

        # Return the Python list directly - MCP will handle serialization
        return formatted_episodes

    except ValueError as e:
        # Input validation errors - not retryable
        error_msg = str(e)
        logger.error(
            f"Invalid get parameters",
            extra={
                'request_id': request_id,
                'error': error_msg,
                'error_type': 'ValueError'
            }
        )
        return {'error': f'Invalid get parameters: {error_msg}'}

    except Exception as e:
        # Unexpected errors - log with full context
        error_msg = str(e)
        logger.error(
            f"Unexpected error during get",
            extra={
                'request_id': request_id,
                'group_id': effective_group_id if effective_group_id else 'None',
                'error': error_msg,
                'error_type': e.__class__.__name__
            },
            exc_info=True
        )
        # Treat unexpected errors as retryable
        raise GetError(f"Unexpected error: {error_msg}") from e


class ClearGraphError(RetryableError):
    """Error during graph clearing operations that may be retryable."""
    pass

@mcp.tool()
@with_retry(
    max_attempts=3,
    base_delay=2.0,
    max_delay=10.0,
    retryable_exceptions=(ClearGraphError, *RETRYABLE_NETWORK_ERRORS),
)
@with_logging()
async def clear_graph() -> SuccessResponse | ErrorResponse:
    """Clear all data from the Graphiti knowledge graph and rebuild indices.
    
    This is a potentially destructive operation that will:
    1. Clear all data from the graph
    2. Rebuild all indices and constraints
    
    The operation is retried up to 3 times in case of transient failures.
    """
    global graphiti_client

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    if graphiti_client is None:
        logger.error(
            "Clear graph failed - Graphiti client not initialized",
            extra={'request_id': request_id}
        )
        return {'error': 'Graphiti client not initialized'}

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        logger.info(
            f"Starting graph clear operation",
            extra={'request_id': request_id}
        )

        try:
            # Clear data with timeout
            async with asyncio.timeout(30):  # 30 second timeout for potentially large operation
                # clear_data is already imported at the top
                await clear_data(client.driver)

                logger.info(
                    f"Graph data cleared, rebuilding indices",
                    extra={'request_id': request_id}
                )

                # Rebuild indices
                await client.build_indices_and_constraints()

        except asyncio.TimeoutError:
            logger.error(
                f"Clear operation timed out",
                extra={'request_id': request_id}
            )
            raise ClearGraphError("Clear operation timed out")
        except (ServiceUnavailable, SessionExpired) as e:
            logger.error(
                f"Database error during clear",
                extra={
                    'request_id': request_id,
                    'error': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            raise ClearGraphError(f"Database error: {str(e)}") from e

        logger.info(
            f"Graph cleared and indices rebuilt successfully",
            extra={'request_id': request_id}
        )

        return {'message': 'Graph cleared successfully and indices rebuilt'}

    except Exception as e:
        # All errors in clear_graph are potentially retryable
        error_msg = str(e)
        logger.error(
            f"Error during graph clear",
            extra={
                'request_id': request_id,
                'error': error_msg,
                'error_type': e.__class__.__name__
            },
            exc_info=True
        )
        raise ClearGraphError(f"Error clearing graph: {error_msg}") from e


class StatusError(RetryableError):
    """Error during status check operations that may be retryable."""
    pass

@mcp.resource('http://graphiti/status')
@with_retry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=5.0,
    retryable_exceptions=(StatusError, *RETRYABLE_NETWORK_ERRORS),
)
@with_logging()
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and Neo4j connection.
    
    This endpoint performs several health checks:
    1. Verifies Graphiti client is initialized
    2. Tests Neo4j database connectivity
    3. Checks LLM client status if configured
    4. Checks embedder client status if configured
    
    The operation is retried up to 3 times in case of transient failures.
    """
    global graphiti_client

    # Generate request ID for tracking
    request_id = str(uuid.uuid4())

    if graphiti_client is None:
        logger.error(
            "Status check failed - Graphiti client not initialized",
            extra={'request_id': request_id}
        )
        return {
            'status': 'error',
            'message': 'Graphiti client not initialized'
        }

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        logger.info(
            f"Starting status check",
            extra={'request_id': request_id}
        )

        status_details = []
        has_errors = False

        try:
            # Test Neo4j connection with timeout
            async with asyncio.timeout(5):  # 5 second timeout
                await client.driver.verify_connectivity()
                status_details.append("Neo4j connection: OK")
        except asyncio.TimeoutError:
            has_errors = True
            status_details.append("Neo4j connection: TIMEOUT")
            logger.error(
                f"Neo4j connection check timed out",
                extra={'request_id': request_id}
            )
            raise StatusError("Neo4j connection check timed out")
        except (ServiceUnavailable, SessionExpired) as e:
            has_errors = True
            status_details.append(f"Neo4j connection: ERROR - {str(e)}")
            logger.error(
                f"Neo4j connection error",
                extra={
                    'request_id': request_id,
                    'error': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            raise StatusError(f"Neo4j connection error: {str(e)}") from e

        # Check LLM client status
        if client.llm_client:
            status_details.append(
                f"LLM client: OK (model: {config.llm.model})"
            )
        else:
            status_details.append("LLM client: NOT CONFIGURED")

        # Check embedder status
        if client.embedder:
            status_details.append(
                f"Embedder client: OK (model: {config.embedder.model})"
            )
        else:
            status_details.append("Embedder client: NOT CONFIGURED")

        # Check cross-encoder status
        if client.cross_encoder:
            status_details.append("Cross-encoder client: OK")
        else:
            status_details.append("Cross-encoder client: NOT CONFIGURED")

        # Log final status
        status = 'error' if has_errors else 'ok'
        message = 'Graphiti MCP server status:\n' + '\n'.join(status_details)

        logger.info(
            f"Status check completed",
            extra={
                'request_id': request_id,
                'status': status,
                'details': status_details
            }
        )

        return {
            'status': status,
            'message': message
        }

    except Exception as e:
        # Unexpected errors - log with full context
        error_msg = str(e)
        logger.error(
            f"Unexpected error during status check",
            extra={
                'request_id': request_id,
                'error': error_msg,
                'error_type': e.__class__.__name__
            },
            exc_info=True
        )
        return {
            'status': 'error',
            'message': f'Status check failed: {error_msg}'
        }


async def initialize_server() -> MCPConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config

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
    parser.add_argument(
        '--model', help=f'Model name to use with the LLM client. (default: {DEFAULT_LLM_MODEL})'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help='Temperature setting for the LLM (0.0-2.0). Lower values make output more deterministic. (default: 0.7)',
    )
    parser.add_argument('--destroy-graph', action='store_true', help='Destroy all Graphiti graphs')
    parser.add_argument(
        '--use-custom-entities',
        action='store_true',
        help='Enable entity extraction using the predefined ENTITY_TYPES',
    )

    args = parser.parse_args()

    # Build configuration from CLI arguments and environment variables
    config = GraphitiConfig.from_cli_and_env(args)

    # Log the group ID configuration
    if args.group_id:
        logger.info(f'Using provided group_id: {config.group_id}')
    else:
        logger.info(f'Generated random group_id: {config.group_id}')

    # Log entity extraction configuration
    if config.use_custom_entities:
        logger.info('Entity extraction enabled using predefined ENTITY_TYPES')
    else:
        logger.info('Entity extraction disabled (no custom entities will be used)')

    # Initialize Graphiti
    await initialize_graphiti()

    # Return MCP configuration
    return MCPConfig.from_cli(args)


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
