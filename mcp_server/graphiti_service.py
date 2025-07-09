"""Graphiti service for managing client lifecycle and operations."""

import logging

from config_manager import GraphitiConfig

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

logger = logging.getLogger(__name__)


class GraphitiService:
    """Service for managing Graphiti client operations."""

    def __init__(self, config: GraphitiConfig, semaphore_limit: int = 10):
        """Initialize the Graphiti service with configuration.

        Args:
            config: The Graphiti configuration
            semaphore_limit: Maximum concurrent operations
        """
        self.config = config
        self.semaphore_limit = semaphore_limit
        self._client: Graphiti | None = None

    @property
    def client(self) -> Graphiti:
        """Get the Graphiti client instance.

        Raises:
            RuntimeError: If the client hasn't been initialized
        """
        if self._client is None:
            raise RuntimeError('Graphiti client not initialized. Call initialize() first.')
        return self._client

    async def initialize(self) -> None:
        """Initialize the Graphiti client with the configured settings."""
        try:
            # Create LLM client if possible
            llm_client = self.config.llm.create_client()
            if not llm_client and self.config.use_custom_entities:
                # If custom entities are enabled, we must have an LLM client
                raise ValueError('OPENAI_API_KEY must be set when custom entities are enabled')

            # Validate Neo4j configuration
            if (
                not self.config.neo4j.uri
                or not self.config.neo4j.user
                or not self.config.neo4j.password
            ):
                raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

            embedder_client = self.config.embedder.create_client()

            # Initialize Graphiti client
            self._client = Graphiti(
                uri=self.config.neo4j.uri,
                user=self.config.neo4j.user,
                password=self.config.neo4j.password,
                llm_client=llm_client,
                embedder=embedder_client,
                max_coroutines=self.semaphore_limit,
            )

            # Destroy graph if requested
            if self.config.destroy_graph:
                logger.info('Destroying graph...')
                await clear_data(self._client.driver)

            # Initialize the graph database with Graphiti's indices
            await self._client.build_indices_and_constraints()
            logger.info('Graphiti client initialized successfully')

            # Log configuration details for transparency
            if llm_client:
                logger.info(f'Using OpenAI model: {self.config.llm.model}')
                logger.info(f'Using temperature: {self.config.llm.temperature}')
            else:
                logger.info('No LLM client configured - entity extraction will be limited')

            logger.info(f'Using group_id: {self.config.group_id}')
            logger.info(
                f'Custom entity extraction: {"enabled" if self.config.use_custom_entities else "disabled"}'
            )
            logger.info(f'Using concurrency limit: {self.semaphore_limit}')

        except Exception as e:
            logger.error(f'Failed to initialize Graphiti: {str(e)}')
            raise

    async def clear_graph(self) -> None:
        """Clear all data from the graph and rebuild indices."""
        if self._client is None:
            raise RuntimeError('Graphiti client not initialized')

        await clear_data(self._client.driver)
        await self._client.build_indices_and_constraints()

    async def verify_connection(self) -> None:
        """Verify the database connection."""
        if self._client is None:
            raise RuntimeError('Graphiti client not initialized')

        await self._client.driver.client.verify_connectivity()  # type: ignore

    def is_initialized(self) -> bool:
        """Check if the client is initialized."""
        return self._client is not None
