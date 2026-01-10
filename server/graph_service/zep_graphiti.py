import logging
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.driver.falkordb_driver import FalkorDriver  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: LLMClient | None = None,
        graph_driver=None,
    ):
        if graph_driver:
            super().__init__(graph_driver=graph_driver, llm_client=llm_client)
        else:
            super().__init__(uri, user, password, llm_client)

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


def _create_graphiti_client(settings: ZepEnvDep) -> ZepGraphiti:
    """Create a Graphiti client based on the configured database provider."""
    # Ensure we have a provider, defaulting to falkordb
    provider = (settings.database_provider or 'falkordb').lower().strip()
    
    if provider == 'falkordb':
        falkor_driver = FalkorDriver(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            password=settings.falkordb_password,
            database=settings.falkordb_database,
        )
        client = ZepGraphiti(graph_driver=falkor_driver)
    else:
        # Default to Neo4j for backward compatibility
        if not settings.neo4j_uri or not settings.neo4j_user or not settings.neo4j_password:
            raise ValueError(
                'Neo4j configuration is required when database_provider is neo4j. '
                'Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables.'
            )
        client = ZepGraphiti(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
    return client


async def get_graphiti(settings: ZepEnvDep):
    client = _create_graphiti_client(settings)
    api_key = settings.openai_api_key.strip() if settings.openai_api_key else None
    
    if settings.openai_base_url is not None:
        client.llm_client.config.base_url = settings.openai_base_url
    if api_key:
        client.llm_client.config.api_key = api_key
        # Also set the embedder API key - OpenAI embedder uses AsyncOpenAI client
        if hasattr(client.embedder, 'client') and hasattr(client.embedder.client, 'api_key'):
            client.embedder.client.api_key = api_key
        # Also update config if it exists
        if hasattr(client.embedder, 'config') and hasattr(client.embedder.config, 'api_key'):
            client.embedder.config.api_key = api_key
    if settings.model_name is not None:
        client.llm_client.model = settings.model_name
    if settings.embedding_model_name is not None and hasattr(client.embedder, 'config'):
        if hasattr(client.embedder.config, 'embedding_model'):
            client.embedder.config.embedding_model = settings.embedding_model_name

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = _create_graphiti_client(settings)
    await client.build_indices_and_constraints()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]


def get_graphiti_for_group(group_id: str, base_graphiti: ZepGraphiti) -> ZepGraphiti:
    """
    Get a Graphiti client for a specific group_id (organization).
    For FalkorDB, this uses the group_id as the database/graph name.
    For other providers, returns the base client.
    """
    import logging
    from graphiti_core.driver.driver import GraphProvider
    
    logger = logging.getLogger(__name__)
    
    if base_graphiti.driver.provider == GraphProvider.FALKORDB:
        # Clone the driver with the group_id as the database name
        cloned_driver = base_graphiti.driver.clone(database=group_id)
        logger.debug(f"Created driver clone for group_id={group_id}, database={getattr(cloned_driver, '_database', 'unknown')}")
        # Create a new ZepGraphiti instance with the cloned driver
        client = ZepGraphiti(graph_driver=cloned_driver, llm_client=base_graphiti.llm_client)
        # Copy embedder reference (needed for entity node operations)
        client.embedder = base_graphiti.embedder
        return client
    else:
        # For other providers, use the base client (they don't support multi-tenant graphs)
        return base_graphiti
