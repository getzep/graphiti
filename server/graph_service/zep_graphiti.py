import logging
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import ZepEnvDep, get_settings
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)

# Singleton graphiti client - stays open for app lifecycle
# This is required because background workers need access to the client
# after the original request has completed
_graphiti_client: 'ZepGraphiti | None' = None


class ZepGraphiti(Graphiti):
    def __init__(self, uri: str, user: str, password: str, llm_client: LLMClient | None = None):
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


def get_graphiti() -> 'ZepGraphiti':
    """Return the singleton graphiti client.

    The client stays open for the entire app lifecycle to support
    background workers that run after requests complete.
    """
    global _graphiti_client
    if _graphiti_client is None:
        raise RuntimeError('Graphiti client not initialized. Call initialize_graphiti() first.')
    return _graphiti_client


async def initialize_graphiti(settings: ZepEnvDep):
    """Initialize the singleton graphiti client on app startup.

    Creates a single client that stays open for the app lifecycle.
    This fixes the 'Driver closed' error that occurred when background
    workers tried to use request-scoped clients after the request ended.
    """
    global _graphiti_client

    if _graphiti_client is not None:
        logger.warning('Graphiti client already initialized, closing existing client')
        await _graphiti_client.close()

    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )

    # Configure LLM client if settings provided
    if settings.openai_base_url is not None:
        client.llm_client.config.base_url = settings.openai_base_url
    if settings.openai_api_key is not None:
        client.llm_client.config.api_key = settings.openai_api_key
    if settings.model_name is not None:
        client.llm_client.model = settings.model_name

    # Build indices and wait for completion
    await client.build_indices_and_constraints()

    _graphiti_client = client
    logger.info('Graphiti client initialized successfully')


async def shutdown_graphiti():
    """Shutdown the singleton graphiti client on app shutdown."""
    global _graphiti_client
    if _graphiti_client is not None:
        await _graphiti_client.close()
        _graphiti_client = None
        logger.info('Graphiti client closed')


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


# Dependency that returns the singleton client
# No cleanup needed since client stays open for app lifecycle
ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
