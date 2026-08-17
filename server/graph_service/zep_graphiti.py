import logging
from typing import Annotated
from urllib.parse import urlparse

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.cross_encoder import LexicalRerankerClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.embedder import (
    EmbedderClient,
    LocalHashEmbedder,
    LocalHashEmbedderConfig,
    OpenAIEmbedder,
    OpenAIEmbedderConfig,
)
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient, OpenAIClient  # type: ignore
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import Settings, ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: LLMClient | None = None,
        **kwargs,
    ):
        super().__init__(uri, user, password, llm_client, **kwargs)  # type: ignore

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
            await self.remove_episode(uuid)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


def _is_non_openai_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False

    parsed = urlparse(base_url)
    hostname = (parsed.hostname or '').lower()
    return parsed.scheme in ('http', 'https') and bool(hostname) and hostname != 'api.openai.com'


def is_llm_configured(settings: Settings) -> bool:
    """Return whether settings are sufficient to construct the configured LLM client."""
    if not settings.openai_api_key or not settings.openai_api_key.strip():
        return False
    if not settings.model_name or not settings.model_name.strip():
        return False
    if settings.openai_base_url:
        parsed_base_url = urlparse(settings.openai_base_url)
        if parsed_base_url.scheme not in ('http', 'https') or not parsed_base_url.hostname:
            return False
    if settings.model_name.startswith(('ep-', 'doubao-')):
        return _is_non_openai_base_url(settings.openai_base_url)
    return True


def _create_llm_client(settings: Settings) -> LLMClient:
    if not settings.openai_api_key or not settings.openai_api_key.strip():
        raise ValueError('ARK_API_KEY or OPENAI_API_KEY is required for graph extraction')
    if not settings.model_name or not settings.model_name.strip():
        raise ValueError(
            'ARK_CHAT_MODEL, OPENAI_MODEL, or MODEL_NAME is required for graph extraction'
        )
    if settings.openai_base_url:
        parsed_base_url = urlparse(settings.openai_base_url)
        if parsed_base_url.scheme not in ('http', 'https') or not parsed_base_url.hostname:
            raise ValueError('ARK_BASE_URL or OPENAI_BASE_URL must be an absolute HTTP(S) URL')
    if settings.model_name.startswith(('ep-', 'doubao-')) and not _is_non_openai_base_url(
        settings.openai_base_url
    ):
        raise ValueError(
            'ARK_BASE_URL must point to an Ark-compatible API when using an Ark model or endpoint'
        )

    config = LLMConfig(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model_name,
        small_model=settings.model_name,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    if _is_non_openai_base_url(settings.openai_base_url):
        return OpenAIGenericClient(
            config=config,
            max_tokens=settings.llm_max_tokens,
            structured_output_mode=settings.structured_output_mode,
        )
    return OpenAIClient(config=config)


def _create_embedder(settings: Settings) -> EmbedderClient:
    if settings.embedding_provider == 'local_hash':
        return LocalHashEmbedder(LocalHashEmbedderConfig(embedding_dim=settings.embedding_dim))

    api_key = settings.embedding_api_key or settings.openai_api_key
    base_url = settings.embedding_base_url or settings.openai_base_url
    if not api_key:
        raise ValueError(
            'ARK_EMBEDDING_API_KEY, OPENAI_EMBEDDING_API_KEY, ARK_API_KEY, '
            'or OPENAI_API_KEY is required when '
            "EMBEDDING_PROVIDER='openai'"
        )
    if not settings.embedding_model_name:
        raise ValueError(
            'ARK_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL, or EMBEDDING_MODEL_NAME '
            "is required when EMBEDDING_PROVIDER='openai'"
        )
    return OpenAIEmbedder(
        OpenAIEmbedderConfig(
            api_key=api_key,
            base_url=base_url,
            embedding_model=settings.embedding_model_name,
            embedding_dim=settings.embedding_dim,
        )
    )


def _create_graph_driver(settings: Settings) -> GraphDriver:
    """Create and validate the configured graph database driver."""
    if settings.db_backend == 'falkordb':
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        return FalkorDriver(  # type: ignore
            host=settings.falkordb_host or 'localhost',  # type: ignore
            port=settings.falkordb_port or 6379,  # type: ignore
            database=settings.falkordb_database or 'default_db',  # type: ignore
        )

    uri = settings.neo4j_uri
    user = settings.neo4j_user
    password = settings.neo4j_password
    database = settings.neo4j_database
    if not uri or not user or not password or not database:
        raise ValueError(
            'Neo4j configuration (neo4j_uri, neo4j_user, neo4j_password, '
            'neo4j_database) is required '
            "when db_backend is 'neo4j'"
        )
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    return Neo4jDriver(uri, user, password, database=database)


def create_graphiti_client(settings: Settings) -> ZepGraphiti:
    """Create a fully configured Graphiti client for ingest and retrieval operations."""
    llm_client = _create_llm_client(settings)
    embedder = _create_embedder(settings)
    driver = _create_graph_driver(settings)
    return ZepGraphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=LexicalRerankerClient(),
    )


async def get_graphiti(settings: ZepEnvDep):
    client = create_graphiti_client(settings)

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    """Initialize only the database so service health is independent of LLM credentials."""
    driver = _create_graph_driver(settings)
    try:
        await driver.build_indices_and_constraints()
    finally:
        await driver.close()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        episodes=edge.episodes or [],
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
