import logging
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient, LLMConfig, OpenAIClient  # type: ignore
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
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


def _create_openai_clients(
    settings: Settings,
) -> tuple[OpenAIClient, OpenAIEmbedder, OpenAIRerankerClient]:
    """Build the OpenAI-backed clients Graphiti needs, configured from settings.

    These have to be constructed with the settings rather than patched afterwards: each
    client builds its AsyncOpenAI in __init__ out of config.api_key and config.base_url,
    so assigning to the config later leaves the already-built client pointing elsewhere.
    """
    api_key = settings.openai_api_key
    base_url = settings.openai_base_url

    embedder_config = OpenAIEmbedderConfig(api_key=api_key, base_url=base_url)
    # Assigned only when set, rather than passed to the constructor: embedding_model
    # defaults to a real model name, so handing it None would overwrite that default.
    # LLMConfig.model below defaults to None, which is why it can be passed straight in.
    if settings.embedding_model_name is not None:
        embedder_config.embedding_model = settings.embedding_model_name

    return (
        OpenAIClient(
            config=LLMConfig(api_key=api_key, base_url=base_url, model=settings.model_name)
        ),
        OpenAIEmbedder(config=embedder_config),
        # No model override here. The reranker scores passages off logprobs on a one-token
        # completion, which is a different job from extraction — leave it on its default.
        OpenAIRerankerClient(config=LLMConfig(api_key=api_key, base_url=base_url)),
    )


def _create_graphiti_client(settings: Settings) -> ZepGraphiti:
    """Create a ZepGraphiti client based on the configured database backend."""
    llm_client, embedder, cross_encoder = _create_openai_clients(settings)

    if settings.db_backend == 'falkordb':
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(
            host=settings.falkordb_host or 'localhost',
            port=settings.falkordb_port or 6379,
            username=settings.falkordb_username,
            password=settings.falkordb_password,
            database=settings.falkordb_database or 'default_db',
        )
        return ZepGraphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )
    else:
        # Validate Neo4j settings are present
        if not all([settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password]):
            raise ValueError(
                'Neo4j configuration (neo4j_uri, neo4j_user, neo4j_password) is required '
                "when db_backend is 'neo4j'"
            )
        return ZepGraphiti(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )


async def get_graphiti(settings: ZepEnvDep):
    client = _create_graphiti_client(settings)
    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = _create_graphiti_client(settings)
    try:
        await client.build_indices_and_constraints()
    finally:
        await client.close()


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
