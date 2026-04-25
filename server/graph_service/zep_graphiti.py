import logging
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.driver.driver import GraphDriver  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
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
        graph_driver: GraphDriver | None = None,
    ):
        super().__init__(
            uri=uri, user=user, password=password, llm_client=llm_client, graph_driver=graph_driver
        )

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


def _build_graph_driver(settings: Settings) -> GraphDriver | None:
    provider = settings.graph_db_provider

    if provider == 'neo4j':
        return None  # Graphiti defaults to Neo4jDriver when no graph_driver is provided

    if provider == 'falkordb':
        from graphiti_core.driver.falkordb_driver import FalkorDriver  # type: ignore

        return FalkorDriver(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            username=settings.falkordb_username,
            password=settings.falkordb_password,
            database=settings.falkordb_database,
        )

    if provider == 'neptune':
        from graphiti_core.driver.neptune_driver import NeptuneDriver  # type: ignore

        if not settings.neptune_host or not settings.aoss_host:
            raise ValueError(
                'NEPTUNE_HOST and AOSS_HOST are required when GRAPH_DB_PROVIDER is neptune'
            )
        return NeptuneDriver(
            host=settings.neptune_host,
            aoss_host=settings.aoss_host,
            port=settings.neptune_port,
            aoss_port=settings.aoss_port,
        )

    if provider == 'kuzu':
        from graphiti_core.driver.kuzu_driver import KuzuDriver  # type: ignore

        return KuzuDriver(
            db=settings.kuzu_db,
            max_concurrent_queries=settings.kuzu_max_concurrent_queries,
        )

    raise ValueError(f'Unknown graph_db_provider: {provider}')


def _build_client(settings: Settings) -> ZepGraphiti:
    graph_driver = _build_graph_driver(settings)
    if graph_driver is not None:
        return ZepGraphiti(graph_driver=graph_driver)
    return ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )


async def get_graphiti(settings: ZepEnvDep):
    client = _build_client(settings)
    if settings.openai_base_url is not None:
        client.llm_client.config.base_url = settings.openai_base_url
    if settings.openai_api_key is not None:
        client.llm_client.config.api_key = settings.openai_api_key
    if settings.model_name is not None:
        client.llm_client.model = settings.model_name

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    client = _build_client(settings)
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
