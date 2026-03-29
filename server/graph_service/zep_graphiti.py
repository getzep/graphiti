import logging
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.embedder import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient, LLMConfig, OpenAIClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
    ):
        super().__init__(
            uri=uri,
            user=user,
            password=password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
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


async def get_graphiti(settings: ZepEnvDep):
    llm_config = LLMConfig(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model_name,
        small_model=settings.small_model_name,
    )
    embedder_config = OpenAIEmbedderConfig(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        embedding_model=settings.embedding_model_name or OpenAIEmbedderConfig().embedding_model,
    )
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=OpenAIClient(config=llm_config),
        embedder=OpenAIEmbedder(config=embedder_config),
    )

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    llm_config = LLMConfig(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model_name,
        small_model=settings.small_model_name,
    )
    embedder_config = OpenAIEmbedderConfig(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        embedding_model=settings.embedding_model_name or OpenAIEmbedderConfig().embedding_model,
    )
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=OpenAIClient(config=llm_config),
        embedder=OpenAIEmbedder(config=embedder_config),
    )
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
