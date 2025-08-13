import logging
import os
from typing import Annotated

from fastapi import Depends, HTTPException
from openai import AsyncAzureOpenAI

from graphiti_core import Graphiti  # type: ignore
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient, LLMConfig, OpenAIClient  # type: ignore
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
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
        embedder: OpenAIEmbedder | None = None,
        cross_encoder: OpenAIRerankerClient | None = None,
    ):
        super().__init__(uri, user, password, llm_client, embedder, cross_encoder)

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
    # Default model names
    DEFAULT_LLM_MODEL = 'gpt-4o-mini'
    SMALL_LLM_MODEL = 'gpt-4o-mini'
            
    # Get model from settings, or use default if not set or empty
    model = settings.model_name if settings.model_name and settings.model_name.strip() else DEFAULT_LLM_MODEL
    small_model = settings.small_model_name if settings.small_model_name and settings.small_model_name.strip() else SMALL_LLM_MODEL

    api_key = settings.api_key
    api_version = settings.api_version
    llm_endpoint = settings.llm_endpoint
    embedding_endpoint = settings.embedding_endpoint
    embedding_model = settings.embedding_model

    # Create separate Azure OpenAI clients for different services
    llm_client_azure = AsyncAzureOpenAI(
        api_key=api_key, api_version=api_version, azure_endpoint=llm_endpoint
    )

    embedding_client_azure = AsyncAzureOpenAI(
        api_key=api_key, api_version=api_version, azure_endpoint=embedding_endpoint
    )

    # Create LLM Config with your Azure deployment names
    azure_llm_config = LLMConfig(
        small_model=small_model,
        model=model,
    )

    llm_client = OpenAIClient(config=azure_llm_config, client=llm_client_azure)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            embedding_model=embedding_model  # Your Azure embedding deployment name
        ),
        client=embedding_client_azure,
    )
    cross_encoder = OpenAIRerankerClient(
        config=LLMConfig(
            model=azure_llm_config.small_model  # Use small model for reranking
        ),
        client=llm_client_azure,
    )

    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    # Default model names
    DEFAULT_LLM_MODEL = 'gpt-4o-mini'
    SMALL_LLM_MODEL = 'gpt-4o-mini'

    # Get model from settings, or use default if not set or empty
    model = settings.model_name if settings.model_name and settings.model_name.strip() else DEFAULT_LLM_MODEL
    small_model = settings.small_model_name if settings.small_model_name and settings.small_model_name.strip() else SMALL_LLM_MODEL

    api_key = settings.api_key
    api_version = settings.api_version
    llm_endpoint = settings.llm_endpoint
    embedding_endpoint = settings.embedding_endpoint
    embedding_model = settings.embedding_model

    # Create separate Azure OpenAI clients for different services
    llm_client_azure = AsyncAzureOpenAI(
        api_key=api_key, api_version=api_version, azure_endpoint=llm_endpoint
    )

    embedding_client_azure = AsyncAzureOpenAI(
        api_key=api_key, api_version=api_version, azure_endpoint=embedding_endpoint
    )

    # Create LLM Config with your Azure deployment names
    azure_llm_config = LLMConfig(
        small_model=small_model,
        model=model,
    )

    llm_client = OpenAIClient(config=azure_llm_config, client=llm_client_azure)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            embedding_model=embedding_model  # Your Azure embedding deployment name
        ),
        client=embedding_client_azure,
    )
    cross_encoder = OpenAIRerankerClient(
        config=LLMConfig(
            model=azure_llm_config.small_model  # Use small model for reranking
        ),
        client=llm_client_azure,
    )
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
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
