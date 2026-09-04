import logging
from collections.abc import Iterable
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.embedder.openai import OpenAIEmbedder  # type: ignore
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


class OpenRouterEmbedder(OpenAIEmbedder):
    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        result = await self.client.embeddings.create(
            input=input_data,  # type: ignore
            model=self.config.embedding_model,
            encoding_format='float',
        )
        return result.data[0].embedding[: self.config.embedding_dim]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        result = await self.client.embeddings.create(
            input=input_data_list,
            model=self.config.embedding_model,
            encoding_format='float',
        )
        return [embedding.embedding[: self.config.embedding_dim] for embedding in result.data]


_OLLAMA_BASE_URL = 'http://localhost:11434/v1'
_OLLAMA_DEFAULT_LLM = 'qwen2.5:7b'
_OLLAMA_DEFAULT_EMBED = 'nomic-embed-text'
_OLLAMA_DEFAULT_EMBED_DIM = 768


def _create_graphiti_client(settings: Settings) -> ZepGraphiti:
    """Create a ZepGraphiti client based on the configured database backend.

    Supports two LLM providers controlled by the ``llm_provider`` setting:
    - ``openrouter`` (default): Routes through OpenRouter.ai with the NVIDIA Nemotron
      models. Requires a custom embedder that forces ``encoding_format='float'``.
    - ``ollama``: Uses a locally running Ollama instance (http://localhost:11434/v1).
      Standard OpenAIEmbedder works here because Ollama supports float encoding natively.
    """
    llm_client = None
    embedder = None
    cross_encoder = None

    if settings.llm_provider == 'ollama':
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.embedder.openai import OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        ollama_url = settings.openai_base_url or _OLLAMA_BASE_URL
        ollama_key = settings.openai_api_key or 'ollama'
        llm_model = settings.model_name or _OLLAMA_DEFAULT_LLM
        embed_model = settings.embedding_model_name or _OLLAMA_DEFAULT_EMBED
        embed_dim = (
            settings.embedding_dim if settings.embedding_dim != 2048 else _OLLAMA_DEFAULT_EMBED_DIM
        )

        logger.info(
            'Ollama provider: llm=%s embed=%s dim=%d url=%s',
            llm_model,
            embed_model,
            embed_dim,
            ollama_url,
        )

        llm_config = LLMConfig(
            api_key=ollama_key,
            base_url=ollama_url,
            model=llm_model,
            small_model=llm_model,
        )
        # Ollama supports float encoding natively — use the plain OpenAIEmbedder
        embedder_config = OpenAIEmbedderConfig(
            api_key=ollama_key,
            base_url=ollama_url,
            embedding_model=embed_model,
            embedding_dim=embed_dim,
        )
        # Ollama >=0.5 supports native json_schema structured output (constrained
        # decoding). json_object mode makes smaller/local models echo the injected
        # schema back instead of filling it in — json_schema is far more reliable.
        llm_client = OpenAIGenericClient(config=llm_config, structured_output_mode='json_schema')
        embedder = OpenAIEmbedder(config=embedder_config)
        cross_encoder = OpenAIRerankerClient(config=llm_config)

    elif settings.openai_api_key or settings.openai_base_url:
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.embedder.openai import OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        llm_config = LLMConfig(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.model_name or 'nvidia/nemotron-3-super-120b-a12b:free',
            small_model=settings.model_name or 'nvidia/nemotron-3-super-120b-a12b:free',
        )
        llm_client = OpenAIGenericClient(config=llm_config, structured_output_mode='json_object')

        embedder_config = OpenAIEmbedderConfig(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            embedding_model=settings.embedding_model_name or 'nvidia/nemotron-3-embed-1b:free',
            embedding_dim=settings.embedding_dim or 2048,
        )
        embedder = OpenRouterEmbedder(config=embedder_config)
        cross_encoder = OpenAIRerankerClient(config=llm_config)

    if settings.db_backend == 'falkordb':
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver = FalkorDriver(  # type: ignore
            host=settings.falkordb_host or 'localhost',  # type: ignore
            port=settings.falkordb_port or 6379,  # type: ignore
            database=settings.falkordb_database or 'default_db',  # type: ignore
        )
        return ZepGraphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )  # type: ignore
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


_global_graphiti_client: ZepGraphiti | None = None


def get_graphiti_singleton(settings: Settings) -> ZepGraphiti:
    global _global_graphiti_client
    if _global_graphiti_client is None:
        _global_graphiti_client = _create_graphiti_client(settings)
    return _global_graphiti_client


async def close_graphiti_singleton():
    global _global_graphiti_client
    if _global_graphiti_client is not None:
        await _global_graphiti_client.close()
        _global_graphiti_client = None


async def get_graphiti(settings: ZepEnvDep):
    yield get_graphiti_singleton(settings)


async def initialize_graphiti(settings: ZepEnvDep):
    client = get_graphiti_singleton(settings)
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
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        episodes=edge.episodes or [],
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
