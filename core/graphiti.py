import asyncio
from datetime import datetime
import logging
from typing import Callable, Tuple, LiteralString
from neo4j import AsyncGraphDatabase

from core.nodes import EntityNode, EpisodicNode, Node
from core.edges import EntityEdge, Edge
from core.utils import bfs, similarity_search, fulltext_search, build_episodic_edges

logger = logging.getLogger(__name__)


class LLMConfig:
    """Configuration for the language model"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com",
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model


class Graphiti:
    def __init__(
        self, uri: str, user: str, password: str, llm_config: LLMConfig | None
    ):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.database = "neo4j"

        self.build_indices()

        if llm_config:
            self.llm_config = llm_config
        else:
            self.llm_config = None

    def close(self):
        self.driver.close()

    async def build_indices(self):
        index_queries: list[LiteralString] = [
            "CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)",
            "CREATE INDEX created_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.created_at)",
            "CREATE INDEX valid_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.valid_at)",
            "CREATE INDEX name_edge_index IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.name)",
            "CREATE INDEX created_at_edge_index IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.created_at)",
            "CREATE INDEX expired_at_edge_index IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.expired_at)",
            "CREATE INDEX valid_at_edge_index IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.valid_at)",
            "CREATE INDEX invalid_at_edge_index IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.invalid_at)",
        ]
        # Add the range indices
        for query in index_queries:
            await self.driver.execute_query(query)

        # Add the entity indices
        await self.driver.execute_query(
            """
            CREATE FULLTEXT INDEX name_and_summary IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.summary]
            """
        )

        await self.driver.execute_query(
            """
            CREATE VECTOR INDEX fact_embedding IF NOT EXISTS
            FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 1024,
             `vector.similarity_function`: 'cosine'
            }}
            """
        )

    async def retrieve_episodes(
        self, last_n: int, sources: list[str] | None = "messages"
    ) -> list[EpisodicNode]:
        """Retrieve the last n episodic nodes from the graph"""
        ...

    # Utility function, to be removed from this class
    async def clear_data(self): ...

    async def search(
        self, query: str, config
    ) -> (list)[Tuple[EntityNode, list[EntityEdge]]]:
        (vec_nodes, vec_edges) = similarity_search(query, embedder)
        (text_nodes, text_edges) = fulltext_search(query)

        nodes = vec_nodes.extend(text_nodes)
        edges = vec_edges.extend(text_edges)

        results = bfs(nodes, edges, k=1)

        episode_ids = ["Mode of episode ids"]

        episodes = get_episodes(episode_ids[:episode_count])

        return [(node, edges)], episodes

    async def get_relevant_schema(
        self, episode: EpisodicNode, previous_episodes: list[EpisodicNode]
    ) -> list[Tuple[EntityNode, list[EntityEdge]]]:
        pass

    # Call llm with the specified messages, and return the response
    # Will be used in the conjunction with a prompt library
    async def generate_llm_response(self, messages: list[any]) -> str: ...

    # Extract new edges from the episode
    async def extract_new_edges(
        self,
        episode: EpisodicNode,
        new_nodes: list[EntityNode],
        relevant_schema: dict[str, any],
        previous_episodes: list[EpisodicNode],
    ) -> list[EntityEdge]: ...

    # Extract new nodes from the episode
    async def extract_new_nodes(
        self,
        episode: EpisodicNode,
        relevant_schema: dict[str, any],
        previous_episodes: list[EpisodicNode],
    ) -> list[EntityNode]: ...

    # Invalidate edges that are no longer valid
    async def invalidate_edges(
        self,
        episode: EpisodicNode,
        new_nodes: list[EntityNode],
        new_edges: list[EntityEdge],
        relevant_schema: dict[str, any],
        previous_episodes: list[EpisodicNode],
    ): ...

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime = None,
        episode_type="string",
        success_callback: Callable | None = None,
        error_callback: Callable | None = None,
    ):
        """Process an episode and update the graph"""
        try:
            nodes: list[Node] = []
            edges: list[Edge] = []
            previous_episodes = await self.retrieve_episodes(last_n=3)
            episode = EpisodicNode()
            await episode.save(self.driver)
            relevant_schema = await self.retrieve_relevant_schema(episode.content)
            new_nodes = await self.extract_new_nodes(
                episode, relevant_schema, previous_episodes
            )
            nodes.extend(new_nodes)
            new_edges = await self.extract_new_edges(
                episode, new_nodes, relevant_schema, previous_episodes
            )
            edges.extend(new_edges)
            episodic_edges = build_episodic_edges(nodes, episode, datetime.now())
            edges.extend(episodic_edges)

            invalidated_edges = await self.invalidate_edges(
                episode, new_nodes, new_edges, relevant_schema, previous_episodes
            )

            edges.extend(invalidated_edges)

            await asyncio.gather(*[node.save(self.driver) for node in nodes])
            await asyncio.gather(*[edge.save(self.driver) for edge in edges])
            for node in nodes:
                if isinstance(node, EntityNode):
                    await node.update_summary(self.driver)
            if success_callback:
                await success_callback(episode)
        except Exception as e:
            if error_callback:
                await error_callback(episode, e)
            else:
                raise e
