import asyncio
from datetime import datetime
import logging
from typing import Callable, Tuple
from neo4j import AsyncGraphDatabase

from core.nodes import SemanticNode, EpisodicNode, Node
from core.edges import SemanticEdge, Edge
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
        if llm_config:
            self.llm_config = llm_config
        else:
            self.llm_config = None

    def close(self):
        self.driver.close()

    async def retrieve_episodes(
        self, last_n: int, sources: list[str] | None = "messages"
    ) -> list[EpisodicNode]:
        """Retrieve the last n episodic nodes from the graph"""
        ...

    # Utility function, to be removed from this class
    async def clear_data(self): ...

    async def search(self, query: str, config) -> (
            list)[Tuple[SemanticNode, list[SemanticEdge]]]:
        (vec_nodes, vec_edges) = similarity_search(query, embedder)
        (text_nodes, text_edges) = fulltext_search(query)

        nodes = vec_nodes.extend(text_nodes)
        edges = vec_edges.extend(text_edges)

        results = bfs(nodes, edges, k=1)

        episode_ids = ["Mode of episode ids"]

        episodes = get_episodes(episode_ids[:episode_count])

        return [(node, edges)], episodes

    async def get_relevant_schema(self, episode: EpisodicNode, previous_episodes: list[EpisodicNode]) -> (
            list)[Tuple[SemanticNode, list[SemanticEdge]]]:
        pass

    # Call llm with the specified messages, and return the response
    # Will be used in the conjunction with a prompt library
    async def generate_llm_response(self, messages: list[any]) -> str: ...

    # Extract new edges from the episode
    async def extract_new_edges(
        self,
        episode: EpisodicNode,
        new_nodes: list[SemanticNode],
        relevant_schema: dict[str, any],
        previous_episodes: list[EpisodicNode],
    ) -> list[SemanticEdge]: ...

    # Extract new nodes from the episode
    async def extract_new_nodes(
        self,
        episode: EpisodicNode,
        relevant_schema: dict[str, any],
        previous_episodes: list[EpisodicNode],
    ) -> list[SemanticNode]: ...

    # Invalidate edges that are no longer valid
    async def invalidate_edges(
        self,
        episode: EpisodicNode,
        new_nodes: list[SemanticNode],
        new_edges: list[SemanticEdge],
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
                if isinstance(node, SemanticNode):
                    await node.update_summary(self.driver)
            if success_callback:
                await success_callback(episode)
        except Exception as e:
            if error_callback:
                await error_callback(episode, e)
            else:
                raise e
