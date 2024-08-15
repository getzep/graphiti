import asyncio
from datetime import datetime
import logging
from typing import Callable
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv
import os
from core.nodes import SemanticNode, EpisodicNode, Node
from core.edges import SemanticEdge, Edge
from core.utils import (
    build_episodic_edges,
    retrieve_relevant_schema,
    extract_new_edges,
    extract_new_nodes,
    clear_data,
    retrieve_episodes,
)
from core.llm_client import LLMClient, OpenAIClient, LLMConfig

logger = logging.getLogger(__name__)

load_dotenv()


class Graphiti:
    def __init__(
        self, uri: str, user: str, password: str, llm_client: LLMClient | None = None
    ):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.database = "neo4j"
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = OpenAIClient(
                LLMConfig(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model="gpt-4o",
                    base_url="https://api.openai.com/v1",
                )
            )

    def close(self):
        self.driver.close()

    async def retrieve_episodes(
        self, last_n: int, sources: list[str] | None = "messages"
    ) -> list[EpisodicNode]:
        """Retrieve the last n episodic nodes from the graph"""
        return await retrieve_episodes(self.driver, last_n, sources)

    # Utility function, to be removed from this class
    async def clear_data(self):
        await clear_data(self.driver)

    async def retrieve_relevant_schema(self, query: str = None) -> dict[str, any]:
        """Retrieve relevant nodes and edges to a specific query"""
        return await retrieve_relevant_schema(self.driver, query)
        ...

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
            episode = EpisodicNode(
                name=name,
                labels=[],
                source="messages",
                content=episode_body,
                source_description=source_description,
                transaction_from=datetime.now(),
                valid_from=reference_time,
                semantic_edges=[],
            )
            await episode.save(self.driver)
            relevant_schema = await self.retrieve_relevant_schema(episode.content)
            new_nodes = await extract_new_nodes(
                self.llm_client, episode, relevant_schema, previous_episodes
            )
            nodes.extend(new_nodes)
            new_edges = await extract_new_edges(
                self.llm_client, episode, new_nodes, relevant_schema, previous_episodes
            )
            edges.extend(new_edges)
            episodic_edges = build_episodic_edges(nodes, episode, datetime.now())
            edges.extend(episodic_edges)

            # invalidated_edges = await self.invalidate_edges(
            #     episode, new_nodes, new_edges, relevant_schema, previous_episodes
            # )

            # edges.extend(invalidated_edges)
            # Future optimization would be using batch operations to save nodes and edges
            await asyncio.gather(*[node.save(self.driver) for node in nodes])
            await asyncio.gather(*[edge.save(self.driver) for edge in edges])
            # for node in nodes:
            #     if isinstance(node, SemanticNode):
            #         await node.update_summary(self.driver)
            if success_callback:
                await success_callback(episode)
        except Exception as e:
            if error_callback:
                await error_callback(episode, e)
            else:
                raise e
