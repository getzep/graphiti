import asyncio
from datetime import datetime
import logging
from typing import Callable, LiteralString, Tuple
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv
import os
from core.nodes import EntityNode, EpisodicNode, Node
from core.edges import EntityEdge, Edge
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

    async def retrieve_relevant_schema(self, query: str = None) -> dict[str, any]:
        """Retrieve relevant nodes and edges to a specific query"""
        return await retrieve_relevant_schema(self.driver, query)
        ...

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
            episode = EpisodicNode(
                name=name,
                labels=[],
                source="messages",
                content=episode_body,
                source_description=source_description,
                created_at=datetime.now(),
                valid_at=reference_time,
            )
            # await episode.save(self.driver)
            relevant_schema = await self.retrieve_relevant_schema(episode.content)
            new_nodes = await extract_new_nodes(
                self.llm_client, episode, relevant_schema, previous_episodes
            )
            nodes.extend(new_nodes)
            new_edges, affected_nodes = await extract_new_edges(
                self.llm_client, episode, new_nodes, relevant_schema, previous_episodes
            )
            edges.extend(new_edges)
            episodic_edges = build_episodic_edges(
                # There may be an overlap between new_nodes and affected_nodes, so we're deduplicating them
                list(set(nodes + affected_nodes)),
                episode,
                datetime.now(),
            )
            # Important to append the episode to the nodes at the end so that self referencing episodic edges are not built
            nodes.append(episode)
            logger.info(f"Built episodic edges: {episodic_edges}")
            edges.extend(episodic_edges)

            # invalidated_edges = await self.invalidate_edges(
            #     episode, new_nodes, new_edges, relevant_schema, previous_episodes
            # )

            # edges.extend(invalidated_edges)
            # Future optimization would be using batch operations to save nodes and edges
            await asyncio.gather(*[node.save(self.driver) for node in nodes])
            await asyncio.gather(*[edge.save(self.driver) for edge in edges])
            # for node in nodes:
            #     if isinstance(node, EntityNode):
            #         await node.update_summary(self.driver)
            if success_callback:
                await success_callback(episode)
        except Exception as e:
            if error_callback:
                await error_callback(episode, e)
            else:
                raise e

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

    async def search(
        self, query: str, config
    ) -> (list)[tuple[EntityNode, list[EntityEdge]]]:
        (vec_nodes, vec_edges) = similarity_search(query, embedder)
        (text_nodes, text_edges) = fulltext_search(query)

        nodes = vec_nodes.extend(text_nodes)
        edges = vec_edges.extend(text_edges)

        results = bfs(nodes, edges, k=1)

        episode_ids = ["Mode of episode ids"]

        episodes = get_episodes(episode_ids[:episode_count])

        return [(node, edges)], episodes

    # Invalidate edges that are no longer valid
    async def invalidate_edges(
        self,
        episode: EpisodicNode,
        new_nodes: list[EntityNode],
        new_edges: list[EntityEdge],
        relevant_schema: dict[str, any],
        previous_episodes: list[EpisodicNode],
    ): ...
