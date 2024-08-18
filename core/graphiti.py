import asyncio
from datetime import datetime
import logging
from typing import Callable, LiteralString
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv
import os

from core.llm_client.config import EMBEDDING_DIM
from core.nodes import EntityNode, EpisodicNode, Node
from core.edges import EntityEdge, Edge, EpisodicEdge
from core.utils import (
    build_episodic_edges,
    retrieve_relevant_schema,
    extract_new_edges,
    extract_new_nodes,
    clear_data,
    retrieve_episodes,
)
from core.llm_client import LLMClient, OpenAIClient, LLMConfig
from core.utils.maintenance.edge_operations import extract_edges, dedupe_extracted_edges
from core.utils.maintenance.node_operations import dedupe_extracted_nodes, extract_nodes
from core.utils.search.search_utils import (
    edge_similarity_search,
    entity_fulltext_search,
    bfs,
    get_relevant_nodes,
    get_relevant_edges,
)

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
                    model="gpt-4o-mini",
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
            nodes: list[EntityNode] = []
            entity_edges: list[EntityEdge] = []
            episodic_edges: list[EpisodicEdge] = []
            embedder = self.llm_client.client.embeddings
            now = datetime.now()

            previous_episodes = await self.retrieve_episodes(last_n=3)
            episode = EpisodicNode(
                name=name,
                labels=[],
                source="messages",
                content=episode_body,
                source_description=source_description,
                created_at=now,
                valid_at=reference_time,
            )
            # relevant_schema = await self.retrieve_relevant_schema(episode.content)

            extracted_nodes = await extract_nodes(
                self.llm_client, episode, previous_episodes
            )

            # Calculate Embeddings

            await asyncio.gather(
                *[node.generate_name_embedding(embedder) for node in extracted_nodes]
            )
            existing_nodes = await get_relevant_nodes(extracted_nodes, self.driver)
            logger.info(
                f"Extracted nodes: {[(n.name, n.uuid) for n in extracted_nodes]}"
            )
            new_nodes = await dedupe_extracted_nodes(
                self.llm_client, extracted_nodes, existing_nodes
            )
            logger.info(
                f"Deduped touched nodes: {[(n.name, n.uuid) for n in new_nodes]}"
            )
            nodes.extend(new_nodes)

            extracted_edges = await extract_edges(
                self.llm_client, episode, new_nodes, previous_episodes
            )

            await asyncio.gather(
                *[edge.generate_embedding(embedder) for edge in extracted_edges]
            )

            existing_edges = await get_relevant_edges(extracted_edges, self.driver)
            logger.info(f"Existing edges: {[(e.name, e.uuid) for e in existing_edges]}")
            logger.info(
                f"Extracted edges: {[(e.name, e.uuid) for e in extracted_edges]}"
            )

            new_edges = await dedupe_extracted_edges(
                self.llm_client, extracted_edges, existing_edges
            )

            logger.info(f"Deduped edges: {[(e.name, e.uuid) for e in new_edges]}")

            entity_edges.extend(new_edges)
            episodic_edges.extend(
                build_episodic_edges(
                    # There may be an overlap between new_nodes and affected_nodes, so we're deduplicating them
                    nodes,
                    episode,
                    now,
                )
            )
            # Important to append the episode to the nodes at the end so that self referencing episodic edges are not built
            logger.info(f"Built episodic edges: {episodic_edges}")

            # invalidated_edges = await self.invalidate_edges(
            #     episode, new_nodes, new_edges, relevant_schema, previous_episodes
            # )

            # edges.extend(invalidated_edges)

            # Future optimization would be using batch operations to save nodes and edges
            await episode.save(self.driver)
            await asyncio.gather(*[node.save(self.driver) for node in nodes])
            await asyncio.gather(*[edge.save(self.driver) for edge in episodic_edges])
            await asyncio.gather(*[edge.save(self.driver) for edge in entity_edges])
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
            "CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)",
            "CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)",
            "CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.uuid)",
            "CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[r:MENTIONS]-() ON (r.uuid)",
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

        # Add the semantic indices
        await self.driver.execute_query(
            """
            CREATE FULLTEXT INDEX name_and_summary IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.summary]
            """
        )

        await self.driver.execute_query(
            """
            CREATE FULLTEXT INDEX name_and_fact IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON EACH [r.name, r.fact]
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

        await self.driver.execute_query(
            """
            CREATE VECTOR INDEX name_embedding IF NOT EXISTS
            FOR (n:Entity) ON (n.name_embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 1024,
             `vector.similarity_function`: 'cosine'
            }}
            """
        )

    async def search(self, query: str) -> list[tuple[EntityNode, list[EntityEdge]]]:
        text = query.replace("\n", " ")
        search_vector = (
            (
                await self.llm_client.client.embeddings.create(
                    input=[text], model="text-embedding-3-small"
                )
            )
            .data[0]
            .embedding[:EMBEDDING_DIM]
        )

        edges = await edge_similarity_search(search_vector, self.driver)
        nodes = await entity_fulltext_search(query, self.driver)

        node_ids = [node.uuid for node in nodes]

        for edge in edges:
            node_ids.append(edge.source_node_uuid)
            node_ids.append(edge.target_node_uuid)

        node_ids = list(dict.fromkeys(node_ids))

        context = await bfs(node_ids, self.driver)

        return context
