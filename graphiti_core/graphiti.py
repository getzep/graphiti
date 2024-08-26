"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
from datetime import datetime
from time import time
from typing import Callable

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search import Reranker, SearchConfig, SearchMethod, hybrid_search
from graphiti_core.search.search_utils import (
    get_relevant_edges,
    get_relevant_nodes,
)
from graphiti_core.utils import (
    build_episodic_edges,
    retrieve_episodes,
)
from graphiti_core.utils.bulk_utils import (
    RawEpisode,
    dedupe_edges_bulk,
    dedupe_nodes_bulk,
    extract_nodes_and_edges_bulk,
    resolve_edge_pointers,
    retrieve_previous_episodes_bulk,
)
from graphiti_core.utils.maintenance.edge_operations import (
    dedupe_extracted_edges,
    extract_edges,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    build_indices_and_constraints,
)
from graphiti_core.utils.maintenance.node_operations import dedupe_extracted_nodes, extract_nodes
from graphiti_core.utils.maintenance.temporal_operations import (
    extract_edge_dates,
    invalidate_edges,
    prepare_edges_for_invalidation,
)

logger = logging.getLogger(__name__)

load_dotenv()


class Graphiti:
    def __init__(self, uri: str, user: str, password: str, llm_client: LLMClient | None = None):
        """
        Initialize a Graphiti instance.

        This constructor sets up a connection to the Neo4j database and initializes
        the LLM client for natural language processing tasks.

        Parameters
        ----------
        uri : str
            The URI of the Neo4j database.
        user : str
            The username for authenticating with the Neo4j database.
        password : str
            The password for authenticating with the Neo4j database.
        llm_client : LLMClient | None, optional
            An instance of LLMClient for natural language processing tasks.
            If not provided, a default OpenAIClient will be initialized.

        Returns
        -------
        None

        Notes
        -----
        This method establishes a connection to the Neo4j database using the provided
        credentials. It also sets up the LLM client, either using the provided client
        or by creating a default OpenAIClient.

        The default database name is set to 'neo4j'. If a different database name
        is required, it should be specified in the URI or set separately after
        initialization.

        The OpenAI API key is expected to be set in the environment variables.
        Make sure to set the OPENAI_API_KEY environment variable before initializing
        Graphiti if you're using the default OpenAIClient.
        """
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.database = 'neo4j'
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = OpenAIClient()

    def close(self):
        """
        Close the connection to the Neo4j database.

        This method safely closes the driver connection to the Neo4j database.
        It should be called when the Graphiti instance is no longer needed or
        when the application is shutting down.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        It's important to close the driver connection to release system resources
        and ensure that all pending transactions are completed or rolled back.
        This method should be called as part of a cleanup process, potentially
        in a context manager or a shutdown hook.

        Example:
            graphiti = Graphiti(uri, user, password)
            try:
                # Use graphiti...
            finally:
                graphiti.close()
        self.driver.close()
        """

    async def build_indices_and_constraints(self):
        """
        Build indices and constraints in the Neo4j database.

        This method sets up the necessary indices and constraints in the Neo4j database
        to optimize query performance and ensure data integrity for the knowledge graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method should typically be called once during the initial setup of the
        knowledge graph or when updating the database schema. It uses the
        `build_indices_and_constraints` function from the
        `graphiti_core.utils.maintenance.graph_data_operations` module to perform
        the actual database operations.

        The specific indices and constraints created depend on the implementation
        of the `build_indices_and_constraints` function. Refer to that function's
        documentation for details on the exact database schema modifications.

        Caution: Running this method on a large existing database may take some time
        and could impact database performance during execution.
        """
        await build_indices_and_constraints(self.driver)

    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = EPISODE_WINDOW_LEN,
    ) -> list[EpisodicNode]:
        """
        Retrieve the last n episodic nodes from the graph.

        This method fetches a specified number of the most recent episodic nodes
        from the graph, relative to the given reference time.

        Parameters
        ----------
        reference_time : datetime
            The reference time to retrieve episodes before.
        last_n : int, optional
            The number of episodes to retrieve. Defaults to EPISODE_WINDOW_LEN.

        Returns
        -------
        list[EpisodicNode]
            A list of the most recent EpisodicNode objects.

        Notes
        -----
        The actual retrieval is performed by the `retrieve_episodes` function
        from the `graphiti_core.utils` module.
        """
        return await retrieve_episodes(self.driver, reference_time, last_n)

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        source: EpisodeType = EpisodeType.message,
        success_callback: Callable | None = None,
        error_callback: Callable | None = None,
    ):
        """
        Process an episode and update the graph.

        This method extracts information from the episode, creates nodes and edges,
        and updates the graph database accordingly.

        Parameters
        ----------
        name : str
            The name of the episode.
        episode_body : str
            The content of the episode.
        source_description : str
            A description of the episode's source.
        reference_time : datetime
            The reference time for the episode.
        source : EpisodeType, optional
            The type of the episode. Defaults to EpisodeType.message.
        success_callback : Callable | None, optional
            A callback function to be called upon successful processing.
        error_callback : Callable | None, optional
            A callback function to be called if an error occurs during processing.

        Returns
        -------
        None

        Notes
        -----
        This method performs several steps including node extraction, edge extraction,
        deduplication, and database updates. It also handles embedding generation
        and edge invalidation.

        It is recommended to run this method as a background process, such as in a queue.
        It's important that each episode is added sequentially and awaited before adding
        the next one. For web applications, consider using FastAPI's background tasks
        or a dedicated task queue like Celery for this purpose.

        Example using FastAPI background tasks:
            @app.post("/add_episode")
            async def add_episode_endpoint(episode_data: EpisodeData):
                background_tasks.add_task(graphiti.add_episode, **episode_data.dict())
                return {"message": "Episode processing started"}
        """
        try:
            start = time()

            nodes: list[EntityNode] = []
            entity_edges: list[EntityEdge] = []
            episodic_edges: list[EpisodicEdge] = []
            embedder = self.llm_client.get_embedder()
            now = datetime.now()

            previous_episodes = await self.retrieve_episodes(reference_time, last_n=3)
            episode = EpisodicNode(
                name=name,
                labels=[],
                source=source,
                content=episode_body,
                source_description=source_description,
                created_at=now,
                valid_at=reference_time,
            )

            extracted_nodes = await extract_nodes(self.llm_client, episode, previous_episodes)
            logger.info(f'Extracted nodes: {[(n.name, n.uuid) for n in extracted_nodes]}')

            # Calculate Embeddings

            await asyncio.gather(
                *[node.generate_name_embedding(embedder) for node in extracted_nodes]
            )
            existing_nodes = await get_relevant_nodes(extracted_nodes, self.driver)
            logger.info(f'Extracted nodes: {[(n.name, n.uuid) for n in extracted_nodes]}')
            touched_nodes, _, brand_new_nodes = await dedupe_extracted_nodes(
                self.llm_client, extracted_nodes, existing_nodes
            )
            logger.info(f'Adjusted touched nodes: {[(n.name, n.uuid) for n in touched_nodes]}')
            nodes.extend(touched_nodes)

            extracted_edges = await extract_edges(
                self.llm_client, episode, touched_nodes, previous_episodes
            )

            await asyncio.gather(*[edge.generate_embedding(embedder) for edge in extracted_edges])

            existing_edges = await get_relevant_edges(extracted_edges, self.driver)
            logger.info(f'Existing edges: {[(e.name, e.uuid) for e in existing_edges]}')
            logger.info(f'Extracted edges: {[(e.name, e.uuid) for e in extracted_edges]}')

            deduped_edges = await dedupe_extracted_edges(
                self.llm_client,
                extracted_edges,
                existing_edges,
            )

            edge_touched_node_uuids = [n.uuid for n in brand_new_nodes]
            for edge in deduped_edges:
                edge_touched_node_uuids.append(edge.source_node_uuid)
                edge_touched_node_uuids.append(edge.target_node_uuid)

            for edge in deduped_edges:
                valid_at, invalid_at, _ = await extract_edge_dates(
                    self.llm_client,
                    edge,
                    episode.valid_at,
                    episode,
                    previous_episodes,
                )
                edge.valid_at = valid_at
                edge.invalid_at = invalid_at
                if edge.invalid_at:
                    edge.expired_at = datetime.now()
            for edge in existing_edges:
                valid_at, invalid_at, _ = await extract_edge_dates(
                    self.llm_client,
                    edge,
                    episode.valid_at,
                    episode,
                    previous_episodes,
                )
                edge.valid_at = valid_at
                edge.invalid_at = invalid_at
                if edge.invalid_at:
                    edge.expired_at = datetime.now()
            (
                old_edges_with_nodes_pending_invalidation,
                new_edges_with_nodes,
            ) = prepare_edges_for_invalidation(
                existing_edges=existing_edges, new_edges=deduped_edges, nodes=nodes
            )

            invalidated_edges = await invalidate_edges(
                self.llm_client,
                old_edges_with_nodes_pending_invalidation,
                new_edges_with_nodes,
                episode,
                previous_episodes,
            )

            for edge in invalidated_edges:
                for existing_edge in existing_edges:
                    if existing_edge.uuid == edge.uuid:
                        existing_edge.expired_at = edge.expired_at
                for deduped_edge in deduped_edges:
                    if deduped_edge.uuid == edge.uuid:
                        deduped_edge.expired_at = edge.expired_at
                edge_touched_node_uuids.append(edge.source_node_uuid)
                edge_touched_node_uuids.append(edge.target_node_uuid)
            logger.info(f'Invalidated edges: {[(e.name, e.uuid) for e in invalidated_edges]}')

            edges_to_save = existing_edges + deduped_edges

            entity_edges.extend(edges_to_save)

            edge_touched_node_uuids = list(set(edge_touched_node_uuids))
            involved_nodes = [node for node in nodes if node.uuid in edge_touched_node_uuids]

            logger.info(f'Edge touched nodes: {[(n.name, n.uuid) for n in involved_nodes]}')

            logger.info(f'Deduped edges: {[(e.name, e.uuid) for e in deduped_edges]}')

            episodic_edges.extend(
                build_episodic_edges(
                    # There may be an overlap between new_nodes and affected_nodes, so we're deduplicating them
                    involved_nodes,
                    episode,
                    now,
                )
            )
            # Important to append the episode to the nodes at the end so that self referencing episodic edges are not built
            logger.info(f'Built episodic edges: {episodic_edges}')

            # Future optimization would be using batch operations to save nodes and edges
            await episode.save(self.driver)
            await asyncio.gather(*[node.save(self.driver) for node in nodes])
            await asyncio.gather(*[edge.save(self.driver) for edge in episodic_edges])
            await asyncio.gather(*[edge.save(self.driver) for edge in entity_edges])

            end = time()
            logger.info(f'Completed add_episode in {(end - start) * 1000} ms')
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

    async def add_episode_bulk(
        self,
        bulk_episodes: list[RawEpisode],
    ):
        """
        Process multiple episodes in bulk and update the graph.

        This method extracts information from multiple episodes, creates nodes and edges,
        and updates the graph database accordingly, all in a single batch operation.

        Parameters
        ----------
        bulk_episodes : list[RawEpisode]
            A list of RawEpisode objects to be processed and added to the graph.

        Returns
        -------
        None

        Notes
        -----
        This method performs several steps including:
        - Saving all episodes to the database
        - Retrieving previous episode context for each new episode
        - Extracting nodes and edges from all episodes
        - Generating embeddings for nodes and edges
        - Deduplicating nodes and edges
        - Saving nodes, episodic edges, and entity edges to the knowledge graph

        This bulk operation is designed for efficiency when processing multiple episodes
        at once. However, it's important to ensure that the bulk operation doesn't
        overwhelm system resources. Consider implementing rate limiting or chunking for
        very large batches of episodes.

        Important: This method does not perform edge invalidation or date extraction steps.
        If these operations are required, use the `add_episode` method instead for each
        individual episode.
        """
        try:
            start = time()
            embedder = self.llm_client.get_embedder()
            now = datetime.now()

            episodes = [
                EpisodicNode(
                    name=episode.name,
                    labels=[],
                    source=episode.source,
                    content=episode.content,
                    source_description=episode.source_description,
                    created_at=now,
                    valid_at=episode.reference_time,
                )
                for episode in bulk_episodes
            ]

            # Save all the episodes
            await asyncio.gather(*[episode.save(self.driver) for episode in episodes])

            # Get previous episode context for each episode
            episode_pairs = await retrieve_previous_episodes_bulk(self.driver, episodes)

            # Extract all nodes and edges
            (
                extracted_nodes,
                extracted_edges,
                episodic_edges,
            ) = await extract_nodes_and_edges_bulk(self.llm_client, episode_pairs)

            # Generate embeddings
            await asyncio.gather(
                *[node.generate_name_embedding(embedder) for node in extracted_nodes],
                *[edge.generate_embedding(embedder) for edge in extracted_edges],
            )

            # Dedupe extracted nodes
            nodes, uuid_map = await dedupe_nodes_bulk(self.driver, self.llm_client, extracted_nodes)

            # save nodes to KG
            await asyncio.gather(*[node.save(self.driver) for node in nodes])

            # re-map edge pointers so that they don't point to discard dupe nodes
            extracted_edges_with_resolved_pointers: list[EntityEdge] = resolve_edge_pointers(
                extracted_edges, uuid_map
            )
            episodic_edges_with_resolved_pointers: list[EpisodicEdge] = resolve_edge_pointers(
                episodic_edges, uuid_map
            )

            # save episodic edges to KG
            await asyncio.gather(
                *[edge.save(self.driver) for edge in episodic_edges_with_resolved_pointers]
            )

            # Dedupe extracted edges
            edges = await dedupe_edges_bulk(
                self.driver, self.llm_client, extracted_edges_with_resolved_pointers
            )
            logger.info(f'extracted edge length: {len(edges)}')

            # invalidate edges

            # save edges to KG
            await asyncio.gather(*[edge.save(self.driver) for edge in edges])

            end = time()
            logger.info(f'Completed add_episode_bulk in {(end - start) * 1000} ms')

        except Exception as e:
            raise e

    async def search(self, query: str, center_node_uuid: str | None = None, num_results=10):
        """
        Perform a hybrid search on the knowledge graph.

        This method executes a search query on the graph, combining vector and
        text-based search techniques to retrieve relevant facts.

        Parameters
        ----------
        query : str
            The search query string.
        center_node_uuid: str, optional
            Facts will be reranked based on proximity to this node
        num_results : int, optional
            The maximum number of results to return. Defaults to 10.

        Returns
        -------
        list
            A list of facts (strings) that are relevant to the search query.

        Notes
        -----
        This method uses a SearchConfig with num_episodes set to 0 and
        num_results set to the provided num_results parameter. It then calls
        the hybrid_search function to perform the actual search operation.

        The search is performed using the current date and time as the reference
        point for temporal relevance.
        """
        reranker = Reranker.rrf if center_node_uuid is None else Reranker.node_distance
        search_config = SearchConfig(
            num_episodes=0,
            num_edges=num_results,
            num_nodes=0,
            search_methods=[SearchMethod.bm25, SearchMethod.cosine_similarity],
            reranker=reranker,
        )
        edges = (
            await hybrid_search(
                self.driver,
                self.llm_client.get_embedder(),
                query,
                datetime.now(),
                search_config,
                center_node_uuid,
            )
        ).edges

        facts = [edge.fact for edge in edges]

        return facts

    async def _search(
        self,
        query: str,
        timestamp: datetime,
        config: SearchConfig,
        center_node_uuid: str | None = None,
    ):
        return await hybrid_search(
            self.driver, self.llm_client.get_embedder(), query, timestamp, config, center_node_uuid
        )
