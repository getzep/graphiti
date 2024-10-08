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

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search import SearchConfig, search
from graphiti_core.search.search_config import DEFAULT_SEARCH_LIMIT, SearchResults
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_utils import (
    RELEVANT_SCHEMA_LIMIT,
    get_communities_by_nodes,
    get_mentioned_nodes,
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
    extract_edge_dates_bulk,
    extract_nodes_and_edges_bulk,
    resolve_edge_pointers,
    retrieve_previous_episodes_bulk,
)
from graphiti_core.utils.maintenance.community_operations import (
    build_communities,
    remove_communities,
    update_community,
)
from graphiti_core.utils.maintenance.edge_operations import (
    extract_edges,
    resolve_extracted_edges,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    build_indices_and_constraints,
)
from graphiti_core.utils.maintenance.node_operations import (
    extract_nodes,
    resolve_extracted_nodes,
)

logger = logging.getLogger(__name__)

load_dotenv()


class AddEpisodeResults(BaseModel):
    episode: EpisodicNode
    nodes: list[EntityNode]
    edges: list[EntityEdge]


class Graphiti:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        store_raw_episode_content: bool = True,
    ):
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
        self.store_raw_episode_content = store_raw_episode_content
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = OpenAIClient()
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = OpenAIEmbedder()

    async def close(self):
        """
        Close the connection to the Neo4j database.

        This method safely closes the driver connection to the Neo4j database.
        It should be called when the Graphiti instance is no longer needed or
        when the application is shutting down.

        Parameters
        ----------
        self

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
        """
        await self.driver.close()

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        """
        Build indices and constraints in the Neo4j database.

        This method sets up the necessary indices and constraints in the Neo4j database
        to optimize query performance and ensure data integrity for the knowledge graph.

        Parameters
        ----------
        self
        delete_existing : bool, optional
            Whether to clear existing indices before creating new ones.


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
        await build_indices_and_constraints(self.driver, delete_existing)

    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = EPISODE_WINDOW_LEN,
        group_ids: list[str] | None = None,
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
        group_ids : list[str | None], optional
            The group ids to return data from.

        Returns
        -------
        list[EpisodicNode]
            A list of the most recent EpisodicNode objects.

        Notes
        -----
        The actual retrieval is performed by the `retrieve_episodes` function
        from the `graphiti_core.utils` module.
        """
        return await retrieve_episodes(self.driver, reference_time, last_n, group_ids)

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        source: EpisodeType = EpisodeType.message,
        group_id: str = '',
        uuid: str | None = None,
        update_communities: bool = False,
    ) -> AddEpisodeResults:
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
        group_id : str | None
            An id for the graph partition the episode is a part of.
        uuid : str | None
            Optional uuid of the episode.
        update_communities : bool
            Optional. Whether to update communities with new node information

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

            entity_edges: list[EntityEdge] = []
            now = datetime.now()

            previous_episodes = await self.retrieve_episodes(
                reference_time, last_n=3, group_ids=[group_id]
            )
            episode = EpisodicNode(
                name=name,
                group_id=group_id,
                labels=[],
                source=source,
                content=episode_body,
                source_description=source_description,
                created_at=now,
                valid_at=reference_time,
            )
            episode.uuid = uuid if uuid is not None else episode.uuid
            if not self.store_raw_episode_content:
                episode.content = ''

            # Extract entities as nodes

            extracted_nodes = await extract_nodes(self.llm_client, episode, previous_episodes)
            logger.info(f'Extracted nodes: {[(n.name, n.uuid) for n in extracted_nodes]}')

            # Calculate Embeddings

            await asyncio.gather(
                *[node.generate_name_embedding(self.embedder) for node in extracted_nodes]
            )

            # Resolve extracted nodes with nodes already in the graph and extract facts
            existing_nodes_lists: list[list[EntityNode]] = list(
                await asyncio.gather(
                    *[get_relevant_nodes([node], self.driver) for node in extracted_nodes]
                )
            )

            logger.info(f'Extracted nodes: {[(n.name, n.uuid) for n in extracted_nodes]}')

            (mentioned_nodes, uuid_map), extracted_edges = await asyncio.gather(
                resolve_extracted_nodes(self.llm_client, extracted_nodes, existing_nodes_lists),
                extract_edges(
                    self.llm_client, episode, extracted_nodes, previous_episodes, group_id
                ),
            )
            logger.info(f'Adjusted mentioned nodes: {[(n.name, n.uuid) for n in mentioned_nodes]}')
            nodes = mentioned_nodes

            extracted_edges_with_resolved_pointers = resolve_edge_pointers(
                extracted_edges, uuid_map
            )

            # calculate embeddings
            await asyncio.gather(
                *[
                    edge.generate_embedding(self.embedder)
                    for edge in extracted_edges_with_resolved_pointers
                ]
            )

            # Resolve extracted edges with related edges already in the graph
            related_edges_list: list[list[EntityEdge]] = list(
                await asyncio.gather(
                    *[
                        get_relevant_edges(
                            self.driver,
                            [edge],
                            edge.source_node_uuid,
                            edge.target_node_uuid,
                            RELEVANT_SCHEMA_LIMIT,
                        )
                        for edge in extracted_edges_with_resolved_pointers
                    ]
                )
            )
            logger.info(
                f'Related edges lists: {[(e.name, e.uuid) for edges_lst in related_edges_list for e in edges_lst]}'
            )
            logger.info(
                f'Extracted edges: {[(e.name, e.uuid) for e in extracted_edges_with_resolved_pointers]}'
            )

            existing_source_edges_list: list[list[EntityEdge]] = list(
                await asyncio.gather(
                    *[
                        get_relevant_edges(
                            self.driver,
                            [edge],
                            edge.source_node_uuid,
                            None,
                            RELEVANT_SCHEMA_LIMIT,
                        )
                        for edge in extracted_edges_with_resolved_pointers
                    ]
                )
            )

            existing_target_edges_list: list[list[EntityEdge]] = list(
                await asyncio.gather(
                    *[
                        get_relevant_edges(
                            self.driver,
                            [edge],
                            None,
                            edge.target_node_uuid,
                            RELEVANT_SCHEMA_LIMIT,
                        )
                        for edge in extracted_edges_with_resolved_pointers
                    ]
                )
            )

            existing_edges_list: list[list[EntityEdge]] = [
                source_lst + target_lst
                for source_lst, target_lst in zip(
                    existing_source_edges_list, existing_target_edges_list
                )
            ]

            resolved_edges, invalidated_edges = await resolve_extracted_edges(
                self.llm_client,
                extracted_edges_with_resolved_pointers,
                related_edges_list,
                existing_edges_list,
                episode,
                previous_episodes,
            )

            entity_edges.extend(resolved_edges + invalidated_edges)

            logger.info(f'Resolved edges: {[(e.name, e.uuid) for e in resolved_edges]}')

            episodic_edges: list[EpisodicEdge] = build_episodic_edges(mentioned_nodes, episode, now)

            logger.info(f'Built episodic edges: {episodic_edges}')

            episode.entity_edges = [edge.uuid for edge in entity_edges]

            # Future optimization would be using batch operations to save nodes and edges
            await episode.save(self.driver)
            await asyncio.gather(*[node.save(self.driver) for node in nodes])
            await asyncio.gather(*[edge.save(self.driver) for edge in episodic_edges])
            await asyncio.gather(*[edge.save(self.driver) for edge in entity_edges])

            # Update any communities
            if update_communities:
                await asyncio.gather(
                    *[
                        update_community(self.driver, self.llm_client, self.embedder, node)
                        for node in nodes
                    ]
                )
            end = time()
            logger.info(f'Completed add_episode in {(end - start) * 1000} ms')

            return AddEpisodeResults(episode=episode, nodes=nodes, edges=entity_edges)

        except Exception as e:
            raise e

    async def add_episode_bulk(self, bulk_episodes: list[RawEpisode], group_id: str = ''):
        """
        Process multiple episodes in bulk and update the graph.

        This method extracts information from multiple episodes, creates nodes and edges,
        and updates the graph database accordingly, all in a single batch operation.

        Parameters
        ----------
        bulk_episodes : list[RawEpisode]
            A list of RawEpisode objects to be processed and added to the graph.
        group_id : str | None
            An id for the graph partition the episode is a part of.

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
            now = datetime.now()

            episodes = [
                EpisodicNode(
                    name=episode.name,
                    labels=[],
                    source=episode.source,
                    content=episode.content,
                    source_description=episode.source_description,
                    group_id=group_id,
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
                *[node.generate_name_embedding(self.embedder) for node in extracted_nodes],
                *[edge.generate_embedding(self.embedder) for edge in extracted_edges],
            )

            # Dedupe extracted nodes, compress extracted edges
            (nodes, uuid_map), extracted_edges_timestamped = await asyncio.gather(
                dedupe_nodes_bulk(self.driver, self.llm_client, extracted_nodes),
                extract_edge_dates_bulk(self.llm_client, extracted_edges, episode_pairs),
            )

            # save nodes to KG
            await asyncio.gather(*[node.save(self.driver) for node in nodes])

            # re-map edge pointers so that they don't point to discard dupe nodes
            extracted_edges_with_resolved_pointers: list[EntityEdge] = resolve_edge_pointers(
                extracted_edges_timestamped, uuid_map
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

    async def build_communities(self, group_ids: list[str] | None = None) -> list[CommunityNode]:
        """
        Use a community clustering algorithm to find communities of nodes. Create community nodes summarising
        the content of these communities.
        ----------
        query : list[str] | None
            Optional. Create communities only for the listed group_ids. If blank the entire graph will be used.
        """
        # Clear existing communities
        await remove_communities(self.driver)

        community_nodes, community_edges = await build_communities(
            self.driver, self.llm_client, group_ids
        )

        await asyncio.gather(
            *[node.generate_name_embedding(self.embedder) for node in community_nodes]
        )

        await asyncio.gather(*[node.save(self.driver) for node in community_nodes])
        await asyncio.gather(*[edge.save(self.driver) for edge in community_edges])

        return community_nodes

    async def search(
        self,
        query: str,
        center_node_uuid: str | None = None,
        group_ids: list[str] | None = None,
        num_results=DEFAULT_SEARCH_LIMIT,
    ) -> list[EntityEdge]:
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
        group_ids : list[str | None] | None, optional
            The graph partitions to return data from.
        num_results : int, optional
            The maximum number of results to return. Defaults to 10.

        Returns
        -------
        list
            A list of EntityEdge objects that are relevant to the search query.

        Notes
        -----
        This method uses a SearchConfig with num_episodes set to 0 and
        num_results set to the provided num_results parameter. It then calls
        the hybrid_search function to perform the actual search operation.

        The search is performed using the current date and time as the reference
        point for temporal relevance.
        """
        search_config = (
            EDGE_HYBRID_SEARCH_RRF if center_node_uuid is None else EDGE_HYBRID_SEARCH_NODE_DISTANCE
        )
        search_config.limit = num_results

        edges = (
            await search(
                self.driver,
                self.embedder,
                query,
                group_ids,
                search_config,
                center_node_uuid,
            )
        ).edges

        return edges

    async def _search(
        self,
        query: str,
        config: SearchConfig,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
    ) -> SearchResults:
        return await search(self.driver, self.embedder, query, group_ids, config, center_node_uuid)

    async def get_nodes_by_query(
        self,
        query: str,
        center_node_uuid: str | None = None,
        group_ids: list[str] | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[EntityNode]:
        """
        Retrieve nodes from the graph database based on a text query.

        This method performs a hybrid search using both text-based and
        embedding-based approaches to find relevant nodes.

        Parameters
        ----------
        query : str
            The text query to search for in the graph
        center_node_uuid: str, optional
            Facts will be reranked based on proximity to this node.
        group_ids : list[str | None] | None, optional
            The graph partitions to return data from.
        limit : int | None, optional
            The maximum number of results to return per search method.
            If None, a default limit will be applied.

        Returns
        -------
        list[EntityNode]
            A list of EntityNode objects that match the search criteria.

        Notes
        -----
        This method uses the following steps:
        1. Generates an embedding for the input query using the LLM client's embedder.
        2. Calls the hybrid_node_search function with both the text query and its embedding.
        3. The hybrid search combines fulltext search and vector similarity search
           to find the most relevant nodes.

        The method leverages the LLM client's embedding capabilities to enhance
        the search with semantic similarity matching. The 'limit' parameter is applied
        to each individual search method before results are combined and deduplicated.
        If not specified, a default limit (defined in the search functions) will be used.
        """
        search_config = (
            NODE_HYBRID_SEARCH_RRF if center_node_uuid is None else NODE_HYBRID_SEARCH_NODE_DISTANCE
        )
        search_config.limit = limit

        nodes = (
            await search(
                self.driver, self.embedder, query, group_ids, search_config, center_node_uuid
            )
        ).nodes
        return nodes

    async def get_episode_mentions(self, episode_uuids: list[str]) -> SearchResults:
        episodes = await EpisodicNode.get_by_uuids(self.driver, episode_uuids)

        edges_list = await asyncio.gather(
            *[EntityEdge.get_by_uuids(self.driver, episode.entity_edges) for episode in episodes]
        )

        edges: list[EntityEdge] = [edge for lst in edges_list for edge in lst]

        nodes = await get_mentioned_nodes(self.driver, episodes)

        communities = await get_communities_by_nodes(self.driver, nodes)

        return SearchResults(edges=edges, nodes=nodes, communities=communities)
