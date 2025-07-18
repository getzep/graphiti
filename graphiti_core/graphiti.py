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

import logging
from datetime import datetime
from time import time

from dotenv import load_dotenv
from pydantic import BaseModel
from typing_extensions import LiteralString

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import (
    get_default_group_id,
    semaphore_gather,
    validate_excluded_entity_types,
    validate_group_id,
)
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search import SearchConfig, search
from graphiti_core.search.search_config import DEFAULT_SEARCH_LIMIT, SearchResults
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    RELEVANT_SCHEMA_LIMIT,
    get_edge_invalidation_candidates,
    get_mentioned_nodes,
    get_relevant_edges,
)
from graphiti_core.telemetry import capture_event
from graphiti_core.utils.bulk_utils import (
    RawEpisode,
    add_nodes_and_edges_bulk,
    dedupe_edges_bulk,
    dedupe_nodes_bulk,
    extract_nodes_and_edges_bulk,
    resolve_edge_pointers,
    retrieve_previous_episodes_bulk,
)
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.community_operations import (
    build_communities,
    remove_communities,
    update_community,
)
from graphiti_core.utils.maintenance.edge_operations import (
    build_duplicate_of_edges,
    build_episodic_edges,
    extract_edges,
    resolve_extracted_edge,
    resolve_extracted_edges,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    build_indices_and_constraints,
    retrieve_episodes,
)
from graphiti_core.utils.maintenance.node_operations import (
    extract_attributes_from_nodes,
    extract_nodes,
    resolve_extracted_nodes,
)
from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types

logger = logging.getLogger(__name__)

load_dotenv()


class AddEpisodeResults(BaseModel):
    episode: EpisodicNode
    nodes: list[EntityNode]
    edges: list[EntityEdge]


class Graphiti:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
        store_raw_episode_content: bool = True,
        graph_driver: GraphDriver | None = None,
        max_coroutines: int | None = None,
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
        embedder : EmbedderClient | None, optional
            An instance of EmbedderClient for embedding tasks.
            If not provided, a default OpenAIEmbedder will be initialized.
        cross_encoder : CrossEncoderClient | None, optional
            An instance of CrossEncoderClient for reranking tasks.
            If not provided, a default OpenAIRerankerClient will be initialized.
        store_raw_episode_content : bool, optional
            Whether to store the raw content of episodes. Defaults to True.
        graph_driver : GraphDriver | None, optional
            An instance of GraphDriver for database operations.
            If not provided, a default Neo4jDriver will be initialized.
        max_coroutines : int | None, optional
            The maximum number of concurrent operations allowed. Overrides SEMAPHORE_LIMIT set in the environment.
            If not set, the Graphiti default is used.

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

        if graph_driver:
            self.driver = graph_driver
        else:
            if uri is None:
                raise ValueError('uri must be provided when graph_driver is None')
            self.driver = Neo4jDriver(uri, user, password)

        self.store_raw_episode_content = store_raw_episode_content
        self.max_coroutines = max_coroutines
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = OpenAIClient()
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = OpenAIEmbedder()
        if cross_encoder:
            self.cross_encoder = cross_encoder
        else:
            self.cross_encoder = OpenAIRerankerClient()

        self.clients = GraphitiClients(
            driver=self.driver,
            llm_client=self.llm_client,
            embedder=self.embedder,
            cross_encoder=self.cross_encoder,
        )

        # Capture telemetry event
        self._capture_initialization_telemetry()

    def _capture_initialization_telemetry(self):
        """Capture telemetry event for Graphiti initialization."""
        try:
            # Detect provider types from class names
            llm_provider = self._get_provider_type(self.llm_client)
            embedder_provider = self._get_provider_type(self.embedder)
            reranker_provider = self._get_provider_type(self.cross_encoder)
            database_provider = self._get_provider_type(self.driver)

            properties = {
                'llm_provider': llm_provider,
                'embedder_provider': embedder_provider,
                'reranker_provider': reranker_provider,
                'database_provider': database_provider,
            }

            capture_event('graphiti_initialized', properties)
        except Exception:
            # Silently handle telemetry errors
            pass

    def _get_provider_type(self, client) -> str:
        """Get provider type from client class name."""
        if client is None:
            return 'none'

        class_name = client.__class__.__name__.lower()

        # LLM providers
        if 'openai' in class_name:
            return 'openai'
        elif 'azure' in class_name:
            return 'azure'
        elif 'anthropic' in class_name:
            return 'anthropic'
        elif 'crossencoder' in class_name:
            return 'crossencoder'
        elif 'gemini' in class_name:
            return 'gemini'
        elif 'groq' in class_name:
            return 'groq'
        # Database providers
        elif 'neo4j' in class_name:
            return 'neo4j'
        elif 'falkor' in class_name:
            return 'falkordb'
        # Embedder providers
        elif 'voyage' in class_name:
            return 'voyage'
        else:
            return 'unknown'

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
        source: EpisodeType | None = None,
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
        return await retrieve_episodes(self.driver, reference_time, last_n, group_ids, source)

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        source: EpisodeType = EpisodeType.message,
        group_id: str | None = None,
        uuid: str | None = None,
        update_communities: bool = False,
        entity_types: dict[str, BaseModel] | None = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        edge_types: dict[str, BaseModel] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
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
        entity_types : dict[str, BaseModel] | None
            Optional. Dictionary mapping entity type names to their Pydantic model definitions.
        excluded_entity_types : list[str] | None
            Optional. List of entity type names to exclude from the graph. Entities classified
            into these types will not be added to the graph. Can include 'Entity' to exclude
            the default entity type.
        previous_episode_uuids : list[str] | None
            Optional.  list of episode uuids to use as the previous episodes. If this is not provided,
            the most recent episodes by created_at date will be used.

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
            now = utc_now()

            # if group_id is None, use the default group id by the provider
            group_id = group_id or get_default_group_id(self.driver.provider)
            validate_entity_types(entity_types)

            validate_excluded_entity_types(excluded_entity_types, entity_types)
            validate_group_id(group_id)

            previous_episodes = (
                await self.retrieve_episodes(
                    reference_time,
                    last_n=RELEVANT_SCHEMA_LIMIT,
                    group_ids=[group_id],
                    source=source,
                )
                if previous_episode_uuids is None
                else await EpisodicNode.get_by_uuids(self.driver, previous_episode_uuids)
            )

            episode = (
                await EpisodicNode.get_by_uuid(self.driver, uuid)
                if uuid is not None
                else EpisodicNode(
                    name=name,
                    group_id=group_id,
                    labels=[],
                    source=source,
                    content=episode_body,
                    source_description=source_description,
                    created_at=now,
                    valid_at=reference_time,
                )
            )

            # Create default edge type map
            edge_type_map_default = (
                {('Entity', 'Entity'): list(edge_types.keys())}
                if edge_types is not None
                else {('Entity', 'Entity'): []}
            )

            # Extract entities as nodes

            extracted_nodes = await extract_nodes(
                self.clients, episode, previous_episodes, entity_types, excluded_entity_types
            )

            # Extract edges and resolve nodes
            (nodes, uuid_map, node_duplicates), extracted_edges = await semaphore_gather(
                resolve_extracted_nodes(
                    self.clients,
                    extracted_nodes,
                    episode,
                    previous_episodes,
                    entity_types,
                ),
                extract_edges(
                    self.clients,
                    episode,
                    extracted_nodes,
                    previous_episodes,
                    edge_type_map or edge_type_map_default,
                    group_id,
                    edge_types,
                ),
                max_coroutines=self.max_coroutines,
            )

            edges = resolve_edge_pointers(extracted_edges, uuid_map)

            (resolved_edges, invalidated_edges), hydrated_nodes = await semaphore_gather(
                resolve_extracted_edges(
                    self.clients,
                    edges,
                    episode,
                    nodes,
                    edge_types or {},
                    edge_type_map or edge_type_map_default,
                ),
                extract_attributes_from_nodes(
                    self.clients, nodes, episode, previous_episodes, entity_types
                ),
                max_coroutines=self.max_coroutines,
            )

            duplicate_of_edges = build_duplicate_of_edges(episode, now, node_duplicates)

            entity_edges = resolved_edges + invalidated_edges + duplicate_of_edges

            episodic_edges = build_episodic_edges(nodes, episode.uuid, now)

            episode.entity_edges = [edge.uuid for edge in entity_edges]

            if not self.store_raw_episode_content:
                episode.content = ''

            await add_nodes_and_edges_bulk(
                self.driver, [episode], episodic_edges, hydrated_nodes, entity_edges, self.embedder
            )

            # Update any communities
            if update_communities:
                await semaphore_gather(
                    *[
                        update_community(self.driver, self.llm_client, self.embedder, node)
                        for node in nodes
                    ],
                    max_coroutines=self.max_coroutines,
                )
            end = time()
            logger.info(f'Completed add_episode in {(end - start) * 1000} ms')

            return AddEpisodeResults(episode=episode, nodes=nodes, edges=entity_edges)

        except Exception as e:
            raise e

    ##### EXPERIMENTAL #####
    async def add_episode_bulk(
        self,
        bulk_episodes: list[RawEpisode],
        group_id: str | None = None,
        entity_types: dict[str, BaseModel] | None = None,
        excluded_entity_types: list[str] | None = None,
        edge_types: dict[str, BaseModel] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
    ):
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
            now = utc_now()

            # if group_id is None, use the default group id by the provider
            group_id = group_id or get_default_group_id(self.driver.provider)
            validate_group_id(group_id)

            # Create default edge type map
            edge_type_map_default = (
                {('Entity', 'Entity'): list(edge_types.keys())}
                if edge_types is not None
                else {('Entity', 'Entity'): []}
            )

            episodes = [
                await EpisodicNode.get_by_uuid(self.driver, episode.uuid)
                if episode.uuid is not None
                else EpisodicNode(
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

            episodes_by_uuid: dict[str, EpisodicNode] = {
                episode.uuid: episode for episode in episodes
            }

            # Save all episodes
            await add_nodes_and_edges_bulk(
                driver=self.driver,
                episodic_nodes=episodes,
                episodic_edges=[],
                entity_nodes=[],
                entity_edges=[],
                embedder=self.embedder,
            )

            # Get previous episode context for each episode
            episode_context = await retrieve_previous_episodes_bulk(self.driver, episodes)

            # Extract all nodes and edges for each episode
            extracted_nodes_bulk, extracted_edges_bulk = await extract_nodes_and_edges_bulk(
                self.clients,
                episode_context,
                edge_type_map=edge_type_map or edge_type_map_default,
                edge_types=edge_types,
                entity_types=entity_types,
                excluded_entity_types=excluded_entity_types,
            )

            # Dedupe extracted nodes in memory
            nodes_by_episode, uuid_map = await dedupe_nodes_bulk(
                self.clients, extracted_nodes_bulk, episode_context, entity_types
            )

            # Create Episodic Edges
            episodic_edges: list[EpisodicEdge] = []
            for episode_uuid, nodes in nodes_by_episode.items():
                episodic_edges.extend(build_episodic_edges(nodes, episode_uuid, now))

            # re-map edge pointers so that they don't point to discard dupe nodes
            extracted_edges_bulk_updated: list[list[EntityEdge]] = [
                resolve_edge_pointers(edges, uuid_map) for edges in extracted_edges_bulk
            ]

            # Dedupe extracted edges in memory
            edges_by_episode = await dedupe_edges_bulk(
                self.clients,
                extracted_edges_bulk_updated,
                episode_context,
                [],
                edge_types or {},
                edge_type_map or edge_type_map_default,
            )

            # Extract node attributes
            nodes_by_uuid: dict[str, EntityNode] = {
                node.uuid: node for nodes in nodes_by_episode.values() for node in nodes
            }

            extract_attributes_params: list[tuple[EntityNode, list[EpisodicNode]]] = []
            for node in nodes_by_uuid.values():
                episode_uuids: list[str] = []
                for episode_uuid, mentioned_nodes in nodes_by_episode.items():
                    for mentioned_node in mentioned_nodes:
                        if node.uuid == mentioned_node.uuid:
                            episode_uuids.append(episode_uuid)
                            break

                episode_mentions: list[EpisodicNode] = [
                    episodes_by_uuid[episode_uuid] for episode_uuid in episode_uuids
                ]
                episode_mentions.sort(key=lambda x: x.valid_at, reverse=True)

                extract_attributes_params.append((node, episode_mentions))

            new_hydrated_nodes: list[list[EntityNode]] = await semaphore_gather(
                *[
                    extract_attributes_from_nodes(
                        self.clients,
                        [params[0]],
                        params[1][0],
                        params[1][0:],
                        entity_types,
                    )
                    for params in extract_attributes_params
                ]
            )

            hydrated_nodes = [node for nodes in new_hydrated_nodes for node in nodes]

            # Update nodes_by_uuid map with the hydrated nodes
            for hydrated_node in hydrated_nodes:
                nodes_by_uuid[hydrated_node.uuid] = hydrated_node

            # Resolve nodes and edges against the existing graph
            nodes_by_episode_unique: dict[str, list[EntityNode]] = {}
            nodes_uuid_set: set[str] = set()
            for episode, _ in episode_context:
                nodes_by_episode_unique[episode.uuid] = []
                nodes = [nodes_by_uuid[node.uuid] for node in nodes_by_episode[episode.uuid]]
                for node in nodes:
                    if node.uuid not in nodes_uuid_set:
                        nodes_by_episode_unique[episode.uuid].append(node)
                        nodes_uuid_set.add(node.uuid)

            node_results = await semaphore_gather(
                *[
                    resolve_extracted_nodes(
                        self.clients,
                        nodes_by_episode_unique[episode.uuid],
                        episode,
                        previous_episodes,
                        entity_types,
                    )
                    for episode, previous_episodes in episode_context
                ]
            )

            resolved_nodes: list[EntityNode] = []
            uuid_map: dict[str, str] = {}
            node_duplicates: list[tuple[EntityNode, EntityNode]] = []
            for result in node_results:
                resolved_nodes.extend(result[0])
                uuid_map.update(result[1])
                node_duplicates.extend(result[2])

            # Update nodes_by_uuid map with the resolved nodes
            for resolved_node in resolved_nodes:
                nodes_by_uuid[resolved_node.uuid] = resolved_node

            # update nodes_by_episode_unique mapping
            for episode_uuid, nodes in nodes_by_episode_unique.items():
                updated_nodes: list[EntityNode] = []
                for node in nodes:
                    updated_node_uuid = uuid_map.get(node.uuid, node.uuid)
                    updated_node = nodes_by_uuid[updated_node_uuid]
                    updated_nodes.append(updated_node)

                nodes_by_episode_unique[episode_uuid] = updated_nodes

            hydrated_nodes_results: list[list[EntityNode]] = await semaphore_gather(
                *[
                    extract_attributes_from_nodes(
                        self.clients,
                        nodes_by_episode_unique[episode.uuid],
                        episode,
                        previous_episodes,
                        entity_types,
                    )
                    for episode, previous_episodes in episode_context
                ]
            )

            final_hydrated_nodes = [node for nodes in hydrated_nodes_results for node in nodes]

            edges_by_episode_unique: dict[str, list[EntityEdge]] = {}
            edges_uuid_set: set[str] = set()
            for episode_uuid, edges in edges_by_episode.items():
                edges_with_updated_pointers = resolve_edge_pointers(edges, uuid_map)
                edges_by_episode_unique[episode_uuid] = []

                for edge in edges_with_updated_pointers:
                    if edge.uuid not in edges_uuid_set:
                        edges_by_episode_unique[episode_uuid].append(edge)
                        edges_uuid_set.add(edge.uuid)

            edge_results = await semaphore_gather(
                *[
                    resolve_extracted_edges(
                        self.clients,
                        edges_by_episode_unique[episode.uuid],
                        episode,
                        hydrated_nodes,
                        edge_types or {},
                        edge_type_map or edge_type_map_default,
                    )
                    for episode in episodes
                ]
            )

            resolved_edges: list[EntityEdge] = []
            invalidated_edges: list[EntityEdge] = []
            for result in edge_results:
                resolved_edges.extend(result[0])
                invalidated_edges.extend(result[1])

            # Resolved pointers for episodic edges
            resolved_episodic_edges = resolve_edge_pointers(episodic_edges, uuid_map)

            # save data to KG
            await add_nodes_and_edges_bulk(
                self.driver,
                episodes,
                resolved_episodic_edges,
                final_hydrated_nodes,
                resolved_edges + invalidated_edges,
                self.embedder,
            )

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

        await semaphore_gather(
            *[node.generate_name_embedding(self.embedder) for node in community_nodes],
            max_coroutines=self.max_coroutines,
        )

        await semaphore_gather(
            *[node.save(self.driver) for node in community_nodes],
            max_coroutines=self.max_coroutines,
        )
        await semaphore_gather(
            *[edge.save(self.driver) for edge in community_edges],
            max_coroutines=self.max_coroutines,
        )

        return community_nodes

    async def search(
        self,
        query: str,
        center_node_uuid: str | None = None,
        group_ids: list[str] | None = None,
        num_results=DEFAULT_SEARCH_LIMIT,
        search_filter: SearchFilters | None = None,
    ) -> list[EntityEdge]:
        """
        Perform a hybrid search on the knowledge graph.

        This method executes a search query on the graph, combining vector and
        text-based search techniques to retrieve relevant facts, returning the edges as a string.

        This is our basic out-of-the-box search, for more robust results we recommend using our more advanced
        search method graphiti.search_().

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
        num_results set to the provided num_results parameter.

        The search is performed using the current date and time as the reference
        point for temporal relevance.
        """
        search_config = (
            EDGE_HYBRID_SEARCH_RRF if center_node_uuid is None else EDGE_HYBRID_SEARCH_NODE_DISTANCE
        )
        search_config.limit = num_results

        edges = (
            await search(
                self.clients,
                query,
                group_ids,
                search_config,
                search_filter if search_filter is not None else SearchFilters(),
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
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
    ) -> SearchResults:
        """DEPRECATED"""
        return await self.search_(
            query, config, group_ids, center_node_uuid, bfs_origin_node_uuids, search_filter
        )

    async def search_(
        self,
        query: str,
        config: SearchConfig = COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
    ) -> SearchResults:
        """search_ (replaces _search) is our advanced search method that returns Graph objects (nodes and edges) rather
        than a list of facts. This endpoint allows the end user to utilize more advanced features such as filters and
        different search and reranker methodologies across different layers in the graph.

        For different config recipes refer to search/search_config_recipes.
        """

        return await search(
            self.clients,
            query,
            group_ids,
            config,
            search_filter if search_filter is not None else SearchFilters(),
            center_node_uuid,
            bfs_origin_node_uuids,
        )

    async def get_nodes_and_edges_by_episode(self, episode_uuids: list[str]) -> SearchResults:
        episodes = await EpisodicNode.get_by_uuids(self.driver, episode_uuids)

        edges_list = await semaphore_gather(
            *[EntityEdge.get_by_uuids(self.driver, episode.entity_edges) for episode in episodes],
            max_coroutines=self.max_coroutines,
        )

        edges: list[EntityEdge] = [edge for lst in edges_list for edge in lst]

        nodes = await get_mentioned_nodes(self.driver, episodes)

        return SearchResults(edges=edges, nodes=nodes, episodes=[], communities=[])

    async def add_triplet(self, source_node: EntityNode, edge: EntityEdge, target_node: EntityNode):
        if source_node.name_embedding is None:
            await source_node.generate_name_embedding(self.embedder)
        if target_node.name_embedding is None:
            await target_node.generate_name_embedding(self.embedder)
        if edge.fact_embedding is None:
            await edge.generate_embedding(self.embedder)

        resolved_nodes, uuid_map, _ = await resolve_extracted_nodes(
            self.clients,
            [source_node, target_node],
        )

        updated_edge = resolve_edge_pointers([edge], uuid_map)[0]

        related_edges = (await get_relevant_edges(self.driver, [updated_edge], SearchFilters()))[0]
        existing_edges = (
            await get_edge_invalidation_candidates(self.driver, [updated_edge], SearchFilters())
        )[0]

        resolved_edge, invalidated_edges, _ = await resolve_extracted_edge(
            self.llm_client,
            updated_edge,
            related_edges,
            existing_edges,
            EpisodicNode(
                name='',
                source=EpisodeType.text,
                source_description='',
                content='',
                valid_at=edge.valid_at or utc_now(),
                entity_edges=[],
                group_id=edge.group_id,
            ),
        )

        await add_nodes_and_edges_bulk(
            self.driver, [], [], resolved_nodes, [resolved_edge] + invalidated_edges, self.embedder
        )

    async def remove_episode(self, episode_uuid: str):
        # Find the episode to be deleted
        episode = await EpisodicNode.get_by_uuid(self.driver, episode_uuid)

        # Find edges mentioned by the episode
        edges = await EntityEdge.get_by_uuids(self.driver, episode.entity_edges)

        # We should only delete edges created by the episode
        edges_to_delete: list[EntityEdge] = []
        for edge in edges:
            if edge.episodes and edge.episodes[0] == episode.uuid:
                edges_to_delete.append(edge)

        # Find nodes mentioned by the episode
        nodes = await get_mentioned_nodes(self.driver, [episode])
        # We should delete all nodes that are only mentioned in the deleted episode
        nodes_to_delete: list[EntityNode] = []
        for node in nodes:
            query: LiteralString = 'MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $uuid}) RETURN count(*) AS episode_count'
            records, _, _ = await self.driver.execute_query(query, uuid=node.uuid, routing_='r')

            for record in records:
                if record['episode_count'] == 1:
                    nodes_to_delete.append(node)

        await semaphore_gather(
            *[node.delete(self.driver) for node in nodes_to_delete],
            max_coroutines=self.max_coroutines,
        )
        await semaphore_gather(
            *[edge.delete(self.driver) for edge in edges_to_delete],
            max_coroutines=self.max_coroutines,
        )
        await episode.delete(self.driver)
