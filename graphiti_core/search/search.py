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
from collections import defaultdict
from time import time

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import SearchRerankerError
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.search.search_config import (
    DEFAULT_SEARCH_LIMIT,
    CommunityReranker,
    CommunitySearchConfig,
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
    SearchResults,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    community_similarity_search,
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    episode_fulltext_search,
    episode_mentions_reranker,
    get_embeddings_for_communities,
    get_embeddings_for_edges,
    get_embeddings_for_nodes,
    maximal_marginal_relevance,
    node_bfs_search,
    node_distance_reranker,
    node_fulltext_search,
    node_similarity_search,
    rrf,
)

logger = logging.getLogger(__name__)


async def search(
    clients: GraphitiClients,
    query: str,
    group_ids: list[str] | None,
    config: SearchConfig,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    query_vector: list[float] | None = None,
) -> SearchResults:
    start = time()

    driver = clients.driver
    embedder = clients.embedder
    cross_encoder = clients.cross_encoder

    if query.strip() == '':
        return SearchResults(
            edges=[],
            nodes=[],
            episodes=[],
            communities=[],
        )
    query_vector = (
        query_vector
        if query_vector is not None
        else await embedder.create(input_data=[query.replace('\n', ' ')])
    )

    # if group_ids is empty, set it to None
    group_ids = group_ids if group_ids and group_ids != [''] else None
    edges, nodes, episodes, communities = await semaphore_gather(
        edge_search(
            driver,
            cross_encoder,
            query,
            query_vector,
            group_ids,
            config.edge_config,
            search_filter,
            center_node_uuid,
            bfs_origin_node_uuids,
            config.limit,
            config.reranker_min_score,
        ),
        node_search(
            driver,
            cross_encoder,
            query,
            query_vector,
            group_ids,
            config.node_config,
            search_filter,
            center_node_uuid,
            bfs_origin_node_uuids,
            config.limit,
            config.reranker_min_score,
        ),
        episode_search(
            driver,
            cross_encoder,
            query,
            query_vector,
            group_ids,
            config.episode_config,
            search_filter,
            config.limit,
            config.reranker_min_score,
        ),
        community_search(
            driver,
            cross_encoder,
            query,
            query_vector,
            group_ids,
            config.community_config,
            config.limit,
            config.reranker_min_score,
        ),
    )

    results = SearchResults(
        edges=edges,
        nodes=nodes,
        episodes=episodes,
        communities=communities,
    )

    latency = (time() - start) * 1000

    logger.debug(f'search returned context for query {query} in {latency} ms')

    return results


async def edge_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: EdgeSearchConfig | None,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
) -> list[EntityEdge]:
    if config is None:
        return []
    search_results: list[list[EntityEdge]] = list(
        await semaphore_gather(
            *[
                edge_fulltext_search(driver, query, search_filter, group_ids, 2 * limit),
                edge_similarity_search(
                    driver,
                    query_vector,
                    None,
                    None,
                    search_filter,
                    group_ids,
                    2 * limit,
                    config.sim_min_score,
                ),
                edge_bfs_search(
                    driver, bfs_origin_node_uuids, config.bfs_max_depth, search_filter, 2 * limit
                ),
            ]
        )
    )

    if EdgeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
        source_node_uuids = [edge.source_node_uuid for result in search_results for edge in result]
        search_results.append(
            await edge_bfs_search(
                driver, source_node_uuids, config.bfs_max_depth, search_filter, 2 * limit
            )
        )

    edge_uuid_map = {edge.uuid: edge for result in search_results for edge in result}
    if not edge_uuid_map:
        return []

    reranked_results: list[tuple[str, float]] = []
    if config.reranker == EdgeReranker.rrf or config.reranker == EdgeReranker.episode_mentions:
        search_result_uuids = [[edge.uuid for edge in result] for result in search_results]
        reranked_results = rrf(search_result_uuids, min_score=reranker_min_score)
    elif config.reranker == EdgeReranker.mmr:
        search_result_uuids_and_vectors = await get_embeddings_for_edges(
            driver, list(edge_uuid_map.values())
        )
        reranked_results = maximal_marginal_relevance(
            query_vector,
            search_result_uuids_and_vectors,
            config.mmr_lambda,
            reranker_min_score,
        )
    elif config.reranker == EdgeReranker.cross_encoder:
        # Limit the number of items sent to the cross-encoder to avoid high cost/latency
        initial_ranked_uuids = [
            uuid
            for uuid, _ in rrf(
                [[edge.uuid for edge in result] for result in search_results],
                min_score=reranker_min_score,
            )
        ]
        candidates = [edge_uuid_map[uuid] for uuid in initial_ranked_uuids[:limit]]
        fact_to_uuid_map = {edge.fact: edge.uuid for edge in candidates}
        if fact_to_uuid_map:
            reranked_facts = await cross_encoder.rank(query, list(fact_to_uuid_map.keys()))
            reranked_results = [
                (fact_to_uuid_map[fact], score)
                for fact, score in reranked_facts
                if score >= reranker_min_score
            ]
    elif config.reranker == EdgeReranker.node_distance:
        if center_node_uuid is None:
            raise SearchRerankerError('No center node provided for Node Distance reranker')

        sorted_result_uuids = [
            uuid
            for uuid, _ in rrf(
                [[edge.uuid for edge in result] for result in search_results],
                min_score=reranker_min_score,
            )
        ]
        sorted_results = [edge_uuid_map[uuid] for uuid in sorted_result_uuids]

        source_to_edge_uuid_map = defaultdict(list)
        for edge in sorted_results:
            source_to_edge_uuid_map[edge.source_node_uuid].append(edge.uuid)

        source_uuids = list(source_to_edge_uuid_map.keys())
        reranked_node_results = await node_distance_reranker(
            driver, source_uuids, center_node_uuid, min_score=reranker_min_score
        )

        temp_reranked_edges: list[EntityEdge] = []
        for node_uuid, score in reranked_node_results:
            for edge_uuid in source_to_edge_uuid_map[node_uuid]:
                edge = edge_uuid_map.get(edge_uuid)
                if edge:
                    edge.score = score
                    temp_reranked_edges.append(edge)
        return temp_reranked_edges[:limit]

    reranked_edges: list[EntityEdge] = []
    for uuid, score in reranked_results:
        edge = edge_uuid_map.get(uuid)
        if edge:
            edge.score = score
            reranked_edges.append(edge)

    if config.reranker == EdgeReranker.episode_mentions:
        for edge in reranked_edges:
            edge.score = float(len(edge.episodes))
        reranked_edges.sort(reverse=True, key=lambda edge: edge.score or 0.0)

    return reranked_edges[:limit]


async def node_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: NodeSearchConfig | None,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
) -> list[EntityNode]:
    if config is None:
        return []
    search_results: list[list[EntityNode]] = list(
        await semaphore_gather(
            *[
                node_fulltext_search(driver, query, search_filter, group_ids, 2 * limit),
                node_similarity_search(
                    driver, query_vector, search_filter, group_ids, 2 * limit, config.sim_min_score
                ),
                node_bfs_search(
                    driver, bfs_origin_node_uuids, search_filter, config.bfs_max_depth, 2 * limit
                ),
            ]
        )
    )

    if NodeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
        origin_node_uuids = [node.uuid for result in search_results for node in result]
        search_results.append(
            await node_bfs_search(
                driver, origin_node_uuids, search_filter, config.bfs_max_depth, 2 * limit
            )
        )

    node_uuid_map = {node.uuid: node for result in search_results for node in result}
    if not node_uuid_map:
        return []

    reranked_results: list[tuple[str, float]] = []
    if config.reranker == NodeReranker.rrf:
        search_result_uuids = [[node.uuid for node in result] for result in search_results]
        reranked_results = rrf(search_result_uuids, min_score=reranker_min_score)
    elif config.reranker == NodeReranker.mmr:
        search_result_uuids_and_vectors = await get_embeddings_for_nodes(
            driver, list(node_uuid_map.values())
        )
        reranked_results = maximal_marginal_relevance(
            query_vector,
            search_result_uuids_and_vectors,
            config.mmr_lambda,
            reranker_min_score,
        )
    elif config.reranker == NodeReranker.cross_encoder:
        initial_ranked_uuids = [
            uuid
            for uuid, _ in rrf(
                [[node.uuid for node in result] for result in search_results],
                min_score=reranker_min_score,
            )
        ]
        candidates = [node_uuid_map[uuid] for uuid in initial_ranked_uuids[:limit]]
        name_to_uuid_map = {node.name: node.uuid for node in candidates}
        if name_to_uuid_map:
            reranked_node_names = await cross_encoder.rank(query, list(name_to_uuid_map.keys()))
            reranked_results = [
                (name_to_uuid_map[name], score)
                for name, score in reranked_node_names
                if score >= reranker_min_score
            ]
    elif config.reranker == NodeReranker.episode_mentions:
        search_result_uuids = [[node.uuid for node in result] for result in search_results]
        reranked_results = await episode_mentions_reranker(
            driver, search_result_uuids, min_score=reranker_min_score
        )
    elif config.reranker == NodeReranker.node_distance:
        if center_node_uuid is None:
            raise SearchRerankerError('No center node provided for Node Distance reranker')
        search_result_uuids = [[node.uuid for node in result] for result in search_results]
        initial_ranked_uuids = [uuid for uuid, _ in rrf(search_result_uuids)]
        reranked_results = await node_distance_reranker(
            driver,
            initial_ranked_uuids,
            center_node_uuid,
            min_score=reranker_min_score,
        )

    reranked_nodes: list[EntityNode] = []
    for uuid, score in reranked_results:
        node = node_uuid_map.get(uuid)
        if node:
            node.score = score
            reranked_nodes.append(node)

    return reranked_nodes[:limit]


async def episode_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    _query_vector: list[float],
    group_ids: list[str] | None,
    config: EpisodeSearchConfig | None,
    search_filter: SearchFilters,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
) -> list[EpisodicNode]:
    if config is None:
        return []
    search_results: list[list[EpisodicNode]] = list(
        await semaphore_gather(
            *[
                episode_fulltext_search(driver, query, search_filter, group_ids, 2 * limit),
            ]
        )
    )

    episode_uuid_map = {episode.uuid: episode for result in search_results for episode in result}
    if not episode_uuid_map:
        return []

    reranked_results: list[tuple[str, float]] = []
    if config.reranker == EpisodeReranker.rrf:
        search_result_uuids = [[episode.uuid for episode in result] for result in search_results]
        reranked_results = rrf(search_result_uuids, min_score=reranker_min_score)
    elif config.reranker == EpisodeReranker.cross_encoder:
        initial_ranked_uuids = [
            uuid
            for uuid, _ in rrf(
                [[episode.uuid for episode in result] for result in search_results],
                min_score=reranker_min_score,
            )
        ]
        candidates = [episode_uuid_map[uuid] for uuid in initial_ranked_uuids[:limit]]
        content_to_uuid_map = {episode.content: episode.uuid for episode in candidates}
        if content_to_uuid_map:
            reranked_contents = await cross_encoder.rank(query, list(content_to_uuid_map.keys()))
            reranked_results = [
                (content_to_uuid_map[content], score)
                for content, score in reranked_contents
                if score >= reranker_min_score
            ]

    reranked_episodes: list[EpisodicNode] = []
    for uuid, score in reranked_results:
        episode = episode_uuid_map.get(uuid)
        if episode:
            episode.score = score
            reranked_episodes.append(episode)

    return reranked_episodes[:limit]


async def community_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: CommunitySearchConfig | None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
) -> list[CommunityNode]:
    if config is None:
        return []

    search_results: list[list[CommunityNode]] = list(
        await semaphore_gather(
            *[
                community_fulltext_search(driver, query, group_ids, 2 * limit),
                community_similarity_search(
                    driver, query_vector, group_ids, 2 * limit, config.sim_min_score
                ),
            ]
        )
    )

    community_uuid_map = {
        community.uuid: community for result in search_results for community in result
    }
    if not community_uuid_map:
        return []

    reranked_results: list[tuple[str, float]] = []
    if config.reranker == CommunityReranker.rrf:
        search_result_uuids = [[community.uuid for community in result] for result in search_results]
        reranked_results = rrf(search_result_uuids, min_score=reranker_min_score)
    elif config.reranker == CommunityReranker.mmr:
        search_result_uuids_and_vectors = await get_embeddings_for_communities(
            driver, list(community_uuid_map.values())
        )
        reranked_results = maximal_marginal_relevance(
            query_vector, search_result_uuids_and_vectors, config.mmr_lambda, reranker_min_score
        )
    elif config.reranker == CommunityReranker.cross_encoder:
        initial_ranked_uuids = [
            uuid
            for uuid, _ in rrf(
                [[community.uuid for community in result] for result in search_results],
                min_score=reranker_min_score,
            )
        ]
        candidates = [community_uuid_map[uuid] for uuid in initial_ranked_uuids[:limit]]
        name_to_uuid_map = {community.name: community.uuid for community in candidates}
        if name_to_uuid_map:
            reranked_nodes = await cross_encoder.rank(query, list(name_to_uuid_map.keys()))
            reranked_results = [
                (name_to_uuid_map[name], score)
                for name, score in reranked_nodes
                if score >= reranker_min_score
            ]

    reranked_communities: list[CommunityNode] = []
    for uuid, score in reranked_results:
        community = community_uuid_map.get(uuid)
        if community:
            community.score = score
            reranked_communities.append(community)

    return reranked_communities[:limit]
