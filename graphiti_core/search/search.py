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
from collections import defaultdict
from time import time

from neo4j import AsyncDriver

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import SearchRerankerError
from graphiti_core.nodes import CommunityNode, EntityNode
from graphiti_core.search.search_config import (
    DEFAULT_SEARCH_LIMIT,
    CommunityReranker,
    CommunitySearchConfig,
    EdgeReranker,
    EdgeSearchConfig,
    NodeReranker,
    NodeSearchConfig,
    SearchConfig,
    SearchResults,
)
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    community_similarity_search,
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    episode_mentions_reranker,
    maximal_marginal_relevance,
    node_distance_reranker,
    node_fulltext_search,
    node_similarity_search,
    rrf,
)

logger = logging.getLogger(__name__)


async def search(
    driver: AsyncDriver,
    embedder: EmbedderClient,
    cross_encoder: CrossEncoderClient,
    query: str,
    group_ids: list[str] | None,
    config: SearchConfig,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
) -> SearchResults:
    start = time()
    query_vector = await embedder.create(input=[query.replace('\n', ' ')])

    # if group_ids is empty, set it to None
    group_ids = group_ids if group_ids else None
    edges, nodes, communities = await asyncio.gather(
        edge_search(
            driver,
            cross_encoder,
            query,
            query_vector,
            group_ids,
            config.edge_config,
            center_node_uuid,
            bfs_origin_node_uuids,
            config.limit,
        ),
        node_search(
            driver,
            cross_encoder,
            query,
            query_vector,
            group_ids,
            config.node_config,
            center_node_uuid,
            bfs_origin_node_uuids,
            config.limit,
        ),
        community_search(
            driver,
            cross_encoder,
            query,
            query_vector,
            group_ids,
            config.community_config,
            bfs_origin_node_uuids,
            config.limit,
        ),
    )

    results = SearchResults(
        edges=edges,
        nodes=nodes,
        communities=communities,
    )

    latency = (time() - start) * 1000

    logger.debug(f'search returned context for query {query} in {latency} ms')

    return results


async def edge_search(
    driver: AsyncDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: EdgeSearchConfig | None,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
) -> list[EntityEdge]:
    if config is None:
        return []

    search_results: list[list[EntityEdge]] = list(
        await asyncio.gather(
            *[
                edge_fulltext_search(driver, query, None, None, group_ids, 2 * limit),
                edge_similarity_search(
                    driver, query_vector, None, None, group_ids, 2 * limit, config.sim_min_score
                ),
                edge_bfs_search(driver, bfs_origin_node_uuids, config.bfs_max_depth),
            ]
        )
    )

    edge_uuid_map = {edge.uuid: edge for result in search_results for edge in result}

    reranked_uuids: list[str] = []
    if config.reranker == EdgeReranker.rrf or config.reranker == EdgeReranker.episode_mentions:
        search_result_uuids = [[edge.uuid for edge in result] for result in search_results]

        reranked_uuids = rrf(search_result_uuids)
    elif config.reranker == EdgeReranker.mmr:
        search_result_uuids_and_vectors = [
            (edge.uuid, edge.fact_embedding if edge.fact_embedding is not None else [0.0] * 1024)
            for result in search_results
            for edge in result
        ]
        reranked_uuids = maximal_marginal_relevance(
            query_vector, search_result_uuids_and_vectors, config.mmr_lambda
        )
    elif config.reranker == EdgeReranker.cross_encoder:
        fact_to_uuid_map = {edge.fact: edge.uuid for result in search_results for edge in result}
        reranked_facts = await cross_encoder.rank(query, list(fact_to_uuid_map.keys()))
        reranked_uuids = [fact_to_uuid_map[fact] for fact, _ in reranked_facts]
    elif config.reranker == EdgeReranker.node_distance:
        if center_node_uuid is None:
            raise SearchRerankerError('No center node provided for Node Distance reranker')

        # use rrf as a preliminary sort
        sorted_result_uuids = rrf([[edge.uuid for edge in result] for result in search_results])
        sorted_results = [edge_uuid_map[uuid] for uuid in sorted_result_uuids]

        # node distance reranking
        source_to_edge_uuid_map = defaultdict(list)
        for edge in sorted_results:
            source_to_edge_uuid_map[edge.source_node_uuid].append(edge.uuid)

        source_uuids = [source_node_uuid for source_node_uuid in source_to_edge_uuid_map]

        reranked_node_uuids = await node_distance_reranker(driver, source_uuids, center_node_uuid)

        for node_uuid in reranked_node_uuids:
            reranked_uuids.extend(source_to_edge_uuid_map[node_uuid])

    reranked_edges = [edge_uuid_map[uuid] for uuid in reranked_uuids]

    if config.reranker == EdgeReranker.episode_mentions:
        reranked_edges.sort(reverse=True, key=lambda edge: len(edge.episodes))

    return reranked_edges[:limit]


async def node_search(
    driver: AsyncDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: NodeSearchConfig | None,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
) -> list[EntityNode]:
    if config is None:
        return []

    search_results: list[list[EntityNode]] = list(
        await asyncio.gather(
            *[
                node_fulltext_search(driver, query, group_ids, 2 * limit),
                node_similarity_search(
                    driver, query_vector, group_ids, 2 * limit, config.sim_min_score
                ),
            ]
        )
    )

    search_result_uuids = [[node.uuid for node in result] for result in search_results]
    node_uuid_map = {node.uuid: node for result in search_results for node in result}

    reranked_uuids: list[str] = []
    if config.reranker == NodeReranker.rrf:
        reranked_uuids = rrf(search_result_uuids)
    elif config.reranker == NodeReranker.mmr:
        search_result_uuids_and_vectors = [
            (node.uuid, node.name_embedding if node.name_embedding is not None else [0.0] * 1024)
            for result in search_results
            for node in result
        ]
        reranked_uuids = maximal_marginal_relevance(
            query_vector, search_result_uuids_and_vectors, config.mmr_lambda
        )
    elif config.reranker == NodeReranker.cross_encoder:
        summary_to_uuid_map = {
            node.summary: node.uuid for result in search_results for node in result
        }
        reranked_summaries = await cross_encoder.rank(query, list(summary_to_uuid_map.keys()))
        reranked_uuids = [summary_to_uuid_map[fact] for fact, _ in reranked_summaries]
    elif config.reranker == NodeReranker.episode_mentions:
        reranked_uuids = await episode_mentions_reranker(driver, search_result_uuids)
    elif config.reranker == NodeReranker.node_distance:
        if center_node_uuid is None:
            raise SearchRerankerError('No center node provided for Node Distance reranker')
        reranked_uuids = await node_distance_reranker(
            driver, rrf(search_result_uuids), center_node_uuid
        )

    reranked_nodes = [node_uuid_map[uuid] for uuid in reranked_uuids]

    return reranked_nodes[:limit]


async def community_search(
    driver: AsyncDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: CommunitySearchConfig | None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
) -> list[CommunityNode]:
    if config is None:
        return []

    search_results: list[list[CommunityNode]] = list(
        await asyncio.gather(
            *[
                community_fulltext_search(driver, query, group_ids, 2 * limit),
                community_similarity_search(
                    driver, query_vector, group_ids, 2 * limit, config.sim_min_score
                ),
            ]
        )
    )

    search_result_uuids = [[community.uuid for community in result] for result in search_results]
    community_uuid_map = {
        community.uuid: community for result in search_results for community in result
    }

    reranked_uuids: list[str] = []
    if config.reranker == CommunityReranker.rrf:
        reranked_uuids = rrf(search_result_uuids)
    elif config.reranker == CommunityReranker.mmr:
        search_result_uuids_and_vectors = [
            (
                community.uuid,
                community.name_embedding if community.name_embedding is not None else [0.0] * 1024,
            )
            for result in search_results
            for community in result
        ]
        reranked_uuids = maximal_marginal_relevance(
            query_vector, search_result_uuids_and_vectors, config.mmr_lambda
        )
    elif config.reranker == CommunityReranker.cross_encoder:
        summary_to_uuid_map = {
            node.summary: node.uuid for result in search_results for node in result
        }
        reranked_summaries = await cross_encoder.rank(query, list(summary_to_uuid_map.keys()))
        reranked_uuids = [summary_to_uuid_map[fact] for fact, _ in reranked_summaries]

    reranked_communities = [community_uuid_map[uuid] for uuid in reranked_uuids]

    return reranked_communities[:limit]
