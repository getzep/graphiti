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
from typing import Any

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.errors import SearchRerankerError
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.search.search_config import (
    DEFAULT_RERANK_CANDIDATE_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
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


def _needs_falkordb_user_ids_post_filter(
    search_filter: SearchFilters, driver: Any
) -> bool:
    """Check if FalkorDB requires post-fetch user_ids filtering."""
    if search_filter.user_ids is None:
        return False
    try:
        from graphiti_core.driver.driver import GraphProvider

        provider_val = getattr(driver, 'provider', None)
        logger.debug(f"FalkorDB post-filter check: provider={provider_val}, user_ids={search_filter.user_ids}")
        return provider_val == GraphProvider.FALKORDB
    except ImportError:
        return False


async def _falkordb_filter_accessible_nodes(
    nodes: list[Any], driver: Any, group_ids: list[str], user_ids: list[str],
) -> list[Any]:
    """Filter nodes to only those with at least one episode whose user_id is in user_ids."""
    if not nodes:
        return nodes

    query = (
        "MATCH (e:Episodic)-[:MENTIONS]->(n:Entity) "
        "WHERE n.uuid IN $uuids AND n.group_id IN $group_ids "
        "AND e.user_id IN $user_ids "
        "RETURN DISTINCT n.uuid AS uuid"
    )

    uuids = [n.uuid for n in nodes]
    records, _, _ = await driver.execute_query(
        query, uuids=uuids, group_ids=group_ids, user_ids=user_ids, routing_='r',
    )
    accessible = {r['uuid'] for r in records}
    return [n for n in nodes if n.uuid in accessible]


async def _falkordb_filter_accessible_edges(
    edges: list[Any], driver: Any, group_ids: list[str], user_ids: list[str],
) -> list[Any]:
    """Filter edges to only those with at least one episode whose user_id is in user_ids."""
    if not edges:
        return edges

    all_episode_uuids: set[str] = set()
    edge_episode_map: dict[str, list[str]] = {}
    for e in edges:
        ep_uuids = getattr(e, 'episodes', []) or []
        if not ep_uuids:
            continue
        edge_episode_map[e.uuid] = ep_uuids
        all_episode_uuids.update(ep_uuids)

    if not all_episode_uuids:
        return edges

    query = (
        "MATCH (e:Episodic) WHERE e.uuid IN $uuids "
        "AND e.user_id IN $user_ids "
        "RETURN e.uuid AS uuid"
    )

    records, _, _ = await driver.execute_query(
        query, uuids=list(all_episode_uuids), user_ids=user_ids, routing_='r',
    )
    accessible_episodes = {r['uuid'] for r in records}

    return [
        e for e in edges
        if not edge_episode_map.get(e.uuid)
        or any(ep in accessible_episodes for ep in edge_episode_map[e.uuid])
    ]


async def search(
    clients: GraphitiClients,
    query: str,
    group_ids: list[str] | None,
    config: SearchConfig,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    query_vector: list[float] | None = None,
    driver: GraphDriver | None = None,
) -> SearchResults:
    start = time()

    driver = driver or clients.driver
    embedder = clients.embedder
    cross_encoder = clients.cross_encoder

    if query.strip() == '':
        return SearchResults()

    if (
        config.edge_config
        and EdgeSearchMethod.cosine_similarity in config.edge_config.search_methods
        or config.edge_config
        and EdgeReranker.mmr == config.edge_config.reranker
        or config.node_config
        and NodeSearchMethod.cosine_similarity in config.node_config.search_methods
        or config.node_config
        and NodeReranker.mmr == config.node_config.reranker
        or (
            config.community_config
            and CommunitySearchMethod.cosine_similarity in config.community_config.search_methods
        )
        or (config.community_config and CommunityReranker.mmr == config.community_config.reranker)
    ):
        search_vector = (
            query_vector
            if query_vector is not None
            else await embedder.create(input_data=[query.replace('\n', ' ')])
        )
    else:
        search_vector = [0.0] * EMBEDDING_DIM

    # if group_ids is empty, set it to None
    group_ids = group_ids if group_ids and group_ids != [''] else None

    # 计算各组件的检索数量
    # 如果使用 cross_encoder reranker，需要检索更多候选以便重排序
    def get_search_limit(component_config, default_limit: int) -> int:
        """根据 reranker 类型决定检索数量"""
        if component_config is None:
            return default_limit

        # 检查是否使用 cross_encoder reranker
        uses_cross_encoder = False
        if hasattr(component_config, 'reranker'):
            reranker = component_config.reranker
            if reranker in (
                EdgeReranker.cross_encoder,
                NodeReranker.cross_encoder,
                EpisodeReranker.cross_encoder,
                CommunityReranker.cross_encoder,
            ):
                uses_cross_encoder = True

        if uses_cross_encoder:
            # 使用 rerank_candidate_limit 或默认值
            candidate_limit = config.rerank_candidate_limit
            if candidate_limit is None:
                candidate_limit = max(default_limit, DEFAULT_RERANK_CANDIDATE_LIMIT)
            return candidate_limit
        return default_limit

    edge_limit = get_search_limit(config.edge_config, config.limit)
    node_limit = get_search_limit(config.node_config, config.limit)
    episode_limit = get_search_limit(config.episode_config, config.limit)
    community_limit = get_search_limit(config.community_config, config.limit)

    (
        (edges, edge_reranker_scores),
        (nodes, node_reranker_scores),
        (episodes, episode_reranker_scores),
        (communities, community_reranker_scores),
    ) = await semaphore_gather(
        edge_search(
            driver,
            cross_encoder,
            query,
            search_vector,
            group_ids,
            config.edge_config,
            search_filter,
            center_node_uuid,
            bfs_origin_node_uuids,
            edge_limit,
            config.reranker_min_score,
        ),
        node_search(
            driver,
            cross_encoder,
            query,
            search_vector,
            group_ids,
            config.node_config,
            search_filter,
            center_node_uuid,
            bfs_origin_node_uuids,
            node_limit,
            config.reranker_min_score,
        ),
        episode_search(
            driver,
            cross_encoder,
            query,
            search_vector,
            group_ids,
            config.episode_config,
            search_filter,
            episode_limit,
            config.reranker_min_score,
        ),
        community_search(
            driver,
            cross_encoder,
            query,
            search_vector,
            group_ids,
            config.community_config,
            community_limit,
            config.reranker_min_score,
        ),
    )

    # FalkorDB post-fetch user_ids filtering
    # FalkorDB does not support EXISTS subqueries in Cypher, so we filter
    # entities and edges after the search results are collected.
    if _needs_falkordb_user_ids_post_filter(search_filter, driver):
        nodes = await _falkordb_filter_accessible_nodes(
            nodes, driver, group_ids, search_filter.user_ids,
        )
        edges = await _falkordb_filter_accessible_edges(
            edges, driver, group_ids, search_filter.user_ids,
        )

    # 截取到用户请求的 limit
    results = SearchResults(
        edges=edges[: config.limit],
        edge_reranker_scores=edge_reranker_scores[: config.limit],
        nodes=nodes[: config.limit],
        node_reranker_scores=node_reranker_scores[: config.limit],
        episodes=episodes[: config.limit],
        episode_reranker_scores=episode_reranker_scores[: config.limit],
        communities=communities[: config.limit],
        community_reranker_scores=community_reranker_scores[: config.limit],
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
) -> tuple[list[EntityEdge], list[float]]:
    if config is None:
        return [], []

    # Build search tasks based on configured search methods
    search_tasks = []
    if EdgeSearchMethod.bm25 in config.search_methods:
        search_tasks.append(
            edge_fulltext_search(driver, query, search_filter, group_ids, 2 * limit)
        )
    if EdgeSearchMethod.cosine_similarity in config.search_methods:
        search_tasks.append(
            edge_similarity_search(
                driver,
                query_vector,
                None,
                None,
                search_filter,
                group_ids,
                2 * limit,
                config.sim_min_score,
            )
        )
    if EdgeSearchMethod.bfs in config.search_methods:
        search_tasks.append(
            edge_bfs_search(
                driver,
                bfs_origin_node_uuids,
                config.bfs_max_depth,
                search_filter,
                group_ids,
                2 * limit,
            )
        )

    # Execute only the configured search methods
    search_results: list[list[EntityEdge]] = []
    if search_tasks:
        search_results = list(await semaphore_gather(*search_tasks))

    if EdgeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
        source_node_uuids = [edge.source_node_uuid for result in search_results for edge in result]
        search_results.append(
            await edge_bfs_search(
                driver,
                source_node_uuids,
                config.bfs_max_depth,
                search_filter,
                group_ids,
                2 * limit,
            )
        )

    edge_uuid_map = {edge.uuid: edge for result in search_results for edge in result}

    reranked_uuids: list[str] = []
    edge_scores: list[float] = []
    if config.reranker == EdgeReranker.rrf or config.reranker == EdgeReranker.episode_mentions:
        search_result_uuids = [[edge.uuid for edge in result] for result in search_results]

        reranked_uuids, edge_scores = rrf(search_result_uuids, min_score=reranker_min_score)
    elif config.reranker == EdgeReranker.mmr:
        search_result_uuids_and_vectors = await get_embeddings_for_edges(
            driver, list(edge_uuid_map.values())
        )
        reranked_uuids, edge_scores = maximal_marginal_relevance(
            query_vector,
            search_result_uuids_and_vectors,
            config.mmr_lambda,
            reranker_min_score,
        )
    elif config.reranker == EdgeReranker.cross_encoder:
        fact_to_uuid_map = {edge.fact: edge.uuid for edge in list(edge_uuid_map.values())[:limit]}
        reranked_facts = await cross_encoder.rank(query, list(fact_to_uuid_map.keys()))
        reranked_uuids = [
            fact_to_uuid_map[fact] for fact, score in reranked_facts if score >= reranker_min_score
        ]
        edge_scores = [score for _, score in reranked_facts if score >= reranker_min_score]
    elif config.reranker == EdgeReranker.node_distance:
        if center_node_uuid is None:
            raise SearchRerankerError('No center node provided for Node Distance reranker')

        # use rrf as a preliminary sort
        sorted_result_uuids, node_scores = rrf(
            [[edge.uuid for edge in result] for result in search_results],
            min_score=reranker_min_score,
        )
        sorted_results = [edge_uuid_map[uuid] for uuid in sorted_result_uuids]

        # node distance reranking
        source_to_edge_uuid_map = defaultdict(list)
        for edge in sorted_results:
            source_to_edge_uuid_map[edge.source_node_uuid].append(edge.uuid)

        source_uuids = [source_node_uuid for source_node_uuid in source_to_edge_uuid_map]

        reranked_node_uuids, edge_scores = await node_distance_reranker(
            driver, source_uuids, center_node_uuid, min_score=reranker_min_score
        )

        for node_uuid in reranked_node_uuids:
            reranked_uuids.extend(source_to_edge_uuid_map[node_uuid])

    reranked_edges = [edge_uuid_map[uuid] for uuid in reranked_uuids]

    if config.reranker == EdgeReranker.episode_mentions:
        reranked_edges.sort(reverse=True, key=lambda edge: len(edge.episodes))

    return reranked_edges[:limit], edge_scores[:limit]


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
) -> tuple[list[EntityNode], list[float]]:
    if config is None:
        return [], []

    # Build search tasks based on configured search methods
    search_tasks = []
    if NodeSearchMethod.bm25 in config.search_methods:
        search_tasks.append(
            node_fulltext_search(driver, query, search_filter, group_ids, 2 * limit)
        )
    if NodeSearchMethod.cosine_similarity in config.search_methods:
        search_tasks.append(
            node_similarity_search(
                driver,
                query_vector,
                search_filter,
                group_ids,
                2 * limit,
                config.sim_min_score,
            )
        )
    if NodeSearchMethod.bfs in config.search_methods:
        search_tasks.append(
            node_bfs_search(
                driver,
                bfs_origin_node_uuids,
                search_filter,
                config.bfs_max_depth,
                group_ids,
                2 * limit,
            )
        )

    # Execute only the configured search methods
    search_results: list[list[EntityNode]] = []
    if search_tasks:
        search_results = list(await semaphore_gather(*search_tasks))

    if NodeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
        origin_node_uuids = [node.uuid for result in search_results for node in result]
        search_results.append(
            await node_bfs_search(
                driver,
                origin_node_uuids,
                search_filter,
                config.bfs_max_depth,
                group_ids,
                2 * limit,
            )
        )

    search_result_uuids = [[node.uuid for node in result] for result in search_results]
    node_uuid_map = {node.uuid: node for result in search_results for node in result}

    reranked_uuids: list[str] = []
    node_scores: list[float] = []
    if config.reranker == NodeReranker.rrf:
        reranked_uuids, node_scores = rrf(search_result_uuids, min_score=reranker_min_score)
    elif config.reranker == NodeReranker.mmr:
        search_result_uuids_and_vectors = await get_embeddings_for_nodes(
            driver, list(node_uuid_map.values())
        )

        reranked_uuids, node_scores = maximal_marginal_relevance(
            query_vector,
            search_result_uuids_and_vectors,
            config.mmr_lambda,
            reranker_min_score,
        )
    elif config.reranker == NodeReranker.cross_encoder:
        # EasyOps: 使用 name + summary 进行 cross_encoder reranking
        # 只用 name 会导致 reranker 无法判断相关性（例如问 "who is the father of X"，只有 summary 包含答案）
        text_to_uuid_map = {}
        for node in node_uuid_map.values():
            if node.summary:
                text = f"{node.name}: {node.summary}"
            else:
                text = node.name
            text_to_uuid_map[text] = node.uuid

        reranked_texts = await cross_encoder.rank(query, list(text_to_uuid_map.keys()))
        reranked_uuids = [
            text_to_uuid_map[text]
            for text, score in reranked_texts
            if score >= reranker_min_score
        ]
        node_scores = [score for _, score in reranked_texts if score >= reranker_min_score]
    elif config.reranker == NodeReranker.episode_mentions:
        reranked_uuids, node_scores = await episode_mentions_reranker(
            driver, search_result_uuids, min_score=reranker_min_score
        )
    elif config.reranker == NodeReranker.node_distance:
        if center_node_uuid is None:
            raise SearchRerankerError('No center node provided for Node Distance reranker')
        reranked_uuids, node_scores = await node_distance_reranker(
            driver,
            rrf(search_result_uuids, min_score=reranker_min_score)[0],
            center_node_uuid,
            min_score=reranker_min_score,
        )

    reranked_nodes = [node_uuid_map[uuid] for uuid in reranked_uuids]

    return reranked_nodes[:limit], node_scores[:limit]


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
) -> tuple[list[EpisodicNode], list[float]]:
    if config is None:
        return [], []
    search_results: list[list[EpisodicNode]] = list(
        await semaphore_gather(
            *[
                episode_fulltext_search(driver, query, search_filter, group_ids, 2 * limit),
            ]
        )
    )

    search_result_uuids = [[episode.uuid for episode in result] for result in search_results]
    episode_uuid_map = {episode.uuid: episode for result in search_results for episode in result}

    reranked_uuids: list[str] = []
    episode_scores: list[float] = []
    if config.reranker == EpisodeReranker.rrf:
        reranked_uuids, episode_scores = rrf(search_result_uuids, min_score=reranker_min_score)

    elif config.reranker == EpisodeReranker.cross_encoder:
        # use rrf as a preliminary reranker
        rrf_result_uuids, episode_scores = rrf(search_result_uuids, min_score=reranker_min_score)
        rrf_results = [episode_uuid_map[uuid] for uuid in rrf_result_uuids][:limit]

        content_to_uuid_map = {episode.content: episode.uuid for episode in rrf_results}

        reranked_contents = await cross_encoder.rank(query, list(content_to_uuid_map.keys()))
        reranked_uuids = [
            content_to_uuid_map[content]
            for content, score in reranked_contents
            if score >= reranker_min_score
        ]
        episode_scores = [score for _, score in reranked_contents if score >= reranker_min_score]

    reranked_episodes = [episode_uuid_map[uuid] for uuid in reranked_uuids]

    return reranked_episodes[:limit], episode_scores[:limit]


async def community_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: CommunitySearchConfig | None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
) -> tuple[list[CommunityNode], list[float]]:
    if config is None:
        return [], []

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

    search_result_uuids = [[community.uuid for community in result] for result in search_results]
    community_uuid_map = {
        community.uuid: community for result in search_results for community in result
    }

    reranked_uuids: list[str] = []
    community_scores: list[float] = []
    if config.reranker == CommunityReranker.rrf:
        reranked_uuids, community_scores = rrf(search_result_uuids, min_score=reranker_min_score)
    elif config.reranker == CommunityReranker.mmr:
        search_result_uuids_and_vectors = await get_embeddings_for_communities(
            driver, list(community_uuid_map.values())
        )

        reranked_uuids, community_scores = maximal_marginal_relevance(
            query_vector, search_result_uuids_and_vectors, config.mmr_lambda, reranker_min_score
        )
    elif config.reranker == CommunityReranker.cross_encoder:
        name_to_uuid_map = {node.name: node.uuid for result in search_results for node in result}
        reranked_nodes = await cross_encoder.rank(query, list(name_to_uuid_map.keys()))
        reranked_uuids = [
            name_to_uuid_map[name] for name, score in reranked_nodes if score >= reranker_min_score
        ]
        community_scores = [score for _, score in reranked_nodes if score >= reranker_min_score]

    reranked_communities = [community_uuid_map[uuid] for uuid in reranked_uuids]

    return reranked_communities[:limit], community_scores[:limit]
