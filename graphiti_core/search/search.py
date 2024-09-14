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

from neo4j import AsyncDriver

from graphiti_core.edges import EntityEdge
from graphiti_core.errors import SearchRerankerError
from graphiti_core.llm_client.config import EMBEDDING_DIM
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_config import SearchConfig, SearchResults, NodeSearchConfig, NodeSearchMethod, \
    EdgeSearchMethod, EdgeReranker, EdgeSearchConfig, NodeReranker
from graphiti_core.search.search_utils import (
    edge_fulltext_search,
    edge_similarity_search,
    get_mentioned_nodes,
    node_distance_reranker,
    rrf, node_fulltext_search, node_similarity_search,
)

logger = logging.getLogger(__name__)


async def search(
        driver: AsyncDriver,
        embedder,
        query: str,
        group_ids: list[str | None] | None,
        config: SearchConfig,
) -> SearchResults:
    start = time()
    query = query.replace('\n', ' ')

    edges = await edge_search(driver, embedder, query, group_ids, config.edge_config)

    context = SearchResults(
        episodes=episodes, nodes=nodes[: config.num_nodes], edges=edges[: config.num_edges]
    )

    end = time()

    logger.info(f'search returned context for query {query} in {(end - start) * 1000} ms')

    return context


async def edge_search(driver: AsyncDriver, embedder, query: str, group_ids: list[str | None] | None,
                      config: EdgeSearchConfig) -> list[EntityEdge]:
    search_results: list[list[EntityEdge]] = []

    if EdgeSearchMethod.bm25 in config.search_methods:
        text_search = await edge_fulltext_search(
            driver, query, None, None, config.group_ids, 2 * config.num_edges
        )
        search_results.append(text_search)

    if EdgeSearchMethod.cosine_similarity in config.search_methods:
        search_vector = (
            (await embedder.create(input=[query], model='text-embedding-3-small'))
            .data[0]
            .embedding[:EMBEDDING_DIM]
        )

        similarity_search = await edge_similarity_search(
            driver, search_vector, None, None, config.group_ids, 2 * config.num_edges
        )
        search_results.append(similarity_search)

        if len(search_results) > 1 and config.reranker is None:
            raise SearchRerankerError('Multiple edge searches enabled without a reranker')

        edge_uuid_map = {edge.uuid: edge for result in search_results for edge in result}
        search_result_uuids = [[edge.uuid for edge in result] for result in search_results]

        reranked_uuids: list[str] = []
        if config.reranker == EdgeReranker.rrf:
            reranked_uuids = rrf(search_result_uuids)
        elif config.reranker == EdgeReranker.node_distance:
            if config.center_node_uuid is None:
                raise SearchRerankerError('No center node provided for Node Distance reranker')

            source_to_edge_uuid_map = {edge.source_node_uuid: edge.uuid for result in search_results for edge in result}
            source_uuids = [[edge.source_node_uuid for edge in result] for result in search_results]

            reranked_node_uuids = await node_distance_reranker(driver, source_uuids, config.center_node_uuid)

            reranked_uuids = [source_to_edge_uuid_map[node_uuid] for node_uuid in reranked_node_uuids]

        reranked_edges = [edge_uuid_map[uuid] for uuid in reranked_uuids]

        return reranked_edges


async def node_search(driver: AsyncDriver, embedder, query: str, group_ids: list[str | None] | None,
                      config: NodeSearchConfig) -> list[EntityNode]:
    search_results: list[list[EntityNode]] = []

    if NodeSearchMethod.bm25 in config.search_methods:
        text_search = await node_fulltext_search(driver, query, group_ids, 2 * config.num_nodes)
        search_results.append(text_search)

    if NodeSearchMethod.cosine_similarity in config.search_methods:
        search_vector = (
            (await embedder.create(input=[query], model='text-embedding-3-small'))
            .data[0]
            .embedding[:EMBEDDING_DIM]
        )

        similarity_search = await node_similarity_search(
            driver, search_vector, config.group_ids, 2 * config.num_edges
        )
        search_results.append(similarity_search)

    if len(search_results) > 1 and config.reranker is None:
        raise SearchRerankerError('Multiple node searches enabled without a reranker')

    node_uuid_map: dict[str: EntityNode] = {}
    search_result_uuids: list[list[str]] = []

    for result in search_results:
        result_uuids = []
        for node in result:
            result_uuids.append(node.uuid)
            node_uuid_map[node.uuid] = node

        search_result_uuids.append(result_uuids)

    search_result_uuids = [[node.uuid for node in result] for result in search_results]

    reranked_uuids: list[str] = []
    if config.reranker == NodeReranker.rrf:
        reranked_uuids = rrf(search_result_uuids)
    elif config.reranker == NodeReranker.node_distance:
        if config.center_node_uuid is None:
            raise SearchRerankerError('No center node provided for Node Distance reranker')
        reranked_uuids = await node_distance_reranker(
            driver, search_result_uuids, config.center_node_uuid
        )

    reranked_nodes = [node_uuid_map[uuid] for uuid in reranked_uuids]

    return reranked_nodes
