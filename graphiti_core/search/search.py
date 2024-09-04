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
from enum import Enum
from time import time

from neo4j import AsyncDriver
from pydantic import BaseModel, Field

from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.config import EMBEDDING_DIM
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_utils import (
    edge_fulltext_search,
    edge_similarity_search,
    get_mentioned_nodes,
    node_distance_reranker,
    rrf,
)
from graphiti_core.utils import retrieve_episodes
from graphiti_core.utils.maintenance.graph_data_operations import EPISODE_WINDOW_LEN

logger = logging.getLogger(__name__)


class SearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'


class Reranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    node_distance = 'node_distance'


class SearchConfig(BaseModel):
    num_edges: int = Field(default=10)
    num_nodes: int = Field(default=10)
    num_episodes: int = EPISODE_WINDOW_LEN
    search_methods: list[SearchMethod]
    reranker: Reranker | None


class SearchResults(BaseModel):
    episodes: list[EpisodicNode]
    nodes: list[EntityNode]
    edges: list[EntityEdge]


async def hybrid_search(
    driver: AsyncDriver,
    embedder,
    query: str,
    timestamp: datetime,
    config: SearchConfig,
    center_node_uuid: str | None = None,
) -> SearchResults:
    start = time()

    episodes = []
    nodes = []
    edges = []

    search_results = []

    if config.num_episodes > 0:
        episodes.extend(await retrieve_episodes(driver, timestamp, config.num_episodes))
        nodes.extend(await get_mentioned_nodes(driver, episodes))

    if SearchMethod.bm25 in config.search_methods:
        text_search = await edge_fulltext_search(driver, query, None, None, 2 * config.num_edges)
        search_results.append(text_search)

    if SearchMethod.cosine_similarity in config.search_methods:
        query_text = query.replace('\n', ' ')
        search_vector = (
            (await embedder.create(input=[query_text], model='text-embedding-3-small'))
            .data[0]
            .embedding[:EMBEDDING_DIM]
        )

        similarity_search = await edge_similarity_search(
            driver, search_vector, None, None, 2 * config.num_edges
        )
        search_results.append(similarity_search)

    if len(search_results) > 1 and config.reranker is None:
        logger.exception('Multiple searches enabled without a reranker')
        raise Exception('Multiple searches enabled without a reranker')

    else:
        edge_uuid_map = {}
        search_result_uuids = []

        for result in search_results:
            result_uuids = []
            for edge in result:
                result_uuids.append(edge.uuid)
                edge_uuid_map[edge.uuid] = edge

            search_result_uuids.append(result_uuids)

        search_result_uuids = [[edge.uuid for edge in result] for result in search_results]

        reranked_uuids: list[str] = []
        if config.reranker == Reranker.rrf:
            reranked_uuids = rrf(search_result_uuids)
        elif config.reranker == Reranker.node_distance:
            if center_node_uuid is None:
                logger.exception('No center node provided for Node Distance reranker')
                raise Exception('No center node provided for Node Distance reranker')
            reranked_uuids = await node_distance_reranker(
                driver, search_result_uuids, center_node_uuid
            )

        reranked_edges = [edge_uuid_map[uuid] for uuid in reranked_uuids]
        edges.extend(reranked_edges)

    context = SearchResults(
        episodes=episodes, nodes=nodes[: config.num_nodes], edges=edges[: config.num_edges]
    )

    end = time()

    logger.info(f'search returned context for query {query} in {(end - start) * 1000} ms')

    return context
