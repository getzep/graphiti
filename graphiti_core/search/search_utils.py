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

import numpy as np
from neo4j import AsyncDriver, Query
from typing_extensions import LiteralString

from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.helpers import (
    DEFAULT_DATABASE,
    RUNTIME_QUERY,
    lucene_sanitize,
    normalize_l2,
    semaphore_gather,
)
from graphiti_core.nodes import (
    ENTITY_NODE_RETURN,
    CommunityNode,
    EntityNode,
    EpisodicNode,
    get_community_node_from_record,
    get_entity_node_from_record,
    get_episodic_node_from_record,
)
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

logger = logging.getLogger(__name__)

RELEVANT_SCHEMA_LIMIT = 10
DEFAULT_MIN_SCORE = 0.6
DEFAULT_MMR_LAMBDA = 0.5
MAX_SEARCH_DEPTH = 3
MAX_QUERY_LENGTH = 32


def fulltext_query(query: str, group_ids: list[str] | None = None):
    group_ids_filter_list = (
        [f'group_id:"{lucene_sanitize(g)}"' for g in group_ids] if group_ids is not None else []
    )
    group_ids_filter = ''
    for f in group_ids_filter_list:
        group_ids_filter += f if not group_ids_filter else f'OR {f}'

    group_ids_filter += ' AND ' if group_ids_filter else ''

    lucene_query = lucene_sanitize(query)
    # If the lucene query is too long return no query
    if len(lucene_query.split(' ')) + len(group_ids or '') >= MAX_QUERY_LENGTH:
        return ''

    full_query = group_ids_filter + '(' + lucene_query + ')'

    return full_query


async def get_episodes_by_mentions(
    driver: AsyncDriver,
    nodes: list[EntityNode],
    edges: list[EntityEdge],
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EpisodicNode]:
    episode_uuids: list[str] = []
    for edge in edges:
        episode_uuids.extend(edge.episodes)

    episodes = await EpisodicNode.get_by_uuids(driver, episode_uuids[:limit])

    return episodes


async def get_mentioned_nodes(
    driver: AsyncDriver, episodes: list[EpisodicNode]
) -> list[EntityNode]:
    episode_uuids = [episode.uuid for episode in episodes]
    records, _, _ = await driver.execute_query(
        """
        MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity) WHERE episode.uuid IN $uuids
        RETURN DISTINCT
            n.uuid As uuid, 
            n.group_id AS group_id,
            n.name AS name,
            n.name_embedding AS name_embedding,
            n.created_at AS created_at, 
            n.summary AS summary,
            labels(n) AS labels,
            properties(n) AS attributes
        """,
        uuids=episode_uuids,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )

    nodes = [get_entity_node_from_record(record) for record in records]

    return nodes


async def get_communities_by_nodes(
    driver: AsyncDriver, nodes: list[EntityNode]
) -> list[CommunityNode]:
    node_uuids = [node.uuid for node in nodes]
    records, _, _ = await driver.execute_query(
        """
    MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity) WHERE n.uuid IN $uuids
    RETURN DISTINCT
        c.uuid As uuid, 
        c.group_id AS group_id,
        c.name AS name,
        c.name_embedding AS name_embedding
        c.created_at AS created_at, 
        c.summary AS summary
    """,
        uuids=node_uuids,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )

    communities = [get_community_node_from_record(record) for record in records]

    return communities


async def edge_fulltext_search(
    driver: AsyncDriver,
    query: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EntityEdge]:
    # fulltext search over facts
    fuzzy_query = fulltext_query(query, group_ids)
    if fuzzy_query == '':
        return []

    filter_query, filter_params = edge_search_filter_query_constructor(search_filter)

    cypher_query = Query(
        """
              CALL db.index.fulltext.queryRelationships("edge_name_and_fact", $query, {limit: $limit}) 
              YIELD relationship AS rel, score
              MATCH (:Entity)-[r:RELATES_TO]->(:Entity)
              WHERE r.group_id IN $group_ids"""
        + filter_query
        + """\nWITH r, score, startNode(r) AS n, endNode(r) AS m
               RETURN
                     r.uuid AS uuid,
                     r.group_id AS group_id,
                     n.uuid AS source_node_uuid,
                     m.uuid AS target_node_uuid,
                     r.created_at AS created_at,
                     r.name AS name,
                     r.fact AS fact,
                     r.fact_embedding AS fact_embedding,
                     r.episodes AS episodes,
                     r.expired_at AS expired_at,
                     r.valid_at AS valid_at,
                     r.invalid_at AS invalid_at
                 ORDER BY score DESC LIMIT $limit
                 """
    )

    records, _, _ = await driver.execute_query(
        cypher_query,
        filter_params,
        query=fuzzy_query,
        group_ids=group_ids,
        limit=limit,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )

    edges = [get_entity_edge_from_record(record) for record in records]

    return edges


async def edge_similarity_search(
    driver: AsyncDriver,
    search_vector: list[float],
    source_node_uuid: str | None,
    target_node_uuid: str | None,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[EntityEdge]:
    # vector similarity search over embedded facts
    query_params: dict[str, Any] = {}

    filter_query, filter_params = edge_search_filter_query_constructor(search_filter)
    query_params.update(filter_params)

    group_filter_query: LiteralString = ''
    if group_ids is not None:
        group_filter_query += 'WHERE r.group_id IN $group_ids'
        query_params['group_ids'] = group_ids
        query_params['source_node_uuid'] = source_node_uuid
        query_params['target_node_uuid'] = target_node_uuid

        if source_node_uuid is not None:
            group_filter_query += '\nAND (n.uuid IN [$source_uuid, $target_uuid])'

        if target_node_uuid is not None:
            group_filter_query += '\nAND (m.uuid IN [$source_uuid, $target_uuid])'

    query: LiteralString = (
        RUNTIME_QUERY
        + """
                                                                                                                                                                MATCH (n:Entity)-[r:RELATES_TO]->(m:Entity)
                                                                                                                                               """
        + group_filter_query
        + filter_query
        + """\nWITH DISTINCT r, vector.similarity.cosine(r.fact_embedding, $search_vector) AS score
                WHERE score > $min_score
                RETURN
                    r.uuid AS uuid,
                    r.group_id AS group_id,
                    startNode(r).uuid AS source_node_uuid,
                    endNode(r).uuid AS target_node_uuid,
                    r.created_at AS created_at,
                    r.name AS name,
                    r.fact AS fact,
                    r.fact_embedding AS fact_embedding,
                    r.episodes AS episodes,
                    r.expired_at AS expired_at,
                    r.valid_at AS valid_at,
                    r.invalid_at AS invalid_at
                ORDER BY score DESC
                LIMIT $limit
        """
    )

    records, _, _ = await driver.execute_query(
        query,
        query_params,
        search_vector=search_vector,
        source_uuid=source_node_uuid,
        target_uuid=target_node_uuid,
        group_ids=group_ids,
        limit=limit,
        min_score=min_score,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )

    edges = [get_entity_edge_from_record(record) for record in records]

    return edges


async def edge_bfs_search(
    driver: AsyncDriver,
    bfs_origin_node_uuids: list[str] | None,
    bfs_max_depth: int,
    search_filter: SearchFilters,
    limit: int,
) -> list[EntityEdge]:
    # vector similarity search over embedded facts
    if bfs_origin_node_uuids is None:
        return []

    filter_query, filter_params = edge_search_filter_query_constructor(search_filter)

    query = Query(
        """
                UNWIND $bfs_origin_node_uuids AS origin_uuid
                MATCH path = (origin:Entity|Episodic {uuid: origin_uuid})-[:RELATES_TO|MENTIONS]->{1,3}(n:Entity)
                UNWIND relationships(path) AS rel
                MATCH ()-[r:RELATES_TO]-()
                WHERE r.uuid = rel.uuid
                """
        + filter_query
        + """  
                RETURN DISTINCT
                    r.uuid AS uuid,
                    r.group_id AS group_id,
                    startNode(r).uuid AS source_node_uuid,
                    endNode(r).uuid AS target_node_uuid,
                    r.created_at AS created_at,
                    r.name AS name,
                    r.fact AS fact,
                    r.fact_embedding AS fact_embedding,
                    r.episodes AS episodes,
                    r.expired_at AS expired_at,
                    r.valid_at AS valid_at,
                    r.invalid_at AS invalid_at
                LIMIT $limit
        """
    )

    records, _, _ = await driver.execute_query(
        query,
        filter_params,
        bfs_origin_node_uuids=bfs_origin_node_uuids,
        depth=bfs_max_depth,
        limit=limit,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )

    edges = [get_entity_edge_from_record(record) for record in records]

    return edges


async def node_fulltext_search(
    driver: AsyncDriver,
    query: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    # BM25 search to get top nodes
    fuzzy_query = fulltext_query(query, group_ids)
    if fuzzy_query == '':
        return []

    filter_query, filter_params = node_search_filter_query_constructor(search_filter)

    query = (
        """
                                        CALL db.index.fulltext.queryNodes("node_name_and_summary", $query, {limit: $limit}) 
                                        YIELD node AS n, score
                                        WHERE n:Entity
                                        """
        + filter_query
        + ENTITY_NODE_RETURN
        + """
        ORDER BY score DESC
        """
    )

    records, _, _ = await driver.execute_query(
        query,
        filter_params,
        query=fuzzy_query,
        group_ids=group_ids,
        limit=limit,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )
    nodes = [get_entity_node_from_record(record) for record in records]

    return nodes


async def node_similarity_search(
    driver: AsyncDriver,
    search_vector: list[float],
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[EntityNode]:
    # vector similarity search over entity names
    query_params: dict[str, Any] = {}

    group_filter_query: LiteralString = ''
    if group_ids is not None:
        group_filter_query += 'WHERE n.group_id IN $group_ids'
        query_params['group_ids'] = group_ids

    filter_query, filter_params = node_search_filter_query_constructor(search_filter)
    query_params.update(filter_params)

    records, _, _ = await driver.execute_query(
        RUNTIME_QUERY
        + """
            MATCH (n:Entity)
            """
        + group_filter_query
        + filter_query
        + """
            WITH n, vector.similarity.cosine(n.name_embedding, $search_vector) AS score
            WHERE score > $min_score"""
        + ENTITY_NODE_RETURN
        + """
        ORDER BY score DESC
        LIMIT $limit
        """,
        query_params,
        search_vector=search_vector,
        group_ids=group_ids,
        limit=limit,
        min_score=min_score,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )
    nodes = [get_entity_node_from_record(record) for record in records]

    return nodes


async def node_bfs_search(
    driver: AsyncDriver,
    bfs_origin_node_uuids: list[str] | None,
    search_filter: SearchFilters,
    bfs_max_depth: int,
    limit: int,
) -> list[EntityNode]:
    # vector similarity search over entity names
    if bfs_origin_node_uuids is None:
        return []

    filter_query, filter_params = node_search_filter_query_constructor(search_filter)

    records, _, _ = await driver.execute_query(
        """
            UNWIND $bfs_origin_node_uuids AS origin_uuid
            MATCH (origin:Entity|Episodic {uuid: origin_uuid})-[:RELATES_TO|MENTIONS]->{1,3}(n:Entity)
            WHERE n.group_id = origin.group_id
            """
        + filter_query
        + ENTITY_NODE_RETURN
        + """
        LIMIT $limit
        """,
        filter_params,
        bfs_origin_node_uuids=bfs_origin_node_uuids,
        depth=bfs_max_depth,
        limit=limit,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )
    nodes = [get_entity_node_from_record(record) for record in records]

    return nodes


async def episode_fulltext_search(
    driver: AsyncDriver,
    query: str,
    _search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EpisodicNode]:
    # BM25 search to get top episodes
    fuzzy_query = fulltext_query(query, group_ids)
    if fuzzy_query == '':
        return []

    records, _, _ = await driver.execute_query(
        """
        CALL db.index.fulltext.queryNodes("episode_content", $query, {limit: $limit}) 
        YIELD node AS episode, score
        MATCH (e:Episodic)
        WHERE e.uuid = episode.uuid
        RETURN 
            e.content AS content,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.uuid AS uuid,
            e.name AS name,
            e.group_id AS group_id,
            e.source_description AS source_description,
            e.source AS source,
            e.entity_edges AS entity_edges
        ORDER BY score DESC
        LIMIT $limit
        """,
        query=fuzzy_query,
        group_ids=group_ids,
        limit=limit,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )
    episodes = [get_episodic_node_from_record(record) for record in records]

    return episodes


async def community_fulltext_search(
    driver: AsyncDriver,
    query: str,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[CommunityNode]:
    # BM25 search to get top communities
    fuzzy_query = fulltext_query(query, group_ids)
    if fuzzy_query == '':
        return []

    records, _, _ = await driver.execute_query(
        """
        CALL db.index.fulltext.queryNodes("community_name", $query, {limit: $limit}) 
        YIELD node AS comm, score
        RETURN
            comm.uuid AS uuid,
            comm.group_id AS group_id, 
            comm.name AS name, 
            comm.name_embedding AS name_embedding,
            comm.created_at AS created_at, 
            comm.summary AS summary
        ORDER BY score DESC
        LIMIT $limit
        """,
        query=fuzzy_query,
        group_ids=group_ids,
        limit=limit,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )
    communities = [get_community_node_from_record(record) for record in records]

    return communities


async def community_similarity_search(
    driver: AsyncDriver,
    search_vector: list[float],
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
    min_score=DEFAULT_MIN_SCORE,
) -> list[CommunityNode]:
    # vector similarity search over entity names
    query_params: dict[str, Any] = {}

    group_filter_query: LiteralString = ''
    if group_ids is not None:
        group_filter_query += 'WHERE comm.group_id IN $group_ids'
        query_params['group_ids'] = group_ids

    records, _, _ = await driver.execute_query(
        RUNTIME_QUERY
        + """
           MATCH (comm:Community)
           """
        + group_filter_query
        + """
           WITH comm, vector.similarity.cosine(comm.name_embedding, $search_vector) AS score
           WHERE score > $min_score
           RETURN
               comm.uuid As uuid,
               comm.group_id AS group_id,
               comm.name AS name, 
               comm.name_embedding AS name_embedding,
               comm.created_at AS created_at, 
               comm.summary AS summary
           ORDER BY score DESC
           LIMIT $limit
        """,
        search_vector=search_vector,
        group_ids=group_ids,
        limit=limit,
        min_score=min_score,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )
    communities = [get_community_node_from_record(record) for record in records]

    return communities


async def hybrid_node_search(
    queries: list[str],
    embeddings: list[list[float]],
    driver: AsyncDriver,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    """
    Perform a hybrid search for nodes using both text queries and embeddings.

    This method combines fulltext search and vector similarity search to find
    relevant nodes in the graph database. It uses a rrf reranker.

    Parameters
    ----------
    queries : list[str]
        A list of text queries to search for.
    embeddings : list[list[float]]
        A list of embedding vectors corresponding to the queries. If empty only fulltext search is performed.
    driver : AsyncDriver
        The Neo4j driver instance for database operations.
    group_ids : list[str] | None, optional
        The list of group ids to retrieve nodes from.
    limit : int | None, optional
        The maximum number of results to return per search method. If None, a default limit will be applied.

    Returns
    -------
    list[EntityNode]
        A list of unique EntityNode objects that match the search criteria.

    Notes
    -----
    This method performs the following steps:
    1. Executes fulltext searches for each query.
    2. Executes vector similarity searches for each embedding.
    3. Combines and deduplicates the results from both search types.
    4. Logs the performance metrics of the search operation.

    The search results are deduplicated based on the node UUIDs to ensure
    uniqueness in the returned list. The 'limit' parameter is applied to each
    individual search method before deduplication. If not specified, a default
    limit (defined in the individual search functions) will be used.
    """

    start = time()
    results: list[list[EntityNode]] = list(
        await semaphore_gather(
            *[
                node_fulltext_search(driver, q, search_filter, group_ids, 2 * limit)
                for q in queries
            ],
            *[
                node_similarity_search(driver, e, search_filter, group_ids, 2 * limit)
                for e in embeddings
            ],
        )
    )

    node_uuid_map: dict[str, EntityNode] = {
        node.uuid: node for result in results for node in result
    }
    result_uuids = [[node.uuid for node in result] for result in results]

    ranked_uuids = rrf(result_uuids)

    relevant_nodes: list[EntityNode] = [node_uuid_map[uuid] for uuid in ranked_uuids]

    end = time()
    logger.debug(f'Found relevant nodes: {ranked_uuids} in {(end - start) * 1000} ms')
    return relevant_nodes


async def get_relevant_nodes(
    driver: AsyncDriver,
    nodes: list[EntityNode],
    search_filter: SearchFilters,
    min_score: float = DEFAULT_MIN_SCORE,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[list[EntityNode]]:
    if len(nodes) == 0:
        return []

    group_id = nodes[0].group_id

    # vector similarity search over entity names
    query_params: dict[str, Any] = {}

    filter_query, filter_params = node_search_filter_query_constructor(search_filter)
    query_params.update(filter_params)

    query = (
        RUNTIME_QUERY
        + """UNWIND $nodes AS node
    MATCH (n:Entity {group_id: $group_id})
            """
        + filter_query
        + """
        WITH node, n, vector.similarity.cosine(n.name_embedding, node.name_embedding) AS score
        WHERE score > $min_score
        WITH node, collect(n)[..$limit] AS top_vector_nodes, collect(n.uuid) AS vector_node_uuids
        
        CALL db.index.fulltext.queryNodes("node_name_and_summary", node.fulltext_query, {limit: $limit}) 
        YIELD node AS m
        WHERE m.group_id = $group_id
        WITH node, top_vector_nodes, vector_node_uuids, collect(m) AS fulltext_nodes
        
        WITH node, 
             top_vector_nodes, 
             [m IN fulltext_nodes WHERE NOT m.uuid IN vector_node_uuids] AS filtered_fulltext_nodes
        
        WITH node, top_vector_nodes + filtered_fulltext_nodes AS combined_nodes
        
        UNWIND combined_nodes AS combined_node
        WITH node, collect(DISTINCT combined_node) AS deduped_nodes
        
        RETURN 
          node.uuid AS search_node_uuid,
          [x IN deduped_nodes | {
            uuid: x.uuid, 
            name: x.name,
            name_embedding: x.name_embedding,
            group_id: x.group_id,
            created_at: x.created_at,
            summary: x.summary,
            labels: labels(x),
            attributes: properties(x)
          }] AS matches
        """
    )

    query_nodes = [
        {
            'uuid': node.uuid,
            'name': node.name,
            'name_embedding': node.name_embedding,
            'fulltext_query': fulltext_query(node.name, [node.group_id]),
        }
        for node in nodes
    ]

    results, _, _ = await driver.execute_query(
        query,
        query_params,
        nodes=query_nodes,
        group_id=group_id,
        limit=limit,
        min_score=min_score,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )

    relevant_nodes_dict: dict[str, list[EntityNode]] = {
        result['search_node_uuid']: [
            get_entity_node_from_record(record) for record in result['matches']
        ]
        for result in results
    }

    relevant_nodes = [relevant_nodes_dict.get(node.uuid, []) for node in nodes]

    return relevant_nodes


async def get_relevant_edges(
    driver: AsyncDriver,
    edges: list[EntityEdge],
    search_filter: SearchFilters,
    min_score: float = DEFAULT_MIN_SCORE,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[list[EntityEdge]]:
    if len(edges) == 0:
        return []

    query_params: dict[str, Any] = {}

    filter_query, filter_params = edge_search_filter_query_constructor(search_filter)
    query_params.update(filter_params)

    query = (
        RUNTIME_QUERY
        + """UNWIND $edges AS edge
    MATCH (n:Entity {uuid: edge.source_node_uuid})-[e:RELATES_TO {group_id: edge.group_id}]-(m:Entity {uuid: edge.target_node_uuid})
            """
        + filter_query
        + """
            WITH e, edge, vector.similarity.cosine(e.fact_embedding, edge.fact_embedding) AS score
            WHERE score > $min_score
            WITH edge, e, score
            ORDER BY score DESC
            RETURN edge.uuid AS search_edge_uuid,
                collect({
                    uuid: e.uuid,
                    source_node_uuid: startNode(e).uuid,
                    target_node_uuid: endNode(e).uuid,
                    created_at: e.created_at,
                    name: e.name,
                    group_id: e.group_id,
                    fact: e.fact,
                    fact_embedding: e.fact_embedding,
                    episodes: e.episodes,
                    expired_at: e.expired_at,
                    valid_at: e.valid_at,
                    invalid_at: e.invalid_at
                })[..$limit] AS matches
        """
    )

    results, _, _ = await driver.execute_query(
        query,
        query_params,
        edges=[edge.model_dump() for edge in edges],
        limit=limit,
        min_score=min_score,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )
    relevant_edges_dict: dict[str, list[EntityEdge]] = {
        result['search_edge_uuid']: [
            get_entity_edge_from_record(record) for record in result['matches']
        ]
        for result in results
    }

    relevant_edges = [relevant_edges_dict.get(edge.uuid, []) for edge in edges]

    return relevant_edges


async def get_edge_invalidation_candidates(
    driver: AsyncDriver,
    edges: list[EntityEdge],
    search_filter: SearchFilters,
    min_score: float = DEFAULT_MIN_SCORE,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[list[EntityEdge]]:
    if len(edges) == 0:
        return []

    query_params: dict[str, Any] = {}

    filter_query, filter_params = edge_search_filter_query_constructor(search_filter)
    query_params.update(filter_params)

    query = (
        RUNTIME_QUERY
        + """UNWIND $edges AS edge
    MATCH (n:Entity)-[e:RELATES_TO {group_id: edge.group_id}]->(m:Entity)
    WHERE n.uuid IN [edge.source_node_uuid, edge.target_node_uuid] OR m.uuid IN [edge.target_node_uuid, edge.source_node_uuid]
            """
        + filter_query
        + """
            WITH edge, e, vector.similarity.cosine(e.fact_embedding, edge.fact_embedding) AS score
            WHERE score > $min_score
            WITH edge, e, score
            ORDER BY score DESC
            RETURN edge.uuid AS search_edge_uuid,
                collect({
                    uuid: e.uuid,
                    source_node_uuid: startNode(e).uuid,
                    target_node_uuid: endNode(e).uuid,
                    created_at: e.created_at,
                    name: e.name,
                    group_id: e.group_id,
                    fact: e.fact,
                    fact_embedding: e.fact_embedding,
                    episodes: e.episodes,
                    expired_at: e.expired_at,
                    valid_at: e.valid_at,
                    invalid_at: e.invalid_at
                })[..$limit] AS matches
        """
    )

    results, _, _ = await driver.execute_query(
        query,
        query_params,
        edges=[edge.model_dump() for edge in edges],
        limit=limit,
        min_score=min_score,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )
    invalidation_edges_dict: dict[str, list[EntityEdge]] = {
        result['search_edge_uuid']: [
            get_entity_edge_from_record(record) for record in result['matches']
        ]
        for result in results
    }

    invalidation_edges = [invalidation_edges_dict.get(edge.uuid, []) for edge in edges]

    return invalidation_edges


# takes in a list of rankings of uuids
def rrf(results: list[list[str]], rank_const=1, min_score: float = 0) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for result in results:
        for i, uuid in enumerate(result):
            scores[uuid] += 1 / (i + rank_const)

    scored_uuids = [term for term in scores.items()]
    scored_uuids.sort(reverse=True, key=lambda term: term[1])

    sorted_uuids = [term[0] for term in scored_uuids]

    return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score]


async def node_distance_reranker(
    driver: AsyncDriver,
    node_uuids: list[str],
    center_node_uuid: str,
    min_score: float = 0,
) -> list[str]:
    # filter out node_uuid center node node uuid
    filtered_uuids = list(filter(lambda node_uuid: node_uuid != center_node_uuid, node_uuids))
    scores: dict[str, float] = {center_node_uuid: 0.0}

    # Find the shortest path to center node
    query = Query("""
        UNWIND $node_uuids AS node_uuid
        MATCH p = SHORTEST 1 (center:Entity {uuid: $center_uuid})-[:RELATES_TO]-+(n:Entity {uuid: node_uuid})
        RETURN length(p) AS score, node_uuid AS uuid
        """)

    path_results, _, _ = await driver.execute_query(
        query,
        node_uuids=filtered_uuids,
        center_uuid=center_node_uuid,
        database_=DEFAULT_DATABASE,
    )

    for result in path_results:
        uuid = result['uuid']
        score = result['score']
        scores[uuid] = score

    for uuid in filtered_uuids:
        if uuid not in scores:
            scores[uuid] = float('inf')

    # rerank on shortest distance
    filtered_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

    # add back in filtered center uuid if it was filtered out
    if center_node_uuid in node_uuids:
        scores[center_node_uuid] = 0.1
        filtered_uuids = [center_node_uuid] + filtered_uuids

    return [uuid for uuid in filtered_uuids if (1 / scores[uuid]) >= min_score]


async def episode_mentions_reranker(
    driver: AsyncDriver, node_uuids: list[list[str]], min_score: float = 0
) -> list[str]:
    # use rrf as a preliminary ranker
    sorted_uuids = rrf(node_uuids)
    scores: dict[str, float] = {}

    # Find the shortest path to center node
    query = Query("""
        UNWIND $node_uuids AS node_uuid 
        MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid: node_uuid})
        RETURN count(*) AS score, n.uuid AS uuid
        """)

    results, _, _ = await driver.execute_query(
        query,
        node_uuids=sorted_uuids,
        database_=DEFAULT_DATABASE,
    )

    for result in results:
        scores[result['uuid']] = result['score']

    # rerank on shortest distance
    sorted_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

    return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score]


def maximal_marginal_relevance(
    query_vector: list[float],
    candidates: list[tuple[str, list[float]]],
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
):
    candidates_with_mmr: list[tuple[str, float]] = []
    for candidate in candidates:
        max_sim = max([np.dot(normalize_l2(candidate[1]), normalize_l2(c[1])) for c in candidates])
        mmr = mmr_lambda * np.dot(candidate[1], query_vector) - (1 - mmr_lambda) * max_sim
        candidates_with_mmr.append((candidate[0], mmr))

    candidates_with_mmr.sort(reverse=True, key=lambda c: c[1])

    return list(set([candidate[0] for candidate in candidates_with_mmr]))
