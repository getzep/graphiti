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
import re
from collections import defaultdict
from time import time

from neo4j import AsyncDriver, Query

from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodicNode,
    get_community_node_from_record,
    get_entity_node_from_record,
)

logger = logging.getLogger(__name__)

RELEVANT_SCHEMA_LIMIT = 3


async def get_mentioned_nodes(driver: AsyncDriver, episodes: list[EpisodicNode]):
    episode_uuids = [episode.uuid for episode in episodes]
    records, _, _ = await driver.execute_query(
        """
        MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity) WHERE episode.uuid IN $uuids
        RETURN DISTINCT
            n.uuid As uuid, 
            n.group_id AS group_id,
            n.name AS name,
            n.name_embedding AS name_embedding
            n.created_at AS created_at, 
            n.summary AS summary
        """,
        uuids=episode_uuids,
    )

    nodes = [get_entity_node_from_record(record) for record in records]

    return nodes


async def edge_fulltext_search(
    driver: AsyncDriver,
    query: str,
    source_node_uuid: str | None,
    target_node_uuid: str | None,
    group_ids: list[str | None] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EntityEdge]:
    # fulltext search over facts
    cypher_query = Query("""
              CALL db.index.fulltext.queryRelationships("name_and_fact", $query) 
              YIELD relationship AS rel, score
              MATCH (n:Entity {uuid: $source_uuid})-[r {uuid: rel.uuid}]-(m:Entity {uuid: $target_uuid})
              WHERE CASE
                WHEN $group_ids IS NULL THEN n.group_id IS NULL
                ELSE n.group_id IN $group_ids
              END
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
                """)

    if source_node_uuid is None and target_node_uuid is None:
        cypher_query = Query("""
                  CALL db.index.fulltext.queryRelationships("name_and_fact", $query) 
                  YIELD relationship AS rel, score
                  MATCH (n:Entity)-[r {uuid: rel.uuid}]-(m:Entity)
                  WHERE CASE
                    WHEN $group_ids IS NULL THEN r.group_id IS NULL
                    ELSE r.group_id IN $group_ids
                  END
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
                    """)
    elif source_node_uuid is None:
        cypher_query = Query("""
                  CALL db.index.fulltext.queryRelationships("name_and_fact", $query) 
                  YIELD relationship AS rel, score
                  MATCH (n:Entity)-[r {uuid: rel.uuid}]-(m:Entity {uuid: $target_uuid})              
                  WHERE CASE
                    WHEN $group_ids IS NULL THEN r.group_id IS NULL
                    ELSE r.group_id IN $group_ids
                  END
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
                    """)
    elif target_node_uuid is None:
        cypher_query = Query("""
                  CALL db.index.fulltext.queryRelationships("name_and_fact", $query) 
                  YIELD relationship AS rel, score
                  MATCH (n:Entity {uuid: $source_uuid})-[r {uuid: rel.uuid}]-(m:Entity)              
                  WHERE CASE
                    WHEN $group_ids IS NULL THEN r.group_id IS NULL
                    ELSE r.group_id IN $group_ids
                  END
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
                    """)

    fuzzy_query = re.sub(r'[^\w\s]', '', query) + '~'

    records, _, _ = await driver.execute_query(
        cypher_query,
        query=fuzzy_query,
        source_uuid=source_node_uuid,
        target_uuid=target_node_uuid,
        group_ids=group_ids,
        limit=limit,
    )

    edges = [get_entity_edge_from_record(record) for record in records]

    return edges


async def edge_similarity_search(
    driver: AsyncDriver,
    search_vector: list[float],
    source_node_uuid: str | None,
    target_node_uuid: str | None,
    group_ids: list[str | None] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EntityEdge]:
    # vector similarity search over embedded facts
    query = Query("""
                CALL db.index.vector.queryRelationships("fact_embedding", $limit, $search_vector)
                YIELD relationship AS rel, score
                MATCH (n:Entity {uuid: $source_uuid})-[r {uuid: rel.uuid}]-(m:Entity {uuid: $target_uuid})
                WHERE CASE
                    WHEN $group_ids IS NULL THEN r.group_id IS NULL
                    ELSE r.group_id IN $group_ids
                END
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
                ORDER BY score DESC
        """)

    if source_node_uuid is None and target_node_uuid is None:
        query = Query("""
                    CALL db.index.vector.queryRelationships("fact_embedding", $limit, $search_vector)
                    YIELD relationship AS rel, score
                    MATCH (n:Entity)-[r {uuid: rel.uuid}]-(m:Entity)
                    WHERE CASE
                        WHEN $group_ids IS NULL THEN r.group_id IS NULL
                        ELSE r.group_id IN $group_ids
                    END
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
                    ORDER BY score DESC
            """)
    elif source_node_uuid is None:
        query = Query("""
                    CALL db.index.vector.queryRelationships("fact_embedding", $limit, $search_vector)
                    YIELD relationship AS rel, score
                    MATCH (n:Entity)-[r {uuid: rel.uuid}]-(m:Entity {uuid: $target_uuid})
                    WHERE CASE
                        WHEN $group_ids IS NULL THEN r.group_id IS NULL
                        ELSE r.group_id IN $group_ids
                    END
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
                    ORDER BY score DESC
            """)
    elif target_node_uuid is None:
        query = Query("""
                    CALL db.index.vector.queryRelationships("fact_embedding", $limit, $search_vector)
                    YIELD relationship AS rel, score
                    MATCH (n:Entity {uuid: $source_uuid})-[r {uuid: rel.uuid}]-(m:Entity)
                    WHERE CASE
                        WHEN $group_ids IS NULL THEN r.group_id IS NULL
                        ELSE r.group_id IN $group_ids
                    END
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
                    ORDER BY score DESC
            """)

    records, _, _ = await driver.execute_query(
        query,
        search_vector=search_vector,
        source_uuid=source_node_uuid,
        target_uuid=target_node_uuid,
        group_ids=group_ids,
        limit=limit,
    )

    edges = [get_entity_edge_from_record(record) for record in records]

    return edges


async def node_fulltext_search(
    driver: AsyncDriver,
    query: str,
    group_ids: list[str | None] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    # BM25 search to get top nodes
    fuzzy_query = re.sub(r'[^\w\s]', '', query) + '~'
    records, _, _ = await driver.execute_query(
        """
    CALL db.index.fulltext.queryNodes("name_and_summary", $query) 
    YIELD node AS n, score
    WHERE CASE
        WHEN $group_ids IS NULL THEN n.group_id IS NULL
        ELSE n.group_id IN $group_ids
    END
    RETURN
        n.uuid AS uuid,
        n.group_id AS group_id, 
        n.name AS name, 
        n.name_embedding AS name_embedding,
        n.created_at AS created_at, 
        n.summary AS summary
    ORDER BY score DESC
    LIMIT $limit
    """,
        query=fuzzy_query,
        group_ids=group_ids,
        limit=limit,
    )
    nodes = [get_entity_node_from_record(record) for record in records]

    return nodes


async def node_similarity_search(
    driver: AsyncDriver,
    search_vector: list[float],
    group_ids: list[str | None] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    group_ids = group_ids if group_ids is not None else [None]

    # vector similarity search over entity names
    records, _, _ = await driver.execute_query(
        """
                CALL db.index.vector.queryNodes("name_embedding", $limit, $search_vector)
                YIELD node AS n, score
                MATCH (n WHERE n.group_id IN $group_ids)
                RETURN
                    n.uuid As uuid,
                    n.group_id AS group_id,
                    n.name AS name, 
                    n.name_embedding AS name_embedding,
                    n.created_at AS created_at, 
                    n.summary AS summary
                ORDER BY score DESC
                """,
        search_vector=search_vector,
        group_ids=group_ids,
        limit=limit,
    )
    nodes = [get_entity_node_from_record(record) for record in records]

    return nodes


async def community_fulltext_search(
    driver: AsyncDriver,
    query: str,
    group_ids: list[str | None] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[CommunityNode]:
    group_ids = group_ids if group_ids is not None else [None]

    # BM25 search to get top communities
    fuzzy_query = re.sub(r'[^\w\s]', '', query) + '~'
    records, _, _ = await driver.execute_query(
        """
    CALL db.index.fulltext.queryNodes("community_name", $query) 
    YIELD node AS comm, score
    MATCH (comm WHERE comm.group_id in $group_ids)
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
    )
    communities = [get_community_node_from_record(record) for record in records]

    return communities


async def community_similarity_search(
    driver: AsyncDriver,
    search_vector: list[float],
    group_ids: list[str | None] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[CommunityNode]:
    group_ids = group_ids if group_ids is not None else [None]

    # vector similarity search over entity names
    records, _, _ = await driver.execute_query(
        """
                CALL db.index.vector.queryNodes("community_name_embedding", $limit, $search_vector)
                YIELD node AS comm, score
                MATCH (comm WHERE comm.group_id IN $group_ids)
                RETURN
                    comm.uuid As uuid,
                    comm.group_id AS group_id,
                    comm.name AS name, 
                    comm.name_embedding AS name_embedding,
                    comm.created_at AS created_at, 
                    comm.summary AS summary
                ORDER BY score DESC
                """,
        search_vector=search_vector,
        group_ids=group_ids,
        limit=limit,
    )
    communities = [get_community_node_from_record(record) for record in records]

    return communities


async def hybrid_node_search(
    queries: list[str],
    embeddings: list[list[float]],
    driver: AsyncDriver,
    group_ids: list[str | None] | None = None,
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
        await asyncio.gather(
            *[node_fulltext_search(driver, q, group_ids, 2 * limit) for q in queries],
            *[node_similarity_search(driver, e, group_ids, 2 * limit) for e in embeddings],
        )
    )

    node_uuid_map: dict[str, EntityNode] = {
        node.uuid: node for result in results for node in result
    }
    result_uuids = [[node.uuid for node in result] for result in results]

    ranked_uuids = rrf(result_uuids)

    relevant_nodes: list[EntityNode] = [node_uuid_map[uuid] for uuid in ranked_uuids]

    end = time()
    logger.info(f'Found relevant nodes: {ranked_uuids} in {(end - start) * 1000} ms')
    return relevant_nodes


async def get_relevant_nodes(
    nodes: list[EntityNode],
    driver: AsyncDriver,
) -> list[EntityNode]:
    """
    Retrieve relevant nodes based on the provided list of EntityNodes.

    This method performs a hybrid search using both the names and embeddings
    of the input nodes to find relevant nodes in the graph database.

    Parameters
    ----------
    nodes : list[EntityNode]
        A list of EntityNode objects to use as the basis for the search.
    driver : AsyncDriver
        The Neo4j driver instance for database operations.

    Returns
    -------
    list[EntityNode]
        A list of EntityNode objects that are deemed relevant based on the input nodes.

    Notes
    -----
    This method uses the hybrid_node_search function to perform the search,
    which combines fulltext search and vector similarity search.
    It extracts the names and name embeddings (if available) from the input nodes
    to use as search criteria.
    """
    relevant_nodes = await hybrid_node_search(
        [node.name for node in nodes],
        [node.name_embedding for node in nodes if node.name_embedding is not None],
        driver,
        [node.group_id for node in nodes],
    )
    return relevant_nodes


async def get_relevant_edges(
    driver: AsyncDriver,
    edges: list[EntityEdge],
    source_node_uuid: str | None,
    target_node_uuid: str | None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EntityEdge]:
    start = time()
    relevant_edges: list[EntityEdge] = []
    relevant_edge_uuids = set()

    results = await asyncio.gather(
        *[
            edge_similarity_search(
                driver,
                edge.fact_embedding,
                source_node_uuid,
                target_node_uuid,
                [edge.group_id],
                limit,
            )
            for edge in edges
            if edge.fact_embedding is not None
        ],
        *[
            edge_fulltext_search(
                driver, edge.fact, source_node_uuid, target_node_uuid, [edge.group_id], limit
            )
            for edge in edges
        ],
    )

    for result in results:
        for edge in result:
            if edge.uuid in relevant_edge_uuids:
                continue

            relevant_edge_uuids.add(edge.uuid)
            relevant_edges.append(edge)

    end = time()
    logger.info(f'Found relevant edges: {relevant_edge_uuids} in {(end - start) * 1000} ms')

    return relevant_edges


# takes in a list of rankings of uuids
def rrf(results: list[list[str]], rank_const=1) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for result in results:
        for i, uuid in enumerate(result):
            scores[uuid] += 1 / (i + rank_const)

    scored_uuids = [term for term in scores.items()]
    scored_uuids.sort(reverse=True, key=lambda term: term[1])

    sorted_uuids = [term[0] for term in scored_uuids]

    return sorted_uuids


async def node_distance_reranker(
    driver: AsyncDriver, node_uuids: list[list[str]], center_node_uuid: str
) -> list[str]:
    # use rrf as a preliminary ranker
    sorted_uuids = rrf(node_uuids)
    scores: dict[str, float] = {}

    # Find the shortest path to center node
    query = Query("""  
        MATCH p = SHORTEST 1 (center:Entity {uuid: $center_uuid})-[:RELATES_TO]-+(n:Entity {uuid: $node_uuid})
        RETURN length(p) AS score
        """)

    path_results = await asyncio.gather(
        *[
            driver.execute_query(
                query,
                node_uuid=uuid,
                center_uuid=center_node_uuid,
            )
            for uuid in sorted_uuids
        ]
    )

    for uuid, result in zip(sorted_uuids, path_results):
        records = result[0]
        record = records[0] if len(records) > 0 else None
        distance: float = record['score'] if record is not None else float('inf')
        distance = 0 if uuid == center_node_uuid else distance
        scores[uuid] = distance

    # rerank on shortest distance
    sorted_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

    return sorted_uuids
