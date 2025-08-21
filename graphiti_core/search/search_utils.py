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
from numpy._typing import NDArray
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.graph_queries import (
    get_nodes_query,
    get_relationships_query,
    get_vector_cosine_func_query,
)
from graphiti_core.helpers import (
    RUNTIME_QUERY,
    lucene_sanitize,
    normalize_l2,
    semaphore_gather,
)
from graphiti_core.models.edges.edge_db_queries import ENTITY_EDGE_RETURN
from graphiti_core.models.nodes.node_db_queries import COMMUNITY_NODE_RETURN, EPISODIC_NODE_RETURN
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
MAX_QUERY_LENGTH = 128


def calculate_cosine_similarity(vector1: list[float], vector2: list[float]) -> float:
    """
    Calculates the cosine similarity between two vectors using NumPy.
    """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0  # Handle cases where one or both vectors are zero vectors

    return dot_product / (norm_vector1 * norm_vector2)


def fulltext_query(query: str, group_ids: list[str] | None = None, fulltext_syntax: str = ''):
    group_ids_filter_list = (
        [fulltext_syntax + f'group_id:"{g}"' for g in group_ids] if group_ids is not None else []
    )
    group_ids_filter = ''
    for f in group_ids_filter_list:
        group_ids_filter += f if not group_ids_filter else f' OR {f}'

    group_ids_filter += ' AND ' if group_ids_filter else ''

    lucene_query = lucene_sanitize(query)
    # If the lucene query is too long return no query
    if len(lucene_query.split(' ')) + len(group_ids or '') >= MAX_QUERY_LENGTH:
        return ''

    full_query = group_ids_filter + '(' + lucene_query + ')'

    return full_query


async def get_episodes_by_mentions(
    driver: GraphDriver,
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
    driver: GraphDriver, episodes: list[EpisodicNode]
) -> list[EntityNode]:
    episode_uuids = [episode.uuid for episode in episodes]

    records, _, _ = await driver.execute_query(
        """
        MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity)
        WHERE episode.uuid IN $uuids
        RETURN DISTINCT
        """
        + ENTITY_NODE_RETURN,
        uuids=episode_uuids,
        routing_='r',
    )

    nodes = [get_entity_node_from_record(record) for record in records]

    return nodes


async def get_communities_by_nodes(
    driver: GraphDriver, nodes: list[EntityNode]
) -> list[CommunityNode]:
    node_uuids = [node.uuid for node in nodes]

    records, _, _ = await driver.execute_query(
        """
        MATCH (n:Community)-[:HAS_MEMBER]->(m:Entity)
        WHERE m.uuid IN $uuids
        RETURN DISTINCT
        """
        + COMMUNITY_NODE_RETURN,
        uuids=node_uuids,
        routing_='r',
    )

    communities = [get_community_node_from_record(record) for record in records]

    return communities


async def edge_fulltext_search(
    driver: GraphDriver,
    query: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EntityEdge]:
    # fulltext search over facts
    fuzzy_query = fulltext_query(query, group_ids, driver.fulltext_syntax)
    if fuzzy_query == '':
        return []

    filter_query, filter_params = edge_search_filter_query_constructor(search_filter)

    if driver.provider == GraphProvider.NEPTUNE:
        res = driver.run_aoss_query('edge_name_and_fact', query)  # pyright: ignore reportAttributeAccessIssue
        if res['hits']['total']['value'] > 0:
            # Calculate Cosine similarity then return the edge ids
            input_ids = []
            for r in res['hits']['hits']:
                input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

            # Match the edge ids and return the values
            query = (
                """
                UNWIND $ids as id
                MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
                WHERE e.group_id IN $group_ids 
                AND id(e)=id 
                """
                + filter_query
                + """
                WITH e, id.score as score, startNode(e) AS n, endNode(e) AS m
                RETURN
                    e.uuid AS uuid,
                    e.group_id AS group_id,
                    n.uuid AS source_node_uuid,
                    m.uuid AS target_node_uuid,
                    e.created_at AS created_at,
                    e.name AS name,
                    e.fact AS fact,
                    split(e.episodes, ",") AS episodes,
                    e.expired_at AS expired_at,
                    e.valid_at AS valid_at,
                    e.invalid_at AS invalid_at,
                    properties(e) AS attributes
                ORDER BY score DESC LIMIT $limit
                            """
            )

            records, _, _ = await driver.execute_query(
                query,
                query=fuzzy_query,
                group_ids=group_ids,
                ids=input_ids,
                limit=limit,
                routing_='r',
                **filter_params,
            )
        else:
            return []
    else:
        query = (
            get_relationships_query('edge_name_and_fact', provider=driver.provider)
            + """
            YIELD relationship AS rel, score
            MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)
            WHERE e.group_id IN $group_ids """
            + filter_query
            + """
            WITH e, score, n, m
            RETURN
            """
            + ENTITY_EDGE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await driver.execute_query(
            query,
            query=fuzzy_query,
            group_ids=group_ids,
            limit=limit,
            routing_='r',
            **filter_params,
        )

    edges = [get_entity_edge_from_record(record) for record in records]

    return edges


async def edge_similarity_search(
    driver: GraphDriver,
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

    group_filter_query: LiteralString = 'WHERE e.group_id IS NOT NULL'
    if group_ids is not None:
        group_filter_query += '\nAND e.group_id IN $group_ids'
        query_params['group_ids'] = group_ids

        if source_node_uuid is not None:
            query_params['source_uuid'] = source_node_uuid
            group_filter_query += '\nAND (n.uuid = $source_uuid)'

        if target_node_uuid is not None:
            query_params['target_uuid'] = target_node_uuid
            group_filter_query += '\nAND (m.uuid = $target_uuid)'

    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            RUNTIME_QUERY
            + """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)            
            """
            + group_filter_query
            + filter_query
            + """
            RETURN DISTINCT id(e) as id, e.fact_embedding as embedding
            """
        )
        resp, header, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
        )

        if len(resp) > 0:
            # Calculate Cosine similarity then return the edge ids
            input_ids = []
            for r in resp:
                if r['embedding']:
                    score = calculate_cosine_similarity(
                        search_vector, list(map(float, r['embedding'].split(',')))
                    )
                    if score > min_score:
                        input_ids.append({'id': r['id'], 'score': score})

            # Match the edge ides and return the values
            query = """
                UNWIND $ids as i
                MATCH ()-[r]->()
                WHERE id(r) = i.id
                RETURN
                    r.uuid AS uuid,
                    r.group_id AS group_id,
                    startNode(r).uuid AS source_node_uuid,
                    endNode(r).uuid AS target_node_uuid,
                    r.created_at AS created_at,
                    r.name AS name,
                    r.fact AS fact,
                    split(r.episodes, ",") AS episodes,
                    r.expired_at AS expired_at,
                    r.valid_at AS valid_at,
                    r.invalid_at AS invalid_at,
                    properties(r) AS attributes
                ORDER BY i.score DESC
                LIMIT $limit
                    """
            records, _, _ = await driver.execute_query(
                query,
                ids=input_ids,
                search_vector=search_vector,
                limit=limit,
                min_score=min_score,
                routing_='r',
                **query_params,
            )
        else:
            return []
    else:
        query = (
            RUNTIME_QUERY
            + """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            """
            + group_filter_query
            + filter_query
            + """
            WITH DISTINCT e, n, m, """
            + get_vector_cosine_func_query('e.fact_embedding', '$search_vector', driver.provider)
            + """ AS score
            WHERE score > $min_score
            RETURN
            """
            + ENTITY_EDGE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
        )

    edges = [get_entity_edge_from_record(record) for record in records]

    return edges


async def edge_bfs_search(
    driver: GraphDriver,
    bfs_origin_node_uuids: list[str] | None,
    bfs_max_depth: int,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EntityEdge]:
    # vector similarity search over embedded facts
    if bfs_origin_node_uuids is None:
        return []

    filter_query, filter_params = edge_search_filter_query_constructor(search_filter)

    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            f"""
                    UNWIND $bfs_origin_node_uuids AS origin_uuid
                    MATCH path = (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS *1..{bfs_max_depth}]->(n:Entity)
                    WHERE origin:Entity OR origin:Episodic
                    UNWIND relationships(path) AS rel
                    MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
                    WHERE e.uuid = rel.uuid
                """
            + filter_query
            + """  
                        RETURN DISTINCT
                            e.uuid AS uuid,
                            e.group_id AS group_id,
                            startNode(e).uuid AS source_node_uuid,
                            endNode(e).uuid AS target_node_uuid,
                            e.created_at AS created_at,
                            e.name AS name,
                            e.fact AS fact,
                            split(e.episodes, ',') AS episodes,
                            e.expired_at AS expired_at,
                            e.valid_at AS valid_at,
                            e.invalid_at AS invalid_at,
                            properties(e) AS attributes
                        LIMIT $limit
                """
        )
    else:
        query = (
            f"""
                UNWIND $bfs_origin_node_uuids AS origin_uuid
                MATCH path = (origin:Entity|Episodic {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{bfs_max_depth}]->(:Entity)
                UNWIND relationships(path) AS rel
                MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
                WHERE e.uuid = rel.uuid
                AND e.group_id IN $group_ids
            """
            + filter_query
            + """
            RETURN DISTINCT
            """
            + ENTITY_EDGE_RETURN
            + """
            LIMIT $limit
            """
        )

    records, _, _ = await driver.execute_query(
        query,
        bfs_origin_node_uuids=bfs_origin_node_uuids,
        depth=bfs_max_depth,
        group_ids=group_ids,
        limit=limit,
        routing_='r',
        **filter_params,
    )

    edges = [get_entity_edge_from_record(record) for record in records]

    return edges


async def node_fulltext_search(
    driver: GraphDriver,
    query: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    # BM25 search to get top nodes
    fuzzy_query = fulltext_query(query, group_ids, driver.fulltext_syntax)
    if fuzzy_query == '':
        return []
    filter_query, filter_params = node_search_filter_query_constructor(search_filter)

    if driver.provider == GraphProvider.NEPTUNE:
        res = driver.run_aoss_query('node_name_and_summary', query, limit=limit)  # pyright: ignore reportAttributeAccessIssue
        if res['hits']['total']['value'] > 0:
            # Calculate Cosine similarity then return the edge ids
            input_ids = []
            for r in res['hits']['hits']:
                input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

            # Match the edge ides and return the values
            query = (
                """
                UNWIND $ids as i
                MATCH (n:Entity)
                WHERE n.uuid=i.id
                RETURN
                """
                + ENTITY_NODE_RETURN
                + """
                ORDER BY i.score DESC
                LIMIT $limit
                            """
            )
            records, _, _ = await driver.execute_query(
                query,
                ids=input_ids,
                query=fuzzy_query,
                group_ids=group_ids,
                limit=limit,
                routing_='r',
                **filter_params,
            )
        else:
            return []
    else:
        query = (
            get_nodes_query(driver.provider, 'node_name_and_summary', '$query')
            + """
          YIELD node AS n, score
          WHERE n:Entity AND n.group_id IN $group_ids
          """
            + filter_query
            + """
          WITH n, score
          ORDER BY score DESC
          LIMIT $limit
          RETURN
          """
            + ENTITY_NODE_RETURN
        )

        records, _, _ = await driver.execute_query(
            query,
            query=fuzzy_query,
            group_ids=group_ids,
            limit=limit,
            routing_='r',
            **filter_params,
        )

    nodes = [get_entity_node_from_record(record) for record in records]

    return nodes


async def node_similarity_search(
    driver: GraphDriver,
    search_vector: list[float],
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[EntityNode]:
    # vector similarity search over entity names
    query_params: dict[str, Any] = {}

    group_filter_query: LiteralString = 'WHERE n.group_id IS NOT NULL'
    if group_ids is not None:
        group_filter_query += ' AND n.group_id IN $group_ids'
        query_params['group_ids'] = group_ids

    filter_query, filter_params = node_search_filter_query_constructor(search_filter)
    query_params.update(filter_params)

    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            RUNTIME_QUERY
            + """
            MATCH (n:Entity)
            """
            + group_filter_query
            + filter_query
            + """
            RETURN DISTINCT id(n) as id, n.name_embedding as embedding
            """
        )
        resp, header, _ = await driver.execute_query(
            query,
            params=query_params,
            search_vector=search_vector,
            group_ids=group_ids,
            limit=limit,
            min_score=min_score,
            routing_='r',
        )

        if len(resp) > 0:
            # Calculate Cosine similarity then return the edge ids
            input_ids = []
            for r in resp:
                if r['embedding']:
                    score = calculate_cosine_similarity(
                        search_vector, list(map(float, r['embedding'].split(',')))
                    )
                    if score > min_score:
                        input_ids.append({'id': r['id'], 'score': score})

            # Match the edge ides and return the values
            query = (
                """
                    UNWIND $ids as i
                    MATCH (n:Entity)
                    WHERE id(n)=i.id
                    RETURN 
                    """
                + ENTITY_NODE_RETURN
                + """
                    ORDER BY i.score DESC
                    LIMIT $limit
                """
            )
            records, header, _ = await driver.execute_query(
                query,
                ids=input_ids,
                search_vector=search_vector,
                limit=limit,
                min_score=min_score,
                routing_='r',
                **query_params,
            )
        else:
            return []
    else:
        query = (
            RUNTIME_QUERY
            + """
            MATCH (n:Entity)
            """
            + group_filter_query
            + filter_query
            + """
            WITH n, """
            + get_vector_cosine_func_query('n.name_embedding', '$search_vector', driver.provider)
            + """ AS score
            WHERE score > $min_score
            RETURN
            """
            + ENTITY_NODE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
        )

    nodes = [get_entity_node_from_record(record) for record in records]

    return nodes


async def node_bfs_search(
    driver: GraphDriver,
    bfs_origin_node_uuids: list[str] | None,
    search_filter: SearchFilters,
    bfs_max_depth: int,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    # vector similarity search over entity names
    if bfs_origin_node_uuids is None:
        return []

    filter_query, filter_params = node_search_filter_query_constructor(search_filter)

    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            f"""
                UNWIND $bfs_origin_node_uuids AS origin_uuid            
                MATCH (origin {{uuid: origin_uuid}})-[e:RELATES_TO|MENTIONS*1..{bfs_max_depth}]->(n:Entity)
                WHERE origin:Entity OR origin.Episode
                AND n.group_id = origin.group_id
            """
            + filter_query
            + """
            RETURN
            """
            + ENTITY_NODE_RETURN
            + """
            LIMIT $limit
            """
        )
    else:
        query = (
            f"""
                UNWIND $bfs_origin_node_uuids AS origin_uuid
                MATCH (origin:Entity|Episodic {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{bfs_max_depth}]->(n:Entity)
                WHERE n.group_id = origin.group_id
                AND origin.group_id IN $group_ids
            """
            + filter_query
            + """
            RETURN
            """
            + ENTITY_NODE_RETURN
            + """
            LIMIT $limit
            """
        )

    records, _, _ = await driver.execute_query(
        query,
        bfs_origin_node_uuids=bfs_origin_node_uuids,
        group_ids=group_ids,
        limit=limit,
        routing_='r',
        **filter_params,
    )
    nodes = [get_entity_node_from_record(record) for record in records]

    return nodes


async def episode_fulltext_search(
    driver: GraphDriver,
    query: str,
    _search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EpisodicNode]:
    # BM25 search to get top episodes
    fuzzy_query = fulltext_query(query, group_ids, driver.fulltext_syntax)
    if fuzzy_query == '':
        return []

    if driver.provider == GraphProvider.NEPTUNE:
        res = driver.run_aoss_query('episode_content', query, limit=limit)  # pyright: ignore reportAttributeAccessIssue
        if res['hits']['total']['value'] > 0:
            # Calculate Cosine similarity then return the edge ids
            input_ids = []
            for r in res['hits']['hits']:
                input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

            # Match the edge ides and return the values
            query = """
                UNWIND $ids as i
                MATCH (e:Episodic)
                WHERE e.uuid=i.id
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
                ORDER BY i.score DESC
                LIMIT $limit
            """
            records, _, _ = await driver.execute_query(
                query,
                ids=input_ids,
                query=fuzzy_query,
                group_ids=group_ids,
                limit=limit,
                routing_='r',
            )
        else:
            return []
    else:
        query = (
            get_nodes_query(driver.provider, 'episode_content', '$query')
            + """
            YIELD node AS episode, score
            MATCH (e:Episodic)
            WHERE e.uuid = episode.uuid
            AND e.group_id IN $group_ids
            RETURN
            """
            + EPISODIC_NODE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await driver.execute_query(
            query,
            query=fuzzy_query,
            group_ids=group_ids,
            limit=limit,
            routing_='r',
        )
    episodes = [get_episodic_node_from_record(record) for record in records]

    return episodes


async def community_fulltext_search(
    driver: GraphDriver,
    query: str,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[CommunityNode]:
    # BM25 search to get top communities
    fuzzy_query = fulltext_query(query, group_ids, driver.fulltext_syntax)
    if fuzzy_query == '':
        return []

    if driver.provider == GraphProvider.NEPTUNE:
        res = driver.run_aoss_query('community_name', query, limit=limit)  # pyright: ignore reportAttributeAccessIssue
        if res['hits']['total']['value'] > 0:
            # Calculate Cosine similarity then return the edge ids
            input_ids = []
            for r in res['hits']['hits']:
                input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

            # Match the edge ides and return the values
            query = """
                UNWIND $ids as i
                MATCH (comm:Community)
                WHERE comm.uuid=i.id
                RETURN
                    comm.uuid AS uuid,
                    comm.group_id AS group_id, 
                    comm.name AS name, 
                    comm.created_at AS created_at, 
                    comm.summary AS summary,
                    [x IN split(comm.name_embedding, ",") | toFloat(x)]AS name_embedding
                ORDER BY i.score DESC
                LIMIT $limit
            """
            records, _, _ = await driver.execute_query(
                query,
                ids=input_ids,
                query=fuzzy_query,
                group_ids=group_ids,
                limit=limit,
                routing_='r',
            )
        else:
            return []
    else:
        query = (
            get_nodes_query(driver.provider, 'community_name', '$query')
            + """
            YIELD node AS n, score
            WHERE n.group_id IN $group_ids
            RETURN
            """
            + COMMUNITY_NODE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await driver.execute_query(
            query,
            query=fuzzy_query,
            group_ids=group_ids,
            limit=limit,
            routing_='r',
        )
    communities = [get_community_node_from_record(record) for record in records]

    return communities


async def community_similarity_search(
    driver: GraphDriver,
    search_vector: list[float],
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
    min_score=DEFAULT_MIN_SCORE,
) -> list[CommunityNode]:
    # vector similarity search over entity names
    query_params: dict[str, Any] = {}

    group_filter_query: LiteralString = ''
    if group_ids is not None:
        group_filter_query += 'WHERE n.group_id IN $group_ids'
        query_params['group_ids'] = group_ids

    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            RUNTIME_QUERY
            + """
            MATCH (n:Community)
            """
            + group_filter_query
            + """
            RETURN DISTINCT id(n) as id, n.name_embedding as embedding
            """
        )
        resp, header, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
        )

        if len(resp) > 0:
            # Calculate Cosine similarity then return the edge ids
            input_ids = []
            for r in resp:
                if r['embedding']:
                    score = calculate_cosine_similarity(
                        search_vector, list(map(float, r['embedding'].split(',')))
                    )
                    if score > min_score:
                        input_ids.append({'id': r['id'], 'score': score})

            # Match the edge ides and return the values
            query = """
                    UNWIND $ids as i
                    MATCH (comm:Community)
                    WHERE id(comm)=i.id
                    RETURN
                        comm.uuid As uuid,
                        comm.group_id AS group_id,
                        comm.name AS name, 
                        comm.created_at AS created_at, 
                        comm.summary AS summary,
                        comm.name_embedding AS name_embedding
                    ORDER BY i.score DESC
                    LIMIT $limit
                """
            records, header, _ = await driver.execute_query(
                query,
                ids=input_ids,
                search_vector=search_vector,
                limit=limit,
                min_score=min_score,
                routing_='r',
                **query_params,
            )
        else:
            return []
    else:
        query = (
            RUNTIME_QUERY
            + """
            MATCH (n:Community)
            """
            + group_filter_query
            + """
            WITH n,
            """
            + get_vector_cosine_func_query('n.name_embedding', '$search_vector', driver.provider)
            + """ AS score
            WHERE score > $min_score
            RETURN
            """
            + COMMUNITY_NODE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        records, _, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
        )
    communities = [get_community_node_from_record(record) for record in records]

    return communities


async def hybrid_node_search(
    queries: list[str],
    embeddings: list[list[float]],
    driver: GraphDriver,
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
    driver : GraphDriver
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

    ranked_uuids, _ = rrf(result_uuids)

    relevant_nodes: list[EntityNode] = [node_uuid_map[uuid] for uuid in ranked_uuids]

    end = time()
    logger.debug(f'Found relevant nodes: {ranked_uuids} in {(end - start) * 1000} ms')
    return relevant_nodes


async def get_relevant_nodes(
    driver: GraphDriver,
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
        + """
        UNWIND $nodes AS node
        MATCH (n:Entity {group_id: $group_id})
        """
        + filter_query
        + """
        WITH node, n, """
        + get_vector_cosine_func_query('n.name_embedding', 'node.name_embedding', driver.provider)
        + """ AS score
        WHERE score > $min_score
        WITH node, collect(n)[..$limit] AS top_vector_nodes, collect(n.uuid) AS vector_node_uuids
        """
        + get_nodes_query(driver.provider, 'node_name_and_summary', 'node.fulltext_query')
        + """
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
            'fulltext_query': fulltext_query(node.name, [node.group_id], driver.fulltext_syntax),
        }
        for node in nodes
    ]

    results, _, _ = await driver.execute_query(
        query,
        nodes=query_nodes,
        group_id=group_id,
        limit=limit,
        min_score=min_score,
        routing_='r',
        **query_params,
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
    driver: GraphDriver,
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

    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            RUNTIME_QUERY
            + """
            UNWIND $edges AS edge
            MATCH (n:Entity {uuid: edge.source_node_uuid})-[e:RELATES_TO {group_id: edge.group_id}]-(m:Entity {uuid: edge.target_node_uuid})
            """
            + filter_query
            + """
            WITH e, edge
            RETURN DISTINCT id(e) as id, e.fact_embedding as source_embedding, edge.uuid as search_edge_uuid, 
            edge.fact_embedding as target_embedding
            """
        )
        resp, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
        )

        # Calculate Cosine similarity then return the edge ids
        input_ids = []
        for r in resp:
            score = calculate_cosine_similarity(
                list(map(float, r['source_embedding'].split(','))), r['target_embedding']
            )
            if score > min_score:
                input_ids.append({'id': r['id'], 'score': score, 'uuid': r['search_edge_uuid']})

        # Match the edge ides and return the values
        query = """                      
        UNWIND $ids AS edge
        MATCH ()-[e]->()
        WHERE id(e) = edge.id
        WITH edge, e
        ORDER BY edge.score DESC
        RETURN edge.uuid AS search_edge_uuid,
            collect({
                uuid: e.uuid,
                source_node_uuid: startNode(e).uuid,
                target_node_uuid: endNode(e).uuid,
                created_at: e.created_at,
                name: e.name,
                group_id: e.group_id,
                fact: e.fact,
                fact_embedding: [x IN split(e.fact_embedding, ",") | toFloat(x)],
                episodes: split(e.episodes, ","),
                expired_at: e.expired_at,
                valid_at: e.valid_at,
                invalid_at: e.invalid_at,
                attributes: properties(e)
            })[..$limit] AS matches
                """

        results, _, _ = await driver.execute_query(
            query,
            params=query_params,
            ids=input_ids,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
        )
    else:
        query = (
            RUNTIME_QUERY
            + """
            UNWIND $edges AS edge
            MATCH (n:Entity {uuid: edge.source_node_uuid})-[e:RELATES_TO {group_id: edge.group_id}]-(m:Entity {uuid: edge.target_node_uuid})
            """
            + filter_query
            + """
            WITH e, edge, """
            + get_vector_cosine_func_query(
                'e.fact_embedding', 'edge.fact_embedding', driver.provider
            )
            + """ AS score
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
                    invalid_at: e.invalid_at,
                    attributes: properties(e)
                })[..$limit] AS matches
            """
        )

        results, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
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
    driver: GraphDriver,
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

    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            RUNTIME_QUERY
            + """
            UNWIND $edges AS edge
            MATCH (n:Entity)-[e:RELATES_TO {group_id: edge.group_id}]->(m:Entity)
            WHERE n.uuid IN [edge.source_node_uuid, edge.target_node_uuid] OR m.uuid IN [edge.target_node_uuid, edge.source_node_uuid]
            """
            + filter_query
            + """
            WITH e, edge
            RETURN DISTINCT id(e) as id, e.fact_embedding as source_embedding,
            edge.fact_embedding as target_embedding,
            edge.uuid as search_edge_uuid
            """
        )
        resp, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
        )

        # Calculate Cosine similarity then return the edge ids
        input_ids = []
        for r in resp:
            score = calculate_cosine_similarity(
                list(map(float, r['source_embedding'].split(','))), r['target_embedding']
            )
            if score > min_score:
                input_ids.append({'id': r['id'], 'score': score, 'uuid': r['search_edge_uuid']})

        # Match the edge ides and return the values
        query = """                      
        UNWIND $ids AS edge
        MATCH ()-[e]->()
        WHERE id(e) = edge.id
        WITH edge, e
        ORDER BY edge.score DESC
        RETURN edge.uuid AS search_edge_uuid,
            collect({
                uuid: e.uuid,
                source_node_uuid: startNode(e).uuid,
                target_node_uuid: endNode(e).uuid,
                created_at: e.created_at,
                name: e.name,
                group_id: e.group_id,
                fact: e.fact,
                fact_embedding: [x IN split(e.fact_embedding, ",") | toFloat(x)],
                episodes: split(e.episodes, ","),
                expired_at: e.expired_at,
                valid_at: e.valid_at,
                invalid_at: e.invalid_at,
                attributes: properties(e)
            })[..$limit] AS matches
                """
        results, _, _ = await driver.execute_query(
            query,
            ids=input_ids,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
        )
    else:
        query = (
            RUNTIME_QUERY
            + """
            UNWIND $edges AS edge
            MATCH (n:Entity)-[e:RELATES_TO {group_id: edge.group_id}]->(m:Entity)
            WHERE n.uuid IN [edge.source_node_uuid, edge.target_node_uuid] OR m.uuid IN [edge.target_node_uuid, edge.source_node_uuid]
            """
            + filter_query
            + """
            WITH edge, e, """
            + get_vector_cosine_func_query(
                'e.fact_embedding', 'edge.fact_embedding', driver.provider
            )
            + """ AS score
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
                    invalid_at: e.invalid_at,
                    attributes: properties(e)
                })[..$limit] AS matches
            """
        )

        results, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
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
def rrf(
    results: list[list[str]], rank_const=1, min_score: float = 0
) -> tuple[list[str], list[float]]:
    scores: dict[str, float] = defaultdict(float)
    for result in results:
        for i, uuid in enumerate(result):
            scores[uuid] += 1 / (i + rank_const)

    scored_uuids = [term for term in scores.items()]
    scored_uuids.sort(reverse=True, key=lambda term: term[1])

    sorted_uuids = [term[0] for term in scored_uuids]

    return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score], [
        scores[uuid] for uuid in sorted_uuids if scores[uuid] >= min_score
    ]


async def node_distance_reranker(
    driver: GraphDriver,
    node_uuids: list[str],
    center_node_uuid: str,
    min_score: float = 0,
) -> tuple[list[str], list[float]]:
    # filter out node_uuid center node node uuid
    filtered_uuids = list(filter(lambda node_uuid: node_uuid != center_node_uuid, node_uuids))
    scores: dict[str, float] = {center_node_uuid: 0.0}

    # Find the shortest path to center node
    results, header, _ = await driver.execute_query(
        """
        UNWIND $node_uuids AS node_uuid
        MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]-(n:Entity {uuid: node_uuid})
        RETURN 1 AS score, node_uuid AS uuid
        """,
        node_uuids=filtered_uuids,
        center_uuid=center_node_uuid,
        routing_='r',
    )
    if driver.provider == GraphProvider.FALKORDB:
        results = [dict(zip(header, row, strict=True)) for row in results]

    for result in results:
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

    return [uuid for uuid in filtered_uuids if (1 / scores[uuid]) >= min_score], [
        1 / scores[uuid] for uuid in filtered_uuids if (1 / scores[uuid]) >= min_score
    ]


async def episode_mentions_reranker(
    driver: GraphDriver, node_uuids: list[list[str]], min_score: float = 0
) -> tuple[list[str], list[float]]:
    # use rrf as a preliminary ranker
    sorted_uuids, _ = rrf(node_uuids)
    scores: dict[str, float] = {}

    # Find the shortest path to center node
    results, _, _ = await driver.execute_query(
        """
        UNWIND $node_uuids AS node_uuid
        MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid: node_uuid})
        RETURN count(*) AS score, n.uuid AS uuid
        """,
        node_uuids=sorted_uuids,
        routing_='r',
    )

    for result in results:
        scores[result['uuid']] = result['score']

    # rerank on shortest distance
    sorted_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

    return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score], [
        scores[uuid] for uuid in sorted_uuids if scores[uuid] >= min_score
    ]


def maximal_marginal_relevance(
    query_vector: list[float],
    candidates: dict[str, list[float]],
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    min_score: float = -2.0,
) -> tuple[list[str], list[float]]:
    start = time()
    query_array = np.array(query_vector)
    candidate_arrays: dict[str, NDArray] = {}
    for uuid, embedding in candidates.items():
        candidate_arrays[uuid] = normalize_l2(embedding)

    uuids: list[str] = list(candidate_arrays.keys())

    similarity_matrix = np.zeros((len(uuids), len(uuids)))

    for i, uuid_1 in enumerate(uuids):
        for j, uuid_2 in enumerate(uuids[:i]):
            u = candidate_arrays[uuid_1]
            v = candidate_arrays[uuid_2]
            similarity = np.dot(u, v)

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    mmr_scores: dict[str, float] = {}
    for i, uuid in enumerate(uuids):
        max_sim = np.max(similarity_matrix[i, :])
        mmr = mmr_lambda * np.dot(query_array, candidate_arrays[uuid]) + (mmr_lambda - 1) * max_sim
        mmr_scores[uuid] = mmr

    uuids.sort(reverse=True, key=lambda c: mmr_scores[c])

    end = time()
    logger.debug(f'Completed MMR reranking in {(end - start) * 1000} ms')

    return [uuid for uuid in uuids if mmr_scores[uuid] >= min_score], [
        mmr_scores[uuid] for uuid in uuids if mmr_scores[uuid] >= min_score
    ]


async def get_embeddings_for_nodes(
    driver: GraphDriver, nodes: list[EntityNode]
) -> dict[str, list[float]]:
    if driver.provider == GraphProvider.NEPTUNE:
        query = """
        MATCH (n:Entity)
        WHERE n.uuid IN $node_uuids
        RETURN DISTINCT
            n.uuid AS uuid,
            split(n.name_embedding, ",") AS name_embedding
        """
    else:
        query = """
        MATCH (n:Entity)
        WHERE n.uuid IN $node_uuids
        RETURN DISTINCT
            n.uuid AS uuid,
            n.name_embedding AS name_embedding
        """
    results, _, _ = await driver.execute_query(
        query,
        node_uuids=[node.uuid for node in nodes],
        routing_='r',
    )

    embeddings_dict: dict[str, list[float]] = {}
    for result in results:
        uuid: str = result.get('uuid')
        embedding: list[float] = result.get('name_embedding')
        if uuid is not None and embedding is not None:
            embeddings_dict[uuid] = embedding

    return embeddings_dict


async def get_embeddings_for_communities(
    driver: GraphDriver, communities: list[CommunityNode]
) -> dict[str, list[float]]:
    if driver.provider == GraphProvider.NEPTUNE:
        query = """
        MATCH (c:Community)
        WHERE c.uuid IN $community_uuids
        RETURN DISTINCT
            c.uuid AS uuid,
            split(c.name_embedding, ",") AS name_embedding
        """
    else:
        query = """
        MATCH (c:Community)
        WHERE c.uuid IN $community_uuids
        RETURN DISTINCT
            c.uuid AS uuid,
            c.name_embedding AS name_embedding
        """
    results, _, _ = await driver.execute_query(
        query,
        community_uuids=[community.uuid for community in communities],
        routing_='r',
    )

    embeddings_dict: dict[str, list[float]] = {}
    for result in results:
        uuid: str = result.get('uuid')
        embedding: list[float] = result.get('name_embedding')
        if uuid is not None and embedding is not None:
            embeddings_dict[uuid] = embedding

    return embeddings_dict


async def get_embeddings_for_edges(
    driver: GraphDriver, edges: list[EntityEdge]
) -> dict[str, list[float]]:
    if driver.provider == GraphProvider.NEPTUNE:
        query = """
        MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
        WHERE e.uuid IN $edge_uuids
        RETURN DISTINCT
            e.uuid AS uuid,
            split(e.fact_embedding, ",") AS fact_embedding
        """
    else:
        query = """
        MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
        WHERE e.uuid IN $edge_uuids
        RETURN DISTINCT
            e.uuid AS uuid,
            e.fact_embedding AS fact_embedding
        """
    results, _, _ = await driver.execute_query(
        query,
        edge_uuids=[edge.uuid for edge in edges],
        routing_='r',
    )

    embeddings_dict: dict[str, list[float]] = {}
    for result in results:
        uuid: str = result.get('uuid')
        embedding: list[float] = result.get('fact_embedding')
        if uuid is not None and embedding is not None:
            embeddings_dict[uuid] = embedding

    return embeddings_dict
