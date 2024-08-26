import asyncio
import logging
import re
import typing
from collections import defaultdict
from datetime import datetime
from time import time

from neo4j import AsyncDriver
from neo4j import time as neo4j_time

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode

logger = logging.getLogger(__name__)

RELEVANT_SCHEMA_LIMIT = 3


def parse_db_date(neo_date: neo4j_time.DateTime | None) -> datetime | None:
    return neo_date.to_native() if neo_date else None


async def get_mentioned_nodes(driver: AsyncDriver, episodes: list[EpisodicNode]):
    episode_uuids = [episode.uuid for episode in episodes]
    records, _, _ = await driver.execute_query(
        """
        MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity) WHERE episode.uuid IN $uuids
        RETURN DISTINCT
            n.uuid As uuid, 
            n.name AS name, 
            n.created_at AS created_at, 
            n.summary AS summary
        """,
        uuids=episode_uuids,
    )

    nodes: list[EntityNode] = []

    for record in records:
        nodes.append(
            EntityNode(
                uuid=record['uuid'],
                name=record['name'],
                labels=['Entity'],
                created_at=record['created_at'].to_native(),
                summary=record['summary'],
            )
        )

    return nodes


async def bfs(node_ids: list[str], driver: AsyncDriver):
    records, _, _ = await driver.execute_query(
        """
        MATCH (n WHERE n.uuid in $node_ids)-[r]->(m)
        RETURN DISTINCT
            n.uuid AS source_node_uuid,
            n.name AS source_name, 
            n.summary AS source_summary,
            m.uuid AS target_node_uuid,
            m.name AS target_name, 
            m.summary AS target_summary,
            r.uuid AS uuid,
            r.created_at AS created_at,
            r.name AS name,
            r.fact AS fact,
            r.fact_embedding AS fact_embedding,
            r.episodes AS episodes,
            r.expired_at AS expired_at,
            r.valid_at AS valid_at,
            r.invalid_at AS invalid_at
        
    """,
        node_ids=node_ids,
    )

    context: dict[str, typing.Any] = {}

    for record in records:
        n_uuid = record['source_node_uuid']
        if n_uuid in context:
            context[n_uuid]['facts'].append(record['fact'])
        else:
            context[n_uuid] = {
                'name': record['source_name'],
                'summary': record['source_summary'],
                'facts': [record['fact']],
            }

        m_uuid = record['target_node_uuid']
        if m_uuid not in context:
            context[m_uuid] = {
                'name': record['target_name'],
                'summary': record['target_summary'],
                'facts': [],
            }
    logger.info(f'bfs search returned context: {context}')
    return context


async def edge_similarity_search(
    search_vector: list[float], driver: AsyncDriver, limit=RELEVANT_SCHEMA_LIMIT
) -> list[EntityEdge]:
    # vector similarity search over embedded facts
    records, _, _ = await driver.execute_query(
        """
                CALL db.index.vector.queryRelationships("fact_embedding", 5, $search_vector)
                YIELD relationship AS r, score
                MATCH (n)-[r:RELATES_TO]->(m)
                RETURN
                    r.uuid AS uuid,
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
                """,
        search_vector=search_vector,
        limit=limit,
    )

    edges: list[EntityEdge] = []

    for record in records:
        edge = EntityEdge(
            uuid=record['uuid'],
            source_node_uuid=record['source_node_uuid'],
            target_node_uuid=record['target_node_uuid'],
            fact=record['fact'],
            name=record['name'],
            episodes=record['episodes'],
            fact_embedding=record['fact_embedding'],
            created_at=record['created_at'].to_native(),
            expired_at=parse_db_date(record['expired_at']),
            valid_at=parse_db_date(record['valid_at']),
            invalid_at=parse_db_date(record['invalid_at']),
        )

        edges.append(edge)

    return edges


async def entity_similarity_search(
    search_vector: list[float], driver: AsyncDriver, limit=RELEVANT_SCHEMA_LIMIT
) -> list[EntityNode]:
    # vector similarity search over entity names
    records, _, _ = await driver.execute_query(
        """
                CALL db.index.vector.queryNodes("name_embedding", $limit, $search_vector)
                YIELD node AS n, score
                RETURN
                    n.uuid As uuid, 
                    n.name AS name, 
                    n.created_at AS created_at, 
                    n.summary AS summary
                ORDER BY score DESC
                """,
        search_vector=search_vector,
        limit=limit,
    )
    nodes: list[EntityNode] = []

    for record in records:
        nodes.append(
            EntityNode(
                uuid=record['uuid'],
                name=record['name'],
                labels=['Entity'],
                created_at=record['created_at'].to_native(),
                summary=record['summary'],
            )
        )

    return nodes


async def entity_fulltext_search(
    query: str, driver: AsyncDriver, limit=RELEVANT_SCHEMA_LIMIT
) -> list[EntityNode]:
    # BM25 search to get top nodes
    fuzzy_query = re.sub(r'[^\w\s]', '', query) + '~'
    records, _, _ = await driver.execute_query(
        """
    CALL db.index.fulltext.queryNodes("name_and_summary", $query) YIELD node, score
    RETURN
        node.uuid As uuid, 
        node.name AS name, 
        node.created_at AS created_at, 
        node.summary AS summary
    ORDER BY score DESC
    LIMIT $limit
    """,
        query=fuzzy_query,
        limit=limit,
    )
    nodes: list[EntityNode] = []

    for record in records:
        nodes.append(
            EntityNode(
                uuid=record['uuid'],
                name=record['name'],
                labels=['Entity'],
                created_at=record['created_at'].to_native(),
                summary=record['summary'],
            )
        )

    return nodes


async def edge_fulltext_search(
    query: str, driver: AsyncDriver, limit=RELEVANT_SCHEMA_LIMIT
) -> list[EntityEdge]:
    # fulltext search over facts
    fuzzy_query = re.sub(r'[^\w\s]', '', query) + '~'

    records, _, _ = await driver.execute_query(
        """
                CALL db.index.fulltext.queryRelationships("name_and_fact", $query) 
                YIELD relationship AS r, score
                MATCH (n:Entity)-[r]->(m:Entity)
                RETURN 
                    r.uuid AS uuid,
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
                """,
        query=fuzzy_query,
        limit=limit,
    )

    edges: list[EntityEdge] = []

    for record in records:
        edge = EntityEdge(
            uuid=record['uuid'],
            source_node_uuid=record['source_node_uuid'],
            target_node_uuid=record['target_node_uuid'],
            fact=record['fact'],
            name=record['name'],
            episodes=record['episodes'],
            fact_embedding=record['fact_embedding'],
            created_at=record['created_at'].to_native(),
            expired_at=parse_db_date(record['expired_at']),
            valid_at=parse_db_date(record['valid_at']),
            invalid_at=parse_db_date(record['invalid_at']),
        )

        edges.append(edge)

    return edges


async def get_relevant_nodes(
    nodes: list[EntityNode],
    driver: AsyncDriver,
) -> list[EntityNode]:
    start = time()
    relevant_nodes: list[EntityNode] = []
    relevant_node_uuids = set()

    results = await asyncio.gather(
        *[entity_fulltext_search(node.name, driver) for node in nodes],
        *[
            entity_similarity_search(node.name_embedding, driver)
            for node in nodes
            if node.name_embedding is not None
        ],
    )

    for result in results:
        for node in result:
            if node.uuid in relevant_node_uuids:
                continue

            relevant_node_uuids.add(node.uuid)
            relevant_nodes.append(node)

    end = time()
    logger.info(f'Found relevant nodes: {relevant_node_uuids} in {(end - start) * 1000} ms')

    return relevant_nodes


async def get_relevant_edges(
    edges: list[EntityEdge],
    driver: AsyncDriver,
) -> list[EntityEdge]:
    start = time()
    relevant_edges: list[EntityEdge] = []
    relevant_edge_uuids = set()

    results = await asyncio.gather(
        *[
            edge_similarity_search(edge.fact_embedding, driver)
            for edge in edges
            if edge.fact_embedding is not None
        ],
        *[edge_fulltext_search(edge.fact, driver) for edge in edges],
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
    driver: AsyncDriver, results: list[list[str]], center_node_uuid: str
) -> list[str]:
    # use rrf as a preliminary ranker
    sorted_uuids = rrf(results)
    scores: dict[str, float] = {}

    for uuid in sorted_uuids:
        # Find shortest path to center node
        records, _, _ = await driver.execute_query(
            """  
        MATCH (source:Entity)-[r:RELATES_TO {uuid: $edge_uuid}]->(target:Entity)
        MATCH p = SHORTEST 1 (center:Entity)-[:RELATES_TO]-+(n:Entity)
        WHERE center.uuid = $center_uuid AND n.uuid IN [source.uuid, target.uuid]
        RETURN min(length(p)) AS score, source.uuid AS source_uuid, target.uuid AS target_uuid
        """,
            edge_uuid=uuid,
            center_uuid=center_node_uuid,
        )
        distance = 0.01

        for record in records:
            if (
                record['source_uuid'] == center_node_uuid
                or record['target_uuid'] == center_node_uuid
            ):
                continue
            distance = record['score']

        if uuid in scores:
            scores[uuid] = min(1 / distance, scores[uuid])
        else:
            scores[uuid] = 1 / distance

    # rerank on shortest distance
    sorted_uuids.sort(reverse=True, key=lambda cur_uuid: scores[cur_uuid])

    return sorted_uuids
