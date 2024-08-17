import logging
from datetime import datetime

from neo4j import AsyncDriver

from core.edges import EpisodicEdge, EntityEdge, Edge
from core.llm_client.config import EMBEDDING_DIM
from core.nodes import EntityNode, EpisodicNode, Node

logger = logging.getLogger(__name__)


async def bfs(node_ids: list[str], driver: AsyncDriver):
    records, _, _ = await driver.execute_query(
        """
        MATCH (n WHERE n.uuid in $node_ids)-[r]->(m)
        RETURN
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

    context = {}

    for record in records:
        n_uuid = record["source_node_uuid"]
        if n_uuid in context.keys():
            context[n_uuid]["facts"].append(record["fact"])
        else:
            context[n_uuid] = {
                "name": record["source_name"],
                "summary": record["source_summary"],
                "facts": [record["fact"]],
            }

        m_uuid = record["target_node_uuid"]
        if m_uuid not in context:
            context[m_uuid] = {
                "name": record["target_name"],
                "summary": record["target_summary"],
                "facts": [],
            }
    logger.info(f"bfs search returned context: {context}")
    return context


async def edge_similarity_search(
    search_vector: list[float], driver: AsyncDriver
) -> list[EntityEdge]:
    # vector similarity search over embedded facts
    records, _, _ = await driver.execute_query(
        """
                MATCH (n)-[r:RELATES_TO]->(m)
                WHERE r.fact_embedding IS NOT NULL
                WITH n, m, r, vector.similarity.cosine(r.fact_embedding, $search_vector) AS score
                WHERE score > 0.5
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
                ORDER BY score DESC LIMIT 10
                """,
        search_vector=search_vector,
    )

    edges: list[EntityEdge] = []

    now = datetime.now()

    for record in records:
        edge = EntityEdge(
            uuid=record["uuid"],
            source_node_uuid=record["source_node_uuid"],
            target_node_uuid=record["target_node_uuid"],
            fact=record["fact"],
            name=record["name"],
            episodes=record["episodes"],
            fact_embedding=record["fact_embedding"],
            created_at=now,
            expired_at=now,
            valid_at=now,
            invalid_At=now,
        )

        edges.append(edge)

    logger.info(f"similarity search results. RESULT: {[edge.uuid for edge in edges]}")

    return edges


async def entity_similarity_search(
    search_vector: list[float], driver: AsyncDriver
) -> list[EntityNode]:
    # vector similarity search over entity names
    records, _, _ = await driver.execute_query(
        """
                MATCH (n:Entity)
                WHERE n.name_embedding IS NOT NULL
                WITH n, vector.similarity.cosine(n.name_embedding, $search_vector) AS score
                WHERE score > 0.8
                RETURN
                    n.uuid As uuid, 
                    n.name AS name, 
                    n.created_at AS created_at, 
                    n.summary AS summary
                ORDER BY score DESC LIMIT 3
                """,
        search_vector=search_vector,
    )
    nodes: list[EntityNode] = []

    for record in records:
        nodes.append(
            EntityNode(
                uuid=record["uuid"],
                name=record["name"],
                labels=[],
                created_at=datetime.now(),
                summary=record["summary"],
            )
        )

    logger.info(f"name semantic search results. RESULT: {nodes}")

    return nodes


async def entity_fulltext_search(query: str, driver: AsyncDriver) -> list[EntityNode]:
    # BM25 search to get top nodes
    fuzzy_query = query + "~"
    records, _, _ = await driver.execute_query(
        """
    CALL db.index.fulltext.queryNodes("name_and_summary", $query) YIELD node, score
    RETURN 
        node.uuid As uuid, 
        node.name AS name, 
        node.created_at AS created_at, 
        node.summary AS summary
    ORDER BY score DESC
    LIMIT 10
    """,
        query=fuzzy_query,
    )
    nodes: list[EntityNode] = []

    for record in records:
        nodes.append(
            EntityNode(
                uuid=record["uuid"],
                name=record["name"],
                labels=[],
                created_at=datetime.now(),
                summary=record["summary"],
            )
        )

    logger.info(f"fulltext search results. QUERY:{query}. RESULT: {nodes}")

    return nodes


async def edge_fulltext_search(query: str, driver: AsyncDriver) -> list[EntityEdge]:
    # fulltext search over facts
    fuzzy_query = query + "~"

    records, _, _ = await driver.execute_query(
        """
                CALL db.index.fulltext.queryNodes("name_and_fact", $query) YIELD r, score
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
                ORDER BY score DESC LIMIT 10
                """,
        query=fuzzy_query,
    )

    edges: list[EntityEdge] = []

    now = datetime.now()

    for record in records:
        edge = EntityEdge(
            uuid=record["uuid"],
            source_node_uuid=record["source_node_uuid"],
            target_node_uuid=record["target_node_uuid"],
            fact=record["fact"],
            name=record["name"],
            episodes=record["episodes"],
            fact_embedding=record["fact_embedding"],
            created_at=now,
            expired_at=now,
            valid_at=now,
            invalid_At=now,
        )

        edges.append(edge)

    logger.info(
        f"similarity search results. QUERY:{query}. RESULT: {[edge.uuid for edge in edges]}"
    )

    return edges


async def get_relevant_nodes(
    nodes: list[EntityNode],
    driver: AsyncDriver,
) -> list[EntityNode]:
    relevant_nodes = []

    for node in nodes:
        relevant_nodes.extend(await entity_fulltext_search(node.name, driver))
        relevant_nodes.extend(
            await entity_similarity_search(node.name_embedding, driver)
        )
    return relevant_nodes


async def get_relevant_edges(
    edges: list[EntityEdge],
    driver: AsyncDriver,
) -> list[EntityEdge]:
    relevant_edges = []

    for edge in edges:
        relevant_edges.extend(await edge_similarity_search(edge.fact_embedding, driver))
    return relevant_edges
