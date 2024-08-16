import logging

from neo4j import AsyncDriver

from core.edges import EpisodicEdge, EntityEdge, Edge
from core.llm_client.config import EMBEDDING_DIM
from core.nodes import EntityNode, EpisodicNode, Node

logger = logging.getLogger(__name__)


async def bfs(node_ids: list[str], driver: AsyncDriver):
    records, _, _ = driver.execute_query(
        """
        MATCH (n WHERE n.uuid in $node_ids)-[r]->(m:Entity)
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
        if n_uuid in context:
            context[n_uuid]["facts"].append(record["fact"])
        else:
            context[n_uuid] = {
                "name": record["source_name"],
                "summary": record["source_summary"],
                "facts": [record["fact"]],
            }

        m_uuid = record["target_node_uuid"]
        if m_uuid not in context:
            context[n_uuid] = {
                "name": record["target_name"],
                "summary": record["target_summary"],
                "facts": [],
            }

    return context


async def similarity_search(
    query: str, driver: AsyncDriver, embedder, model="text-embedding-3-small"
) -> list[EntityEdge]:
    # vector similarity search over embedded facts
    text = query.replace("\n", " ")
    search_vector = (
        embedder.create(input=[text], model=model).data[0].embedding[:EMBEDDING_DIM]
    )

    records, _, _ = driver.execute_query(
        """
                MATCH (n)-[r]->(m)
                WHERE r.fact_embedding IS NOT NULL
                WITH r, vector.similarity.cosine(r.fact_embedding, $search_vector) AS score
                WHERE score > 0.8
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

    logger.info(f"fulltext search results. QUERY:{query}. RESULT: {edges}")

    return edges


async def fulltext_search(query: str, driver: AsyncDriver) -> list[EntityNode]:
    # BM25 search to get top nodes
    fuzzy_query = query + "~"
    records, _, _ = await driver.execute_query(
        """
    CALL db.index.fulltext.queryNodes("name_and_summary", $query) YIELD node, score
    RETURN 
        node.uuid As uuid, 
        node.name AS name, 
        node.labels AS labels, 
        node.created_at AS created_at, 
        node.summary AS summary
    ORDER BY score DESC
    LIMIT 10
    """,
        query=fuzzy_query,
    )
    nodes: list[EntityNode] = []

    for record in records:
        nodes.append(EntityNode(**record))

    logger.info(f"fulltext search results. QUERY:{query}. RESULT: {nodes}")

    return nodes
