"""Backfill utility to populate a vector store from an existing graph database."""

import logging
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.vector_store.client import VectorStoreClient
from graphiti_core.vector_store.milvus_utils import (
    COLLECTION_COMMUNITY_NODES,
    COLLECTION_ENTITY_EDGES,
    COLLECTION_ENTITY_NODES,
    COLLECTION_EPISODIC_NODES,
    community_node_to_milvus_dict,
    entity_edge_to_milvus_dict,
    entity_node_to_milvus_dict,
    episodic_node_to_milvus_dict,
)

logger = logging.getLogger(__name__)


async def backfill_vector_store(
    driver: GraphDriver,
    vector_store: VectorStoreClient,
    group_ids: list[str] | None = None,
    batch_size: int = 100,
) -> dict[str, int]:
    """Backfill a vector store from an existing graph database.

    Reads all entity nodes, entity edges, episodic nodes, and community nodes
    from the graph DB and upserts them into the vector store.

    Parameters
    ----------
    driver : GraphDriver
        The graph database driver to read from.
    vector_store : VectorStoreClient
        The vector store client to write to.
    group_ids : list[str] | None
        Optional list of group IDs to filter by. If None, syncs all data.
    batch_size : int
        Number of records to process per batch.

    Returns
    -------
    dict[str, int]
        Counts of synced records per collection type.
    """
    await vector_store.ensure_ready()
    counts: dict[str, int] = {
        'entity_nodes': 0,
        'entity_edges': 0,
        'episodic_nodes': 0,
        'community_nodes': 0,
    }

    group_filter = ''
    params: dict[str, Any] = {}
    if group_ids is not None:
        group_filter = 'WHERE n.group_id IN $group_ids'
        params['group_ids'] = group_ids

    # Sync entity nodes
    counts['entity_nodes'] = await _sync_entity_nodes(
        driver, vector_store, group_filter, params, batch_size
    )

    # Sync entity edges
    edge_group_filter = group_filter.replace('n.group_id', 'r.group_id')
    counts['entity_edges'] = await _sync_entity_edges(
        driver, vector_store, edge_group_filter, params, batch_size
    )

    # Sync episodic nodes
    counts['episodic_nodes'] = await _sync_episodic_nodes(
        driver, vector_store, group_filter, params, batch_size
    )

    # Sync community nodes
    counts['community_nodes'] = await _sync_community_nodes(
        driver, vector_store, group_filter, params, batch_size
    )

    logger.info(f'Backfill complete: {counts}')
    return counts


async def _sync_entity_nodes(
    driver: GraphDriver,
    vector_store: VectorStoreClient,
    group_filter: str,
    params: dict[str, Any],
    batch_size: int,
) -> int:
    """Sync entity nodes from graph DB to vector store."""
    from graphiti_core.nodes import get_entity_node_from_record

    records, _, _ = await driver.execute_query(
        f"""
        MATCH (n:Entity)
        {group_filter}
        RETURN
            n.uuid AS uuid,
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at,
            n.summary AS summary,
            n.name_embedding AS name_embedding,
            labels(n) AS labels,
            properties(n) AS attributes
        """,
        **params,
        routing_='r',
    )

    count = 0
    col = vector_store.collection_name(COLLECTION_ENTITY_NODES)
    batch: list[dict[str, Any]] = []

    for record in records:
        node = get_entity_node_from_record(record, driver.provider)
        embedding = record.get('name_embedding')
        if embedding is not None:
            node.name_embedding = embedding
        else:
            logger.debug(f'Skipping entity node {node.uuid}: no embedding')
            continue

        batch.append(entity_node_to_milvus_dict(node))
        if len(batch) >= batch_size:
            await vector_store.upsert(collection_name=col, data=batch)
            count += len(batch)
            batch = []

    if batch:
        await vector_store.upsert(collection_name=col, data=batch)
        count += len(batch)

    logger.info(f'Synced {count} entity nodes')
    return count


async def _sync_entity_edges(
    driver: GraphDriver,
    vector_store: VectorStoreClient,
    group_filter: str,
    params: dict[str, Any],
    batch_size: int,
) -> int:
    """Sync entity edges from graph DB to vector store."""
    from graphiti_core.edges import EntityEdge

    records, _, _ = await driver.execute_query(
        f"""
        MATCH (src)-[r:RELATES_TO]->(tgt)
        {group_filter}
        RETURN
            r.uuid AS uuid,
            r.group_id AS group_id,
            src.uuid AS source_node_uuid,
            tgt.uuid AS target_node_uuid,
            r.name AS name,
            r.fact AS fact,
            r.fact_embedding AS fact_embedding,
            r.episodes AS episodes,
            r.created_at AS created_at,
            r.expired_at AS expired_at,
            r.valid_at AS valid_at,
            r.invalid_at AS invalid_at
        """,
        **params,
        routing_='r',
    )

    count = 0
    col = vector_store.collection_name(COLLECTION_ENTITY_EDGES)
    batch: list[dict[str, Any]] = []

    for record in records:
        embedding = record.get('fact_embedding')
        if embedding is None:
            logger.debug(f'Skipping edge {record.get("uuid")}: no embedding')
            continue

        edge = EntityEdge(
            uuid=record['uuid'],
            group_id=record['group_id'],
            source_node_uuid=record['source_node_uuid'],
            target_node_uuid=record['target_node_uuid'],
            name=record.get('name', ''),
            fact=record.get('fact', ''),
            fact_embedding=embedding,
            episodes=record.get('episodes') or [],
            created_at=record['created_at'],
            expired_at=record.get('expired_at'),
            valid_at=record.get('valid_at'),
            invalid_at=record.get('invalid_at'),
        )

        batch.append(entity_edge_to_milvus_dict(edge))
        if len(batch) >= batch_size:
            await vector_store.upsert(collection_name=col, data=batch)
            count += len(batch)
            batch = []

    if batch:
        await vector_store.upsert(collection_name=col, data=batch)
        count += len(batch)

    logger.info(f'Synced {count} entity edges')
    return count


async def _sync_episodic_nodes(
    driver: GraphDriver,
    vector_store: VectorStoreClient,
    group_filter: str,
    params: dict[str, Any],
    batch_size: int,
) -> int:
    """Sync episodic nodes from graph DB to vector store."""
    from graphiti_core.nodes import EpisodicNode

    records, _, _ = await driver.execute_query(
        f"""
        MATCH (n:Episodic)
        {group_filter}
        RETURN
            n.uuid AS uuid,
            n.group_id AS group_id,
            n.name AS name,
            n.content AS content,
            n.source AS source,
            n.source_description AS source_description,
            n.created_at AS created_at,
            n.valid_at AS valid_at,
            n.entity_edges AS entity_edges
        """,
        **params,
        routing_='r',
    )

    count = 0
    col = vector_store.collection_name(COLLECTION_EPISODIC_NODES)
    batch: list[dict[str, Any]] = []

    for record in records:
        node = EpisodicNode(
            uuid=record['uuid'],
            group_id=record['group_id'],
            name=record.get('name', ''),
            content=record.get('content', ''),
            source=record.get('source', 'text'),
            source_description=record.get('source_description', ''),
            created_at=record['created_at'],
            valid_at=record.get('valid_at') or record['created_at'],
            entity_edges=record.get('entity_edges') or [],
        )

        batch.append(episodic_node_to_milvus_dict(node))
        if len(batch) >= batch_size:
            await vector_store.upsert(collection_name=col, data=batch)
            count += len(batch)
            batch = []

    if batch:
        await vector_store.upsert(collection_name=col, data=batch)
        count += len(batch)

    logger.info(f'Synced {count} episodic nodes')
    return count


async def _sync_community_nodes(
    driver: GraphDriver,
    vector_store: VectorStoreClient,
    group_filter: str,
    params: dict[str, Any],
    batch_size: int,
) -> int:
    """Sync community nodes from graph DB to vector store."""
    from graphiti_core.nodes import CommunityNode

    records, _, _ = await driver.execute_query(
        f"""
        MATCH (n:Community)
        {group_filter}
        RETURN
            n.uuid AS uuid,
            n.group_id AS group_id,
            n.name AS name,
            n.summary AS summary,
            n.created_at AS created_at,
            n.name_embedding AS name_embedding
        """,
        **params,
        routing_='r',
    )

    count = 0
    col = vector_store.collection_name(COLLECTION_COMMUNITY_NODES)
    batch: list[dict[str, Any]] = []

    for record in records:
        embedding = record.get('name_embedding')
        if embedding is None:
            logger.debug(f'Skipping community node {record.get("uuid")}: no embedding')
            continue

        node = CommunityNode(
            uuid=record['uuid'],
            group_id=record['group_id'],
            name=record.get('name', ''),
            summary=record.get('summary', ''),
            created_at=record['created_at'],
            name_embedding=embedding,
        )

        batch.append(community_node_to_milvus_dict(node))
        if len(batch) >= batch_size:
            await vector_store.upsert(collection_name=col, data=batch)
            count += len(batch)
            batch = []

    if batch:
        await vector_store.upsert(collection_name=col, data=batch)
        count += len(batch)

    logger.info(f'Synced {count} community nodes')
    return count
