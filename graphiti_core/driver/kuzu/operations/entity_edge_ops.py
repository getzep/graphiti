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

import json
import logging
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.kuzu.operations.record_parsers import parse_kuzu_entity_edge
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_return_query,
    get_entity_edge_save_query,
)

logger = logging.getLogger(__name__)


class KuzuEntityEdgeOperations(EntityEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        params: dict[str, Any] = {
            'uuid': edge.uuid,
            'source_uuid': edge.source_node_uuid,
            'target_uuid': edge.target_node_uuid,
            'name': edge.name,
            'fact': edge.fact,
            'fact_embedding': edge.fact_embedding,
            'group_id': edge.group_id,
            'episodes': edge.episodes,
            'created_at': edge.created_at,
            'expired_at': edge.expired_at,
            'valid_at': edge.valid_at,
            'invalid_at': edge.invalid_at,
            'attributes': json.dumps(edge.attributes or {}),
        }

        query = get_entity_edge_save_query(GraphProvider.KUZU)
        if tx is not None:
            await tx.run(query, **params)
        else:
            await executor.execute_query(query, **params)

        logger.debug(f'Saved Edge to Graph: {edge.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        # Kuzu doesn't support UNWIND - iterate and save individually
        for edge in edges:
            await self.save(executor, edge, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(m:Entity)
            DETACH DELETE e
        """
        if tx is not None:
            await tx.run(query, uuid=edge.uuid)
        else:
            await executor.execute_query(query, uuid=edge.uuid)

        logger.debug(f'Deleted Edge: {edge.uuid}')

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
            WHERE e.uuid IN $uuids
            DETACH DELETE e
        """
        if tx is not None:
            await tx.run(query, uuids=uuids)
        else:
            await executor.execute_query(query, uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> EntityEdge:
        query = """
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(m:Entity)
            RETURN
            """ + get_entity_edge_return_query(GraphProvider.KUZU)
        records, _, _ = await executor.execute_query(query, uuid=uuid)
        edges = [parse_kuzu_entity_edge(r) for r in records]
        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EntityEdge]:
        if not uuids:
            return []
        query = """
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
            WHERE e.uuid IN $uuids
            RETURN
            """ + get_entity_edge_return_query(GraphProvider.KUZU)
        records, _, _ = await executor.execute_query(query, uuids=uuids)
        return [parse_kuzu_entity_edge(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityEdge]:
        cursor_clause = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_clause = 'LIMIT $limit' if limit is not None else ''
        query = (
            """
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
            WHERE e.group_id IN $group_ids
            """
            + cursor_clause
            + """
            RETURN
            """
            + get_entity_edge_return_query(GraphProvider.KUZU)
            + """
            ORDER BY e.uuid DESC
            """
            + limit_clause
        )
        records, _, _ = await executor.execute_query(
            query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
        )
        return [parse_kuzu_entity_edge(r) for r in records]

    async def get_between_nodes(
        self,
        executor: QueryExecutor,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[EntityEdge]:
        query = """
            MATCH (n:Entity {uuid: $source_node_uuid})-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity {uuid: $target_node_uuid})
            RETURN
            """ + get_entity_edge_return_query(GraphProvider.KUZU)
        records, _, _ = await executor.execute_query(
            query,
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
        )
        return [parse_kuzu_entity_edge(r) for r in records]

    async def get_by_node_uuid(
        self,
        executor: QueryExecutor,
        node_uuid: str,
    ) -> list[EntityEdge]:
        query = """
            MATCH (n:Entity {uuid: $node_uuid})-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
            RETURN
            """ + get_entity_edge_return_query(GraphProvider.KUZU)
        records, _, _ = await executor.execute_query(query, node_uuid=node_uuid)
        return [parse_kuzu_entity_edge(r) for r in records]

    async def load_embeddings(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
    ) -> None:
        query = """
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(m:Entity)
            RETURN e.fact_embedding AS fact_embedding
        """
        records, _, _ = await executor.execute_query(query, uuid=edge.uuid)
        if len(records) == 0:
            raise EdgeNotFoundError(edge.uuid)
        edge.fact_embedding = records[0]['fact_embedding']

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        batch_size: int = 100,
    ) -> None:
        uuids = [e.uuid for e in edges]
        query = """
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
            WHERE e.uuid IN $edge_uuids
            RETURN DISTINCT e.uuid AS uuid, e.fact_embedding AS fact_embedding
        """
        records, _, _ = await executor.execute_query(query, edge_uuids=uuids)
        embedding_map = {r['uuid']: r['fact_embedding'] for r in records}
        for edge in edges:
            if edge.uuid in embedding_map:
                edge.fact_embedding = embedding_map[edge.uuid]
