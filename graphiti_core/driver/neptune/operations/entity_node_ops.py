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
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.record_parsers import entity_node_from_record
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.models.nodes.node_db_queries import (
    get_entity_node_return_query,
    get_entity_node_save_bulk_query,
    get_entity_node_save_query,
)
from graphiti_core.nodes import EntityNode

logger = logging.getLogger(__name__)


class NeptuneEntityNodeOperations(EntityNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        entity_data: dict[str, Any] = {
            'uuid': node.uuid,
            'name': node.name,
            'name_embedding': node.name_embedding,
            'group_id': node.group_id,
            'summary': node.summary,
            'created_at': node.created_at,
        }
        entity_data.update(node.attributes or {})
        labels = ':'.join(list(set(node.labels + ['Entity'])))

        query = get_entity_node_save_query(GraphProvider.NEPTUNE, labels)

        if tx is not None:
            await tx.run(query, entity_data=entity_data)
        else:
            await executor.execute_query(query, entity_data=entity_data)

        logger.debug(f'Saved Node to Graph: {node.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        prepared: list[dict[str, Any]] = []
        for node in nodes:
            entity_data: dict[str, Any] = {
                'uuid': node.uuid,
                'name': node.name,
                'group_id': node.group_id,
                'summary': node.summary,
                'created_at': node.created_at,
                'name_embedding': node.name_embedding,
                'labels': list(set(node.labels + ['Entity'])),
            }
            entity_data.update(node.attributes or {})
            prepared.append(entity_data)

        queries = get_entity_node_save_bulk_query(GraphProvider.NEPTUNE, prepared)

        for query in queries:
            if tx is not None:
                await tx.run(query, nodes=prepared)
            else:
                await executor.execute_query(query, nodes=prepared)

    async def delete(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (n {uuid: $uuid})
            WHERE n:Entity OR n:Episodic OR n:Community
            DETACH DELETE n
        """
        if tx is not None:
            await tx.run(query, uuid=node.uuid)
        else:
            await executor.execute_query(query, uuid=node.uuid)

        logger.debug(f'Deleted Node: {node.uuid}')

    async def delete_by_group_id(
        self,
        executor: QueryExecutor,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        query = """
            MATCH (n:Entity {group_id: $group_id})
            DETACH DELETE n
        """
        if tx is not None:
            await tx.run(query, group_id=group_id)
        else:
            await executor.execute_query(query, group_id=group_id)

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            DETACH DELETE n
        """
        if tx is not None:
            await tx.run(query, uuids=uuids)
        else:
            await executor.execute_query(query, uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> EntityNode:
        query = """
            MATCH (n:Entity {uuid: $uuid})
            RETURN
            """ + get_entity_node_return_query(GraphProvider.NEPTUNE)
        records, _, _ = await executor.execute_query(query, uuid=uuid)
        nodes = [entity_node_from_record(r) for r in records]
        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)
        return nodes[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EntityNode]:
        query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN
            """ + get_entity_node_return_query(GraphProvider.NEPTUNE)
        records, _, _ = await executor.execute_query(query, uuids=uuids)
        return [entity_node_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityNode]:
        cursor_clause = 'AND n.uuid < $uuid' if uuid_cursor else ''
        limit_clause = 'LIMIT $limit' if limit is not None else ''
        query = (
            """
            MATCH (n:Entity)
            WHERE n.group_id IN $group_ids
            """
            + cursor_clause
            + """
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.NEPTUNE)
            + """
            ORDER BY n.uuid DESC
            """
            + limit_clause
        )
        records, _, _ = await executor.execute_query(
            query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
        )
        return [entity_node_from_record(r) for r in records]

    async def load_embeddings(
        self,
        executor: QueryExecutor,
        node: EntityNode,
    ) -> None:
        query = """
            MATCH (n:Entity {uuid: $uuid})
            RETURN [x IN split(n.name_embedding, ",") | toFloat(x)] AS name_embedding
        """
        records, _, _ = await executor.execute_query(query, uuid=node.uuid)
        if len(records) == 0:
            raise NodeNotFoundError(node.uuid)
        node.name_embedding = records[0]['name_embedding']

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        batch_size: int = 100,
    ) -> None:
        uuids = [n.uuid for n in nodes]
        query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN DISTINCT n.uuid AS uuid, [x IN split(n.name_embedding, ",") | toFloat(x)] AS name_embedding
        """
        records, _, _ = await executor.execute_query(query, uuids=uuids)
        embedding_map = {r['uuid']: r['name_embedding'] for r in records}
        for node in nodes:
            if node.uuid in embedding_map:
                node.name_embedding = embedding_map[node.uuid]
