# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_filters import ComparisonOperator, PropertyFilter, SearchFilters

from ..codecs import encode_json, ensure_vector, entity_node_from_record, entity_node_to_binds
from ..search.filters import build_node_where
from ..sql.nodes import (
    DELETE_GROUP_EDGES,
    DELETE_GROUP_NODES,
    DELETE_NODE,
    DELETE_NODE_EDGES,
    DELETE_NODES_BY_UUIDS,
    GET_ENTITIES_BY_ATTRIBUTES,
    GET_ENTITIES_BY_GROUPS,
    GET_ENTITIES_BY_NAMES,
    GET_ENTITIES_BY_UUIDS,
    GET_ENTITY_BY_UUID,
    GET_ENTITY_EMBEDDING,
    GET_NODE_SCOPE,
    MERGE_ENTITY_NODE,
)
from .base import (
    composite_transaction,
    resolve_group_ids,
    resolve_group_scope,
    run_read,
    run_write,
    validate_request_scope,
    validate_uuid_scope,
)


class OracleEntityNodeOperations(EntityNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        binds = entity_node_to_binds(node)
        if tx is None:
            async with composite_transaction(executor, None) as active_tx:
                await self.save(executor, node, tx=active_tx)
            return
        await validate_uuid_scope(
            executor,
            tx,
            GET_NODE_SCOPE,
            uuid=node.uuid,
            kind='ENTITY',
            group_id=node.group_id,
        )
        await run_write(executor, tx, MERGE_ENTITY_NODE, **binds)

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        if not nodes:
            return
        async with composite_transaction(executor, tx) as active_tx:
            for offset in range(0, len(nodes), batch_size):
                for node in nodes[offset : offset + batch_size]:
                    await self.save(executor, node, tx=active_tx)

    async def delete(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        validate_request_scope(executor, kind='ENTITY', group_id=node.group_id)
        async with composite_transaction(executor, tx) as active_tx:
            await run_write(
                executor,
                active_tx,
                DELETE_NODE_EDGES,
                uuid=node.uuid,
                scope_group_id=node.group_id,
            )
            await run_write(
                executor,
                active_tx,
                DELETE_NODE,
                uuid=node.uuid,
                kind='ENTITY',
                scope_group_id=node.group_id,
            )

    async def delete_by_group_id(
        self,
        executor: QueryExecutor,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        group_id = resolve_group_scope(executor, group_id) or group_id
        async with composite_transaction(executor, tx) as active_tx:
            await run_write(executor, active_tx, DELETE_GROUP_EDGES, group_id=group_id)
            await run_write(
                executor, active_tx, DELETE_GROUP_NODES, group_id=group_id, kind='ENTITY'
            )

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
        group_id: str | None = None,
    ) -> None:
        scope_group_id = resolve_group_scope(executor, group_id)
        if not uuids:
            return
        async with composite_transaction(executor, tx) as active_tx:
            for offset in range(0, len(uuids), batch_size):
                batch = uuids[offset : offset + batch_size]
                for uuid in batch:
                    await run_write(
                        executor,
                        active_tx,
                        DELETE_NODE_EDGES,
                        uuid=uuid,
                        scope_group_id=scope_group_id,
                    )
                await run_write(
                    executor,
                    active_tx,
                    DELETE_NODES_BY_UUIDS,
                    kind='ENTITY',
                    uuids_json=encode_json(batch),
                    scope_group_id=scope_group_id,
                )

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
        group_id: str | None = None,
    ) -> EntityNode:
        records = await run_read(
            executor,
            GET_ENTITY_BY_UUID,
            uuid=uuid,
            scope_group_id=resolve_group_scope(executor, group_id),
        )
        if not records:
            raise NodeNotFoundError(uuid)
        return entity_node_from_record(records[0])

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        group_id: str | None = None,
    ) -> list[EntityNode]:
        if not uuids:
            return []
        records = await run_read(
            executor,
            GET_ENTITIES_BY_UUIDS,
            uuids_json=encode_json(uuids),
            scope_group_id=resolve_group_scope(executor, group_id),
        )
        return [entity_node_from_record(record) for record in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityNode]:
        group_ids = resolve_group_ids(executor, group_ids) or []
        if not group_ids:
            return []
        records = await run_read(
            executor,
            GET_ENTITIES_BY_GROUPS,
            group_ids_json=encode_json(group_ids),
            uuid_cursor=uuid_cursor,
            result_limit=limit if limit is not None else 2_147_483_647,
        )
        return [entity_node_from_record(record) for record in records]

    async def get_by_names(
        self,
        executor: QueryExecutor,
        names: list[str],
        group_ids: list[str],
    ) -> list[EntityNode]:
        unique_names = sorted({name for name in names if name})
        scoped_group_ids = resolve_group_ids(executor, group_ids) or []
        unique_group_ids = sorted({group_id for group_id in scoped_group_ids if group_id})
        if not unique_names or not unique_group_ids:
            return []
        records = await run_read(
            executor,
            GET_ENTITIES_BY_NAMES,
            names_json=encode_json(unique_names),
            group_ids_json=encode_json(unique_group_ids),
        )
        return [entity_node_from_record(record) for record in records]

    async def get_by_attribute_values(
        self,
        executor: QueryExecutor,
        group_id: str,
        attributes: dict[str, Any],
        node_labels: list[str] | None = None,
        limit: int = 100,
    ) -> list[EntityNode]:
        if not attributes or node_labels == [] or limit <= 0:
            return []
        group_id = resolve_group_scope(executor, group_id) or group_id
        property_filters = [
            PropertyFilter(
                property_name=name,
                property_value=value,
                comparison_operator=ComparisonOperator.equals,
            )
            for name, value in sorted(attributes.items())
        ]
        where = build_node_where(
            SearchFilters(node_labels=node_labels, property_filters=property_filters),
            group_ids=[group_id],
        )
        records = await run_read(
            executor,
            GET_ENTITIES_BY_ATTRIBUTES.format(where_sql=where.sql),
            **dict(where.binds),
            result_limit=limit,
        )
        return [entity_node_from_record(record) for record in records]

    async def load_embeddings(self, executor: QueryExecutor, node: EntityNode) -> None:
        validate_request_scope(executor, kind='ENTITY', group_id=node.group_id)
        records = await run_read(
            executor,
            GET_ENTITY_EMBEDDING,
            uuid=node.uuid,
            scope_group_id=node.group_id,
        )
        if not records:
            raise NodeNotFoundError(node.uuid)
        node.name_embedding = ensure_vector(records[0]['name_embedding'])

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        batch_size: int = 100,
    ) -> None:
        by_uuid = {node.uuid: node for node in nodes}
        for offset in range(0, len(nodes), batch_size):
            batch = nodes[offset : offset + batch_size]
            by_group: dict[str, list[EntityNode]] = {}
            for node in batch:
                validate_request_scope(executor, kind='ENTITY', group_id=node.group_id)
                by_group.setdefault(node.group_id, []).append(node)
            for group_id, group_nodes in by_group.items():
                records = await run_read(
                    executor,
                    GET_ENTITIES_BY_UUIDS,
                    uuids_json=encode_json([node.uuid for node in group_nodes]),
                    scope_group_id=group_id,
                )
                for record in records:
                    by_uuid[str(record['uuid'])].name_embedding = ensure_vector(
                        record.get('name_embedding')
                    )
