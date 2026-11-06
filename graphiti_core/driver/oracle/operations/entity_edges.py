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

from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError

from ..codecs import edge_from_record, edge_to_binds, encode_json, ensure_vector
from ..sql.edges import (
    DELETE_EDGE,
    DELETE_EDGES_BY_UUIDS,
    GET_EDGE_BY_UUID,
    GET_EDGE_EMBEDDING,
    GET_EDGE_SCOPE,
    GET_EDGES_BY_GROUPS,
    GET_EDGES_BY_UUIDS,
    GET_ENTITY_EDGES_BETWEEN,
    GET_ENTITY_EDGES_BY_NODE,
    MERGE_EDGE,
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
from .edge_base import validate_endpoints


class OracleEntityEdgeOperations(EntityEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        binds = edge_to_binds(edge, kind='RELATES_TO')
        validate_request_scope(executor, kind='RELATES_TO', group_id=edge.group_id)
        if tx is None:
            async with composite_transaction(executor, None) as active_tx:
                await self.save(executor, edge, tx=active_tx)
            return
        await validate_endpoints(
            executor,
            edge,
            kind='RELATES_TO',
            source_kind='ENTITY',
            target_kind='ENTITY',
            tx=tx,
        )
        await validate_uuid_scope(
            executor,
            tx,
            GET_EDGE_SCOPE,
            uuid=edge.uuid,
            kind='RELATES_TO',
            group_id=edge.group_id,
        )
        await run_write(executor, tx, MERGE_EDGE, **binds)

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        if not edges:
            return
        async with composite_transaction(executor, tx) as active_tx:
            for offset in range(0, len(edges), batch_size):
                for edge in edges[offset : offset + batch_size]:
                    await self.save(executor, edge, tx=active_tx)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: EntityEdge,
        tx: Transaction | None = None,
    ) -> None:
        validate_request_scope(executor, kind='RELATES_TO', group_id=edge.group_id)
        await run_write(
            executor,
            tx,
            DELETE_EDGE,
            uuid=edge.uuid,
            kind='RELATES_TO',
            scope_group_id=edge.group_id,
        )

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        group_id: str | None = None,
    ) -> None:
        if uuids:
            await run_write(
                executor,
                tx,
                DELETE_EDGES_BY_UUIDS,
                kind='RELATES_TO',
                uuids_json=encode_json(uuids),
                scope_group_id=resolve_group_scope(executor, group_id),
            )

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
        group_id: str | None = None,
    ) -> EntityEdge:
        records = await run_read(
            executor,
            GET_EDGE_BY_UUID,
            uuid=uuid,
            kind='RELATES_TO',
            scope_group_id=resolve_group_scope(executor, group_id),
        )
        if not records:
            raise EdgeNotFoundError(uuid)
        return edge_from_record(records[0])  # type: ignore[return-value]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        group_id: str | None = None,
    ) -> list[EntityEdge]:
        if not uuids:
            return []
        records = await run_read(
            executor,
            GET_EDGES_BY_UUIDS,
            uuids_json=encode_json(uuids),
            kind='RELATES_TO',
            scope_group_id=resolve_group_scope(executor, group_id),
        )
        return [edge_from_record(record) for record in records]  # type: ignore[misc]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityEdge]:
        group_ids = resolve_group_ids(executor, group_ids) or []
        if not group_ids:
            return []
        records = await run_read(
            executor,
            GET_EDGES_BY_GROUPS,
            group_ids_json=encode_json(group_ids),
            uuid_cursor=uuid_cursor,
            result_limit=limit if limit is not None else 2_147_483_647,
            kind='RELATES_TO',
        )
        if not records:
            raise GroupsEdgesNotFoundError(group_ids)
        return [edge_from_record(record) for record in records]  # type: ignore[misc]

    async def get_between_nodes(
        self,
        executor: QueryExecutor,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[EntityEdge]:
        records = await run_read(
            executor,
            GET_ENTITY_EDGES_BETWEEN,
            source_uuid=source_node_uuid,
            target_uuid=target_node_uuid,
            scope_group_id=resolve_group_scope(executor),
        )
        return [edge_from_record(record) for record in records]  # type: ignore[misc]

    async def get_by_node_uuid(
        self,
        executor: QueryExecutor,
        node_uuid: str,
    ) -> list[EntityEdge]:
        records = await run_read(
            executor,
            GET_ENTITY_EDGES_BY_NODE,
            node_uuid=node_uuid,
            scope_group_id=resolve_group_scope(executor),
        )
        return [edge_from_record(record) for record in records]  # type: ignore[misc]

    async def load_embeddings(self, executor: QueryExecutor, edge: EntityEdge) -> None:
        validate_request_scope(executor, kind='RELATES_TO', group_id=edge.group_id)
        records = await run_read(
            executor,
            GET_EDGE_EMBEDDING,
            uuid=edge.uuid,
            scope_group_id=edge.group_id,
        )
        if not records:
            raise EdgeNotFoundError(edge.uuid)
        edge.fact_embedding = ensure_vector(records[0]['fact_embedding'])

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EntityEdge],
        batch_size: int = 100,
    ) -> None:
        by_uuid = {edge.uuid: edge for edge in edges}
        for offset in range(0, len(edges), batch_size):
            batch = edges[offset : offset + batch_size]
            by_group: dict[str, list[EntityEdge]] = {}
            for edge in batch:
                validate_request_scope(executor, kind='RELATES_TO', group_id=edge.group_id)
                by_group.setdefault(edge.group_id, []).append(edge)
            for group_id, group_edges in by_group.items():
                records = await run_read(
                    executor,
                    GET_EDGES_BY_UUIDS,
                    uuids_json=encode_json([edge.uuid for edge in group_edges]),
                    kind='RELATES_TO',
                    scope_group_id=group_id,
                )
                for record in records:
                    by_uuid[str(record['uuid'])].fact_embedding = ensure_vector(
                        record.get('fact_embedding')
                    )
