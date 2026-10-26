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

from typing import ClassVar, Generic, TypeVar, cast

from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import Edge
from graphiti_core.errors import EdgeNotFoundError

from ..codecs import edge_from_record, edge_to_binds, encode_json
from ..errors import InvalidEdgeEndpoint
from ..sql.edges import (
    DELETE_EDGE,
    DELETE_EDGES_BY_UUIDS,
    GET_EDGE_BY_UUID,
    GET_EDGE_SCOPE,
    GET_EDGES_BY_GROUPS,
    GET_EDGES_BY_UUIDS,
    MERGE_EDGE,
    VALIDATE_ENDPOINTS,
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


async def validate_endpoints(
    executor: QueryExecutor,
    edge: Edge,
    *,
    kind: str,
    source_kind: str,
    target_kind: str,
    tx: Transaction | None = None,
) -> None:
    if tx is None:
        raise TypeError('endpoint validation requires an active transaction')
    binds = {
        'source_uuid': edge.source_node_uuid,
        'target_uuid': edge.target_node_uuid,
    }
    records, _, _ = await tx.run(VALIDATE_ENDPOINTS, **binds)
    endpoints = {str(record['uuid']): record for record in records}
    source = endpoints.get(edge.source_node_uuid)
    target = endpoints.get(edge.target_node_uuid)
    actual_source = source.get('kind') if source else None
    actual_target = target.get('kind') if target else None
    if (actual_source, actual_target) != (source_kind, target_kind):
        raise InvalidEdgeEndpoint(
            f'{kind} requires {source_kind} -> {target_kind}; '
            f'received {actual_source} -> {actual_target}'
        )
    source_group_id = source.get('group_id') if source else None
    target_group_id = target.get('group_id') if target else None
    if source_group_id != edge.group_id or target_group_id != edge.group_id:
        raise InvalidEdgeEndpoint(
            f'{kind} endpoints and edge must belong to one group; '
            f'edge={edge.group_id}, source={source_group_id}, target={target_group_id}'
        )


EdgeT = TypeVar('EdgeT', bound=Edge)


class StructuralEdgeOperationsMixin(Generic[EdgeT]):
    kind: ClassVar[str]
    source_kind: ClassVar[str]
    target_kind: ClassVar[str]

    async def save(
        self,
        executor: QueryExecutor,
        edge: EdgeT,
        tx: Transaction | None = None,
    ) -> None:
        binds = edge_to_binds(edge, kind=self.kind)
        validate_request_scope(executor, kind=self.kind, group_id=edge.group_id)
        if tx is None:
            async with composite_transaction(executor, None) as active_tx:
                await self.save(executor, edge, tx=active_tx)
            return
        await validate_endpoints(
            executor,
            edge,
            kind=self.kind,
            source_kind=self.source_kind,
            target_kind=self.target_kind,
            tx=tx,
        )
        await validate_uuid_scope(
            executor,
            tx,
            GET_EDGE_SCOPE,
            uuid=edge.uuid,
            kind=self.kind,
            group_id=edge.group_id,
        )
        await run_write(executor, tx, MERGE_EDGE, **binds)

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EdgeT],
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
        edge: EdgeT,
        tx: Transaction | None = None,
    ) -> None:
        validate_request_scope(executor, kind=self.kind, group_id=edge.group_id)
        await run_write(
            executor,
            tx,
            DELETE_EDGE,
            uuid=edge.uuid,
            kind=self.kind,
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
                kind=self.kind,
                uuids_json=encode_json(uuids),
                scope_group_id=resolve_group_scope(executor, group_id),
            )

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
        group_id: str | None = None,
    ) -> EdgeT:
        records = await run_read(
            executor,
            GET_EDGE_BY_UUID,
            uuid=uuid,
            kind=self.kind,
            scope_group_id=resolve_group_scope(executor, group_id),
        )
        if not records:
            raise EdgeNotFoundError(uuid)
        return cast(EdgeT, edge_from_record(records[0]))

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        group_id: str | None = None,
    ) -> list[EdgeT]:
        if not uuids:
            return []
        records = await run_read(
            executor,
            GET_EDGES_BY_UUIDS,
            uuids_json=encode_json(uuids),
            kind=self.kind,
            scope_group_id=resolve_group_scope(executor, group_id),
        )
        return [cast(EdgeT, edge_from_record(record)) for record in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EdgeT]:
        group_ids = resolve_group_ids(executor, group_ids) or []
        if not group_ids:
            return []
        records = await run_read(
            executor,
            GET_EDGES_BY_GROUPS,
            group_ids_json=encode_json(group_ids),
            uuid_cursor=uuid_cursor,
            result_limit=limit if limit is not None else 2_147_483_647,
            kind=self.kind,
        )
        return [cast(EdgeT, edge_from_record(record)) for record in records]
