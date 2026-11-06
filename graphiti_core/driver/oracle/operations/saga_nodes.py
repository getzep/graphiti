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

from datetime import datetime

from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import SagaNode

from ..codecs import (
    decode_oracle_datetime,
    encode_json,
    optional_oracle_datetime,
    optional_utc_datetime,
    saga_node_from_record,
    saga_node_to_binds,
)
from ..sql.nodes import (
    DELETE_GROUP_EDGES,
    DELETE_GROUP_NODES,
    DELETE_NODE,
    DELETE_NODE_EDGES,
    DELETE_NODES_BY_UUIDS,
    GET_NODE_SCOPE,
    GET_PREVIOUS_SAGA_EPISODE,
    GET_SAGA_BY_UUID,
    GET_SAGA_EPISODE_CONTENTS,
    GET_SAGAS_BY_GROUPS,
    GET_SAGAS_BY_UUIDS,
    MERGE_SAGA_NODE,
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


class OracleSagaNodeOperations(SagaNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: SagaNode,
        tx: Transaction | None = None,
    ) -> None:
        binds = saga_node_to_binds(node)
        if tx is None:
            async with composite_transaction(executor, None) as active_tx:
                await self.save(executor, node, tx=active_tx)
            return
        await validate_uuid_scope(
            executor,
            tx,
            GET_NODE_SCOPE,
            uuid=node.uuid,
            kind='SAGA',
            group_id=node.group_id,
        )
        await run_write(executor, tx, MERGE_SAGA_NODE, **binds)

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[SagaNode],
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
        node: SagaNode,
        tx: Transaction | None = None,
    ) -> None:
        validate_request_scope(executor, kind='SAGA', group_id=node.group_id)
        async with composite_transaction(executor, tx) as active_tx:
            await run_write(
                executor, active_tx, DELETE_NODE_EDGES, uuid=node.uuid, scope_group_id=node.group_id
            )
            await run_write(
                executor,
                active_tx,
                DELETE_NODE,
                uuid=node.uuid,
                kind='SAGA',
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
            await run_write(executor, active_tx, DELETE_GROUP_NODES, group_id=group_id, kind='SAGA')

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
                    kind='SAGA',
                    uuids_json=encode_json(batch),
                    scope_group_id=scope_group_id,
                )

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
        group_id: str | None = None,
    ) -> SagaNode:
        records = await run_read(
            executor,
            GET_SAGA_BY_UUID,
            uuid=uuid,
            scope_group_id=resolve_group_scope(executor, group_id),
        )
        if not records:
            raise NodeNotFoundError(uuid)
        return saga_node_from_record(records[0])

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        group_id: str | None = None,
    ) -> list[SagaNode]:
        if not uuids:
            return []
        records = await run_read(
            executor,
            GET_SAGAS_BY_UUIDS,
            uuids_json=encode_json(uuids),
            scope_group_id=resolve_group_scope(executor, group_id),
        )
        return [saga_node_from_record(record) for record in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[SagaNode]:
        group_ids = resolve_group_ids(executor, group_ids) or []
        if not group_ids:
            return []
        records = await run_read(
            executor,
            GET_SAGAS_BY_GROUPS,
            group_ids_json=encode_json(group_ids),
            uuid_cursor=uuid_cursor,
            result_limit=limit if limit is not None else 2_147_483_647,
        )
        return [saga_node_from_record(record) for record in records]

    async def get_previous_episode_uuid(
        self,
        executor: QueryExecutor,
        saga_uuid: str,
        current_episode_uuid: str,
    ) -> str | None:
        records = await run_read(
            executor,
            GET_PREVIOUS_SAGA_EPISODE,
            saga_uuid=saga_uuid,
            current_episode_uuid=current_episode_uuid,
            scope_group_id=resolve_group_scope(executor),
        )
        return str(records[0]['uuid']) if records else None

    async def get_episode_contents(
        self,
        executor: QueryExecutor,
        saga_uuid: str,
        since: datetime | None = None,
        after_uuid: str | None = None,
        limit: int = 200,
    ) -> list[tuple[str, datetime | None, datetime, str]]:
        records = await run_read(
            executor,
            GET_SAGA_EPISODE_CONTENTS,
            saga_uuid=saga_uuid,
            since=optional_utc_datetime(since),
            after_uuid=after_uuid,
            result_limit=max(0, limit),
            scope_group_id=resolve_group_scope(executor),
        )
        return [
            (
                str(record['content']),
                optional_oracle_datetime(record.get('valid_at')),
                decode_oracle_datetime(record['created_at']),
                str(record['uuid']),
            )
            for record in records
            if record.get('content') and record.get('created_at') is not None
        ]
