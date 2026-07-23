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
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.models.nodes.node_db_queries import SAGA_NODE_RETURN, get_saga_node_save_query
from graphiti_core.nodes import SagaNode

logger = logging.getLogger(__name__)


def _saga_node_from_record(record: Any) -> SagaNode:
    return SagaNode(
        uuid=record['uuid'],
        name=record['name'],
        group_id=record['group_id'],
        created_at=parse_db_date(record['created_at']),  # type: ignore[arg-type]
    )


class FalkorSagaNodeOperations(SagaNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: SagaNode,
        tx: Transaction | None = None,
    ) -> None:
        query = get_saga_node_save_query(GraphProvider.FALKORDB)
        params: dict[str, Any] = {
            'uuid': node.uuid,
            'name': node.name,
            'group_id': node.group_id,
            'created_at': node.created_at,
        }
        if tx is not None:
            await tx.run(query, **params)
        else:
            await executor.execute_query(query, **params)

        logger.debug(f'Saved Saga Node to Graph: {node.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[SagaNode],
        tx: Transaction | None = None,
        batch_size: int = 100,  # noqa: ARG002
    ) -> None:
        for node in nodes:
            await self.save(executor, node, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        node: SagaNode,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (n:Saga {uuid: $uuid})
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
        batch_size: int = 100,  # noqa: ARG002
    ) -> None:
        query = """
            MATCH (n:Saga {group_id: $group_id})
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
        batch_size: int = 100,  # noqa: ARG002
    ) -> None:
        query = """
            MATCH (n:Saga)
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
    ) -> SagaNode:
        query = (
            """
            MATCH (s:Saga {uuid: $uuid})
            RETURN
            """
            + SAGA_NODE_RETURN
        )
        records, _, _ = await executor.execute_query(query, uuid=uuid)
        nodes = [_saga_node_from_record(r) for r in records]
        if len(nodes) == 0:
            raise NodeNotFoundError(uuid)
        return nodes[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[SagaNode]:
        query = (
            """
            MATCH (s:Saga)
            WHERE s.uuid IN $uuids
            RETURN
            """
            + SAGA_NODE_RETURN
        )
        records, _, _ = await executor.execute_query(query, uuids=uuids)
        return [_saga_node_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[SagaNode]:
        cursor_clause = 'AND s.uuid < $uuid' if uuid_cursor else ''
        limit_clause = 'LIMIT $limit' if limit is not None else ''
        query = (
            """
            MATCH (s:Saga)
            WHERE s.group_id IN $group_ids
            """
            + cursor_clause
            + """
            RETURN
            """
            + SAGA_NODE_RETURN
            + """
            ORDER BY s.uuid DESC
            """
            + limit_clause
        )
        records, _, _ = await executor.execute_query(
            query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
        )
        return [_saga_node_from_record(r) for r in records]
