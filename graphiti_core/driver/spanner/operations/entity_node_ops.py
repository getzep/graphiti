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

from __future__ import annotations

from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.spanner.operations.base import spanner_driver
from graphiti_core.driver.spanner.operations.node_common import (
    delete_group,
    delete_nodes,
    save_rows,
    select_by_group_ids,
    select_by_uuid,
    select_by_uuids,
    select_embeddings,
)
from graphiti_core.driver.spanner.rows import (
    ENTITY_NODE_COLUMNS,
    entity_node_from_row,
    entity_node_to_row,
)
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EntityNode

TABLE = 'Entity'


class SpannerEntityNodeOperations(EntityNodeOperations):
    """Spanner implementation of ``EntityNodeOperations``."""

    async def save(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        await save_rows(
            spanner_driver(executor), TABLE, [entity_node_to_row(node)], ENTITY_NODE_COLUMNS, tx
        )

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        rows = [entity_node_to_row(node) for node in nodes]
        await save_rows(spanner_driver(executor), TABLE, rows, ENTITY_NODE_COLUMNS, tx, batch_size)

    async def delete(
        self,
        executor: QueryExecutor,
        node: EntityNode,
        tx: Transaction | None = None,
    ) -> None:
        await delete_nodes(spanner_driver(executor), TABLE, [node.uuid], tx)

    async def delete_by_group_id(
        self,
        executor: QueryExecutor,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await delete_group(spanner_driver(executor), TABLE, group_id, tx, batch_size)

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        await delete_nodes(spanner_driver(executor), TABLE, uuids, tx, batch_size)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> EntityNode:
        return entity_node_from_row(await select_by_uuid(spanner_driver(executor), TABLE, uuid))

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EntityNode]:
        records = await select_by_uuids(spanner_driver(executor), TABLE, uuids)
        return [entity_node_from_row(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityNode]:
        records = await select_by_group_ids(
            spanner_driver(executor), TABLE, group_ids, limit, uuid_cursor
        )
        return [entity_node_from_row(r) for r in records]

    async def load_embeddings(
        self,
        executor: QueryExecutor,
        node: EntityNode,
    ) -> None:
        embeddings = await select_embeddings(spanner_driver(executor), TABLE, [node.uuid])
        if node.uuid not in embeddings:
            raise NodeNotFoundError(node.uuid)
        node.name_embedding = embeddings[node.uuid]

    async def load_embeddings_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
        batch_size: int = 100,
    ) -> None:
        driver = spanner_driver(executor)
        embeddings = await select_embeddings(driver, TABLE, [n.uuid for n in nodes], batch_size)
        for node in nodes:
            if node.uuid in embeddings:
                node.name_embedding = embeddings[node.uuid]
