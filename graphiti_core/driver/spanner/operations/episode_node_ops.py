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

from datetime import datetime
from typing import Any

from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.spanner.codecs import STRING_ARRAY
from graphiti_core.driver.spanner.operations.base import spanner_driver
from graphiti_core.driver.spanner.operations.node_common import (
    delete_group,
    delete_nodes,
    save_rows,
    select_by_group_ids,
    select_by_uuid,
    select_by_uuids,
)
from graphiti_core.driver.spanner.rows import (
    EPISODIC_NODE_COLUMNS,
    episodic_node_from_row,
    episodic_node_to_row,
)
from graphiti_core.driver.spanner.schema import node_select_columns
from graphiti_core.nodes import EpisodicNode

TABLE = 'Episodic'


class SpannerEpisodeNodeOperations(EpisodeNodeOperations):
    """Spanner implementation of ``EpisodeNodeOperations``."""

    async def save(
        self,
        executor: QueryExecutor,
        node: EpisodicNode,
        tx: Transaction | None = None,
    ) -> None:
        await save_rows(
            spanner_driver(executor), TABLE, [episodic_node_to_row(node)], EPISODIC_NODE_COLUMNS, tx
        )

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EpisodicNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        rows = [episodic_node_to_row(node) for node in nodes]
        await save_rows(
            spanner_driver(executor), TABLE, rows, EPISODIC_NODE_COLUMNS, tx, batch_size
        )

    async def delete(
        self,
        executor: QueryExecutor,
        node: EpisodicNode,
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
    ) -> EpisodicNode:
        return episodic_node_from_row(await select_by_uuid(spanner_driver(executor), TABLE, uuid))

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EpisodicNode]:
        records = await select_by_uuids(spanner_driver(executor), TABLE, uuids)
        return [episodic_node_from_row(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EpisodicNode]:
        records = await select_by_group_ids(
            spanner_driver(executor), TABLE, group_ids, limit, uuid_cursor
        )
        return [episodic_node_from_row(r) for r in records]

    async def get_by_entity_node_uuid(
        self,
        executor: QueryExecutor,
        entity_node_uuid: str,
    ) -> list[EpisodicNode]:
        # Mentions.uuid is the episode (source) uuid. IN (subquery) de-duplicates episodes
        # that mention the entity through several edges.
        records = await spanner_driver(executor).read(
            f'SELECT {node_select_columns(TABLE)} FROM {TABLE} WHERE uuid IN '
            '(SELECT uuid FROM Mentions WHERE target_node_uuid = @entity_uuid)',
            {'entity_uuid': entity_node_uuid},
        )
        return [episodic_node_from_row(r) for r in records]

    async def retrieve_episodes(
        self,
        executor: QueryExecutor,
        reference_time: datetime,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: str | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        sql, params = build_retrieve_episodes_query(reference_time, last_n, group_ids, source, saga)
        types = {'g': STRING_ARRAY} if 'g' in params else None
        records = await spanner_driver(executor).read(sql, params, types)
        return [episodic_node_from_row(r) for r in records]

    async def count_mentions_of_entity(
        self,
        executor: QueryExecutor,
        entity_node_uuid: str,
    ) -> int:
        records = await spanner_driver(executor).read(
            'SELECT COUNT(*) AS mention_count FROM Mentions WHERE target_node_uuid = @uuid',
            {'uuid': entity_node_uuid},
        )
        return int(records[0]['mention_count']) if records else 0


def build_retrieve_episodes_query(
    reference_time: datetime,
    last_n: int,
    group_ids: list[str] | None,
    source: str | None,
    saga: str | None,
) -> tuple[str, dict[str, Any]]:
    """SQL and parameters for ``retrieve_episodes``; only the parameters it uses are bound.

    Like Neo4j, the result is newest first (``valid_at DESC``); it is not reversed.
    """
    where = ['valid_at <= @reference_time']
    params: dict[str, Any] = {'reference_time': reference_time, 'last_n': last_n}
    if saga is not None:
        # Neo4j matches the saga in the first requested group and does not filter the
        # episodes' own group in that case.
        saga_clause = (
            'EXISTS (SELECT 1 FROM HasEpisode h JOIN Saga s ON s.uuid = h.uuid '
            'WHERE h.target_node_uuid = Episodic.uuid AND s.name = @saga'
        )
        params['saga'] = saga
        if group_ids:
            saga_clause += ' AND s.group_id = @saga_group'
            params['saga_group'] = group_ids[0]
        where.append(saga_clause + ')')
    elif group_ids:
        where.append('group_id IN UNNEST(@g)')
        params['g'] = list(group_ids)
    if source:
        where.append('source = @source')
        params['source'] = source
    sql = (
        f'SELECT {node_select_columns(TABLE)} FROM {TABLE} WHERE '
        + ' AND '.join(where)
        + ' ORDER BY valid_at DESC LIMIT @last_n'
    )
    return sql, params
