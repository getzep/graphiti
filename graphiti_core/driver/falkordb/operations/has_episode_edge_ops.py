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

from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import HasEpisodeEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.models.edges.edge_db_queries import (
    HAS_EPISODE_EDGE_RETURN,
    HAS_EPISODE_EDGE_SAVE,
)

logger = logging.getLogger(__name__)


def _has_episode_edge_from_record(record: Any) -> HasEpisodeEdge:
    return HasEpisodeEdge(
        uuid=record['uuid'],
        group_id=record['group_id'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_db_date(record['created_at']),  # type: ignore[arg-type]
    )


class FalkorHasEpisodeEdgeOperations(HasEpisodeEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: HasEpisodeEdge,
        tx: Transaction | None = None,
    ) -> None:
        params: dict[str, Any] = {
            'saga_uuid': edge.source_node_uuid,
            'episode_uuid': edge.target_node_uuid,
            'uuid': edge.uuid,
            'group_id': edge.group_id,
            'created_at': edge.created_at,
        }
        if tx is not None:
            await tx.run(HAS_EPISODE_EDGE_SAVE, **params)
        else:
            await executor.execute_query(HAS_EPISODE_EDGE_SAVE, **params)

        logger.debug(f'Saved Edge to Graph: {edge.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[HasEpisodeEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,  # noqa: ARG002
    ) -> None:
        for edge in edges:
            await self.save(executor, edge, tx=tx)

    async def delete(
        self,
        executor: QueryExecutor,
        edge: HasEpisodeEdge,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (n:Saga)-[e:HAS_EPISODE {uuid: $uuid}]->(m:Episodic)
            DELETE e
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
            MATCH (n:Saga)-[e:HAS_EPISODE]->(m:Episodic)
            WHERE e.uuid IN $uuids
            DELETE e
        """
        if tx is not None:
            await tx.run(query, uuids=uuids)
        else:
            await executor.execute_query(query, uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> HasEpisodeEdge:
        query = (
            """
            MATCH (n:Saga)-[e:HAS_EPISODE {uuid: $uuid}]->(m:Episodic)
            RETURN
            """
            + HAS_EPISODE_EDGE_RETURN
        )
        records, _, _ = await executor.execute_query(query, uuid=uuid)
        edges = [_has_episode_edge_from_record(r) for r in records]
        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[HasEpisodeEdge]:
        query = (
            """
            MATCH (n:Saga)-[e:HAS_EPISODE]->(m:Episodic)
            WHERE e.uuid IN $uuids
            RETURN
            """
            + HAS_EPISODE_EDGE_RETURN
        )
        records, _, _ = await executor.execute_query(query, uuids=uuids)
        return [_has_episode_edge_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[HasEpisodeEdge]:
        cursor_clause = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_clause = 'LIMIT $limit' if limit is not None else ''
        query = (
            """
            MATCH (n:Saga)-[e:HAS_EPISODE]->(m:Episodic)
            WHERE e.group_id IN $group_ids
            """
            + cursor_clause
            + """
            RETURN
            """
            + HAS_EPISODE_EDGE_RETURN
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
        return [_has_episode_edge_from_record(r) for r in records]
