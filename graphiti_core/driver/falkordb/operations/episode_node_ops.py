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
from datetime import datetime
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.driver.record_parsers import episodic_node_from_record
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.models.nodes.node_db_queries import (
    EPISODIC_NODE_RETURN,
    get_episode_node_save_bulk_query,
    get_episode_node_save_query,
)
from graphiti_core.nodes import EpisodicNode

logger = logging.getLogger(__name__)


class FalkorEpisodeNodeOperations(EpisodeNodeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        node: EpisodicNode,
        tx: Transaction | None = None,
    ) -> None:
        query = get_episode_node_save_query(GraphProvider.FALKORDB)
        params: dict[str, Any] = {
            'uuid': node.uuid,
            'name': node.name,
            'group_id': node.group_id,
            'source_description': node.source_description,
            'content': node.content,
            'entity_edges': node.entity_edges,
            'created_at': node.created_at,
            'valid_at': node.valid_at,
            'source': node.source.value,
        }
        if tx is not None:
            await tx.run(query, **params)
        else:
            await executor.execute_query(query, **params)

        logger.debug(f'Saved Episode to Graph: {node.uuid}')

    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[EpisodicNode],
        tx: Transaction | None = None,
        batch_size: int = 100,  # noqa: ARG002
    ) -> None:
        episodes = []
        for node in nodes:
            ep = dict(node)
            ep['source'] = str(ep['source'].value)
            ep.pop('labels', None)
            episodes.append(ep)

        query = get_episode_node_save_bulk_query(GraphProvider.FALKORDB)
        if tx is not None:
            await tx.run(query, episodes=episodes)
        else:
            await executor.execute_query(query, episodes=episodes)

    async def delete(
        self,
        executor: QueryExecutor,
        node: EpisodicNode,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (n {uuid: $uuid})
            WHERE n:Entity OR n:Episodic OR n:Community
            OPTIONAL MATCH (n)-[r]-()
            WITH collect(r.uuid) AS edge_uuids, n
            DETACH DELETE n
            RETURN edge_uuids
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
            MATCH (n:Episodic {group_id: $group_id})
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
            MATCH (n:Episodic)
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
    ) -> EpisodicNode:
        query = (
            """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN
            """
            + EPISODIC_NODE_RETURN
        )
        records, _, _ = await executor.execute_query(query, uuid=uuid)
        episodes = [episodic_node_from_record(r) for r in records]
        if len(episodes) == 0:
            raise NodeNotFoundError(uuid)
        return episodes[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EpisodicNode]:
        query = (
            """
            MATCH (e:Episodic)
            WHERE e.uuid IN $uuids
            RETURN DISTINCT
            """
            + EPISODIC_NODE_RETURN
        )
        records, _, _ = await executor.execute_query(query, uuids=uuids)
        return [episodic_node_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EpisodicNode]:
        cursor_clause = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_clause = 'LIMIT $limit' if limit is not None else ''
        query = (
            """
            MATCH (e:Episodic)
            WHERE e.group_id IN $group_ids
            """
            + cursor_clause
            + """
            RETURN DISTINCT
            """
            + EPISODIC_NODE_RETURN
            + """
            ORDER BY uuid DESC
            """
            + limit_clause
        )
        records, _, _ = await executor.execute_query(
            query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
        )
        return [episodic_node_from_record(r) for r in records]

    async def get_by_entity_node_uuid(
        self,
        executor: QueryExecutor,
        entity_node_uuid: str,
    ) -> list[EpisodicNode]:
        query = (
            """
            MATCH (e:Episodic)-[r:MENTIONS]->(n:Entity {uuid: $entity_node_uuid})
            RETURN DISTINCT
            """
            + EPISODIC_NODE_RETURN
        )
        records, _, _ = await executor.execute_query(query, entity_node_uuid=entity_node_uuid)
        return [episodic_node_from_record(r) for r in records]

    async def retrieve_episodes(
        self,
        executor: QueryExecutor,
        reference_time: datetime,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: str | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        if saga is not None and group_ids is not None and len(group_ids) > 0:
            source_clause = 'AND e.source = $source' if source else ''
            query = (
                """
                MATCH (s:Saga {name: $saga_name, group_id: $group_id})-[:HAS_EPISODE]->(e:Episodic)
                WHERE e.valid_at <= $reference_time
                """
                + source_clause
                + """
                RETURN
                """
                + EPISODIC_NODE_RETURN
                + """
                ORDER BY e.valid_at DESC
                LIMIT $num_episodes
                """
            )
            records, _, _ = await executor.execute_query(
                query,
                saga_name=saga,
                group_id=group_ids[0],
                reference_time=reference_time,
                source=source,
                num_episodes=last_n,
            )
        else:
            source_clause = 'AND e.source = $source' if source else ''
            group_clause = 'AND e.group_id IN $group_ids' if group_ids else ''
            query = (
                """
                MATCH (e:Episodic)
                WHERE e.valid_at <= $reference_time
                """
                + group_clause
                + source_clause
                + """
                RETURN
                """
                + EPISODIC_NODE_RETURN
                + """
                ORDER BY e.valid_at DESC
                LIMIT $num_episodes
                """
            )
            records, _, _ = await executor.execute_query(
                query,
                reference_time=reference_time,
                group_ids=group_ids,
                source=source,
                num_episodes=last_n,
            )

        return [episodic_node_from_record(r) for r in records]
