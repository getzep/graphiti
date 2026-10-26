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

from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode

from ..codecs import community_node_from_record, encode_json, entity_node_from_record
from ..community import Neighbor, label_propagation
from ..sql.graph import (
    CLEAR_ALL_EDGES,
    CLEAR_ALL_NODES,
    CLEAR_GROUP_EDGES,
    CLEAR_GROUP_NODES,
    GET_COMMUNITIES_BY_NODES,
    GET_EXISTING_ENTITY_COMMUNITY,
    GET_GROUP_ENTITIES,
    GET_GROUP_IDS,
    GET_GROUP_RELATIONS,
    GET_MENTIONED_NODES,
    GET_NEIGHBOR_COMMUNITIES,
    REMOVE_ALL_COMMUNITIES,
    REMOVE_GROUP_COMMUNITIES,
)
from .base import (
    composite_transaction,
    require_unscoped_admin,
    resolve_group_ids,
    resolve_group_scope,
    run_read,
    run_write,
    validate_request_scope,
)
from .episode_removal import SummaryRebuilder, remove_episode


class OracleGraphMaintenanceOperations(GraphMaintenanceOperations):
    async def remove_episode(
        self,
        executor: QueryExecutor,
        episode_uuid: str,
        rebuild_summary: SummaryRebuilder,
    ) -> None:
        await remove_episode(executor, episode_uuid, rebuild_summary)

    async def clear_data(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        group_ids = resolve_group_ids(executor, group_ids)
        if group_ids is None:
            async with composite_transaction(executor, None) as tx:
                await run_write(executor, tx, CLEAR_ALL_EDGES)
                await run_write(executor, tx, CLEAR_ALL_NODES)
            return
        if not group_ids:
            return
        binds = {'group_ids_json': encode_json(group_ids)}
        async with composite_transaction(executor, None) as tx:
            await run_write(executor, tx, CLEAR_GROUP_EDGES, **binds)
            await run_write(executor, tx, CLEAR_GROUP_NODES, **binds)

    async def build_indices_and_constraints(
        self,
        executor: QueryExecutor,
        delete_existing: bool = False,
    ) -> None:
        require_unscoped_admin(executor, operation='build_indices_and_constraints')
        await executor.build_indices_and_constraints(delete_existing=delete_existing)  # type: ignore[attr-defined]

    async def delete_all_indexes(self, executor: QueryExecutor) -> None:
        require_unscoped_admin(executor, operation='delete_all_indexes')
        await executor.delete_all_indexes()  # type: ignore[attr-defined]

    async def get_community_clusters(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> list[Any]:
        group_ids = resolve_group_ids(executor, group_ids)
        if group_ids is None:
            records = await run_read(executor, GET_GROUP_IDS)
            group_ids = [str(record['group_id']) for record in records]

        clusters: list[list[EntityNode]] = []
        for group_id in group_ids:
            node_records = await run_read(executor, GET_GROUP_ENTITIES, group_id=group_id)
            nodes = [entity_node_from_record(record) for record in node_records]
            node_map = {node.uuid: node for node in nodes}
            projection = {node.uuid: [] for node in nodes}
            relations = await run_read(executor, GET_GROUP_RELATIONS, group_id=group_id)
            for relation in relations:
                source = str(relation['source_uuid'])
                target = str(relation['target_uuid'])
                count = int(relation['edge_count'])
                if source in projection and target in projection:
                    projection[source].append(Neighbor(node_uuid=target, edge_count=count))
                    projection[target].append(Neighbor(node_uuid=source, edge_count=count))
            for uuids in label_propagation(projection):
                cluster = [node_map[uuid] for uuid in uuids if uuid in node_map]
                if cluster:
                    clusters.append(cluster)
        return clusters

    async def remove_communities(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        group_ids = resolve_group_ids(executor, group_ids)
        if group_ids is None:
            async with composite_transaction(executor, None) as tx:
                await run_write(
                    executor,
                    tx,
                    "DELETE FROM graphiti_edges WHERE kind = 'HAS_MEMBER'",
                )
                await run_write(executor, tx, REMOVE_ALL_COMMUNITIES)
        elif group_ids:
            binds = {'group_ids_json': encode_json(group_ids)}
            async with composite_transaction(executor, None) as tx:
                await run_write(
                    executor,
                    tx,
                    "DELETE FROM graphiti_edges WHERE kind = 'HAS_MEMBER' "
                    'AND group_id IN (SELECT group_id FROM JSON_TABLE('
                    ":group_ids_json, '$[*]' COLUMNS (group_id VARCHAR2(255) PATH '$')))",
                    **binds,
                )
                await run_write(executor, tx, REMOVE_GROUP_COMMUNITIES, **binds)

    async def determine_entity_community(
        self,
        executor: QueryExecutor,
        entity: EntityNode,
    ) -> None:
        await self.determine_entity_community_result(executor, entity)

    async def determine_entity_community_result(
        self,
        executor: QueryExecutor,
        entity: EntityNode,
    ) -> tuple[CommunityNode | None, bool]:
        validate_request_scope(executor, kind='ENTITY', group_id=entity.group_id)
        scope_group_id = resolve_group_scope(executor, entity.group_id)
        records = await run_read(
            executor,
            GET_EXISTING_ENTITY_COMMUNITY,
            entity_uuid=entity.uuid,
            scope_group_id=scope_group_id,
        )
        if records:
            return community_node_from_record(records[0]), False
        records = await run_read(
            executor,
            GET_NEIGHBOR_COMMUNITIES,
            entity_uuid=entity.uuid,
            scope_group_id=scope_group_id,
        )
        return (community_node_from_record(records[0]), True) if records else (None, False)

    async def get_mentioned_nodes(
        self,
        executor: QueryExecutor,
        episodes: list[EpisodicNode],
    ) -> list[EntityNode]:
        if not episodes:
            return []
        for episode in episodes:
            validate_request_scope(executor, kind='EPISODIC', group_id=episode.group_id)
        records = await run_read(
            executor,
            GET_MENTIONED_NODES,
            episode_uuids=encode_json([episode.uuid for episode in episodes]),
            scope_group_id=resolve_group_scope(executor),
        )
        return [entity_node_from_record(record) for record in records]

    async def get_communities_by_nodes(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
    ) -> list[CommunityNode]:
        if not nodes:
            return []
        for node in nodes:
            validate_request_scope(executor, kind='ENTITY', group_id=node.group_id)
        records = await run_read(
            executor,
            GET_COMMUNITIES_BY_NODES,
            node_uuids=encode_json([node.uuid for node in nodes]),
            scope_group_id=resolve_group_scope(executor),
        )
        return [community_node_from_record(record) for record in records]
