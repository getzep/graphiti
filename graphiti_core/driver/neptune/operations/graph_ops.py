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

import logging
from typing import TYPE_CHECKING, Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.graph_utils import Neighbor, label_propagation
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.driver.record_parsers import community_node_from_record, entity_node_from_record
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN_NEPTUNE,
    get_entity_node_return_query,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode

if TYPE_CHECKING:
    from graphiti_core.driver.neptune_driver import NeptuneDriver

logger = logging.getLogger(__name__)


class NeptuneGraphMaintenanceOperations(GraphMaintenanceOperations):
    def __init__(self, driver: NeptuneDriver | None = None):
        self._driver = driver

    async def clear_data(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        if group_ids is None:
            await executor.execute_query('MATCH (n) DETACH DELETE n')
        else:
            for label in ['Entity', 'Episodic', 'Community']:
                await executor.execute_query(
                    f"""
                    MATCH (n:{label})
                    WHERE n.group_id IN $group_ids
                    DETACH DELETE n
                    """,
                    group_ids=group_ids,
                )

    async def build_indices_and_constraints(
        self,
        executor: QueryExecutor,
        delete_existing: bool = False,
    ) -> None:
        if self._driver is None:
            return

        if delete_existing:
            await self._driver.delete_aoss_indices()

        await self._driver.create_aoss_indices()

    async def delete_all_indexes(
        self,
        executor: QueryExecutor,
    ) -> None:
        if self._driver is None:
            return
        await self._driver.delete_aoss_indices()

    async def get_community_clusters(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> list[Any]:
        community_clusters: list[list[EntityNode]] = []

        if group_ids is None:
            group_id_values, _, _ = await executor.execute_query(
                """
                MATCH (n:Entity)
                WHERE n.group_id IS NOT NULL
                RETURN
                    collect(DISTINCT n.group_id) AS group_ids
                """
            )
            group_ids = group_id_values[0]['group_ids'] if group_id_values else []

        resolved_group_ids: list[str] = group_ids or []
        for group_id in resolved_group_ids:
            projection: dict[str, list[Neighbor]] = {}

            # Get all entity nodes for this group
            node_records, _, _ = await executor.execute_query(
                """
                MATCH (n:Entity)
                WHERE n.group_id IN $group_ids
                RETURN
                """
                + get_entity_node_return_query(GraphProvider.NEPTUNE),
                group_ids=[group_id],
            )
            nodes = [entity_node_from_record(r) for r in node_records]

            for node in nodes:
                records, _, _ = await executor.execute_query(
                    """
                    MATCH (n:Entity {group_id: $group_id, uuid: $uuid})-[e:RELATES_TO]-(m: Entity {group_id: $group_id})
                    WITH count(e) AS count, m.uuid AS uuid
                    RETURN
                        uuid,
                        count
                    """,
                    uuid=node.uuid,
                    group_id=group_id,
                )

                projection[node.uuid] = [
                    Neighbor(node_uuid=record['uuid'], edge_count=record['count'])
                    for record in records
                ]

            cluster_uuids = label_propagation(projection)

            # Fetch full node objects for each cluster
            for cluster in cluster_uuids:
                if not cluster:
                    continue
                cluster_records, _, _ = await executor.execute_query(
                    """
                    MATCH (n:Entity)
                    WHERE n.uuid IN $uuids
                    RETURN
                    """
                    + get_entity_node_return_query(GraphProvider.NEPTUNE),
                    uuids=cluster,
                )
                community_clusters.append([entity_node_from_record(r) for r in cluster_records])

        return community_clusters

    async def remove_communities(
        self,
        executor: QueryExecutor,
    ) -> None:
        await executor.execute_query(
            """
            MATCH (c:Community)
            DETACH DELETE c
            """
        )

    async def determine_entity_community(
        self,
        executor: QueryExecutor,
        entity: EntityNode,
    ) -> None:
        # Check if the node is already part of a community
        records, _, _ = await executor.execute_query(
            """
            MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {uuid: $entity_uuid})
            WITH c AS n
            RETURN
            """
            + COMMUNITY_NODE_RETURN_NEPTUNE,
            entity_uuid=entity.uuid,
        )

        if len(records) > 0:
            return

        # If the node has no community, find the mode community of surrounding entities
        records, _, _ = await executor.execute_query(
            """
            MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
            WITH c AS n
            RETURN
            """
            + COMMUNITY_NODE_RETURN_NEPTUNE,
            entity_uuid=entity.uuid,
        )

    async def get_mentioned_nodes(
        self,
        executor: QueryExecutor,
        episodes: list[EpisodicNode],
    ) -> list[EntityNode]:
        episode_uuids = [episode.uuid for episode in episodes]

        records, _, _ = await executor.execute_query(
            """
            MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity)
            WHERE episode.uuid IN $uuids
            RETURN DISTINCT
            """
            + get_entity_node_return_query(GraphProvider.NEPTUNE),
            uuids=episode_uuids,
        )

        return [entity_node_from_record(r) for r in records]

    async def get_communities_by_nodes(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
    ) -> list[CommunityNode]:
        node_uuids = [node.uuid for node in nodes]

        records, _, _ = await executor.execute_query(
            """
            MATCH (n:Community)-[:HAS_MEMBER]->(m:Entity)
            WHERE m.uuid IN $uuids
            RETURN DISTINCT
            """
            + COMMUNITY_NODE_RETURN_NEPTUNE,
            uuids=node_uuids,
        )

        return [community_node_from_record(r) for r in records]
