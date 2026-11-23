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
import re
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.graph_utils import Neighbor, label_propagation
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.driver.record_parsers import community_node_from_record, entity_node_from_record
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices
from graphiti_core.helpers import SAFE_CYPHER_IDENTIFIER_PATTERN
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN,
    get_entity_node_return_query,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode

logger = logging.getLogger(__name__)

_CREATE_INDEX_NAME_RE = re.compile(
    r'CREATE\s+(?:(?:FULLTEXT|RANGE|VECTOR|POINT)\s+)?INDEX\s+([A-Za-z_][A-Za-z0-9_]*)\b',
    re.IGNORECASE,
)


def _known_index_names() -> list[str]:
    names: list[str] = []
    statements = get_range_indices(GraphProvider.RELATE) + get_fulltext_indices(
        GraphProvider.RELATE
    )
    for stmt in statements:
        match = _CREATE_INDEX_NAME_RE.search(stmt)
        if match is not None:
            names.append(match.group(1))
    return names


class RelateGraphMaintenanceOperations(GraphMaintenanceOperations):
    async def clear_data(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        if group_ids is None:
            await executor.execute_query('MATCH (n) DETACH DELETE n')
        else:
            for label in ['Entity', 'Episodic', 'Community', 'Saga']:
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
        if delete_existing:
            await self.delete_all_indexes(executor)

        range_indices = get_range_indices(GraphProvider.RELATE)
        fulltext_indices = get_fulltext_indices(GraphProvider.RELATE)
        for query in range_indices + fulltext_indices:
            await executor.execute_query(query)

    async def delete_all_indexes(
        self,
        executor: QueryExecutor,
    ) -> None:
        names: list[str] = []
        seen: set[str] = set()

        result = await executor.execute_query('SHOW INDEXES')
        records = result[0] if result else []
        for record in records:
            if record.get('type') == 'LOOKUP':
                continue
            name = record.get('name')
            if isinstance(name, str) and name not in seen:
                seen.add(name)
                names.append(name)

        for name in _known_index_names():
            if name not in seen:
                seen.add(name)
                names.append(name)

        for name in names:
            if SAFE_CYPHER_IDENTIFIER_PATTERN.match(name):
                await executor.execute_query(f'DROP INDEX {name} IF EXISTS')

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

            node_records, _, _ = await executor.execute_query(
                """
                MATCH (n:Entity)
                WHERE n.group_id IN $group_ids
                RETURN
                """
                + get_entity_node_return_query(GraphProvider.RELATE),
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

            for cluster in cluster_uuids:
                if not cluster:
                    continue
                cluster_records, _, _ = await executor.execute_query(
                    """
                    MATCH (n:Entity)
                    WHERE n.uuid IN $uuids
                    RETURN
                    """
                    + get_entity_node_return_query(GraphProvider.RELATE),
                    uuids=cluster,
                )
                community_clusters.append([entity_node_from_record(r) for r in cluster_records])

        return community_clusters

    async def determine_entity_community(
        self,
        executor: QueryExecutor,
        entity: EntityNode,
    ) -> None:
        records, _, _ = await executor.execute_query(
            """
            MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {uuid: $entity_uuid})
            RETURN
            """
            + COMMUNITY_NODE_RETURN,
            entity_uuid=entity.uuid,
        )

        if len(records) > 0:
            return

        records, _, _ = await executor.execute_query(
            """
            MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
            RETURN
            """
            + COMMUNITY_NODE_RETURN,
            entity_uuid=entity.uuid,
        )

    async def get_mentioned_nodes(
        self,
        executor: QueryExecutor,
        episodes: list[EpisodicNode],
    ) -> list[EntityNode]:
        episode_uuids = [episode.uuid for episode in episodes]
        if not episode_uuids:
            return []

        records, _, _ = await executor.execute_query(
            """
            MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity)
            WHERE episode.uuid IN $uuids
            RETURN DISTINCT
            """
            + get_entity_node_return_query(GraphProvider.RELATE),
            uuids=episode_uuids,
        )

        return [entity_node_from_record(r) for r in records]

    async def get_communities_by_nodes(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
    ) -> list[CommunityNode]:
        node_uuids = [node.uuid for node in nodes]
        if not node_uuids:
            return []

        records, _, _ = await executor.execute_query(
            """
            MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)
            WHERE m.uuid IN $uuids
            RETURN DISTINCT
            """
            + COMMUNITY_NODE_RETURN,
            uuids=node_uuids,
        )

        return [community_node_from_record(r) for r in records]
