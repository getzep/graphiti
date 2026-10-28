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

# Graph maintenance for Spanner: clearing data, schema, and community detection support.
#
# Group deletes use partitioned DML on every EDGE table before the NODE tables (spec section
# 6): deleting parent rows alone cascades into interleaved children whose global index
# entries exceed the mutation limit. Community queries mirror the Neo4j ops: the clustering
# projection is one GQL query per group instead of one Cypher query per entity, and the
# label propagation itself is the shared ``graph_utils.label_propagation``.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.graph_utils import Neighbor, label_propagation
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.driver.spanner.codecs import STRING_ARRAY
from graphiti_core.driver.spanner.operations.base import spanner_driver
from graphiti_core.driver.spanner.rows import community_node_from_row, entity_node_from_row
from graphiti_core.driver.spanner.schema import (
    COMMUNITY_TABLE,
    EDGE_TABLES,
    ENTITY_TABLE,
    HAS_MEMBER_TABLE,
    MENTIONS_TABLE,
    NODE_TABLES,
    RELATES_TO_TABLE,
    node_select_columns,
)
from graphiti_core.helpers import semaphore_gather
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode

if TYPE_CHECKING:
    from graphiti_core.driver.spanner.driver import SpannerDriver

ENTITY = ENTITY_TABLE.name
COMMUNITY = COMMUNITY_TABLE.name
HAS_MEMBER = HAS_MEMBER_TABLE.name
MENTIONS = MENTIONS_TABLE.name

# Maximum uuids bound in one IN UNNEST parameter; longer inputs are read in chunks.
_READ_CHUNK = 1000

# ---------------------------------------------------------------------------
# Query builders (table, label and column names come from schema constants only)
# ---------------------------------------------------------------------------


def build_delete_by_groups_dml(table: str) -> str:
    """Partitioned DML deleting the rows of ``table`` whose ``group_id`` is in ``@group_ids``."""
    return f'DELETE FROM {table} WHERE group_id IN UNNEST(@group_ids)'


def build_delete_all_dml(table: str) -> str:
    return f'DELETE FROM {table} WHERE TRUE'


def build_entity_group_ids_query() -> str:
    """Distinct group ids that have at least one entity."""
    return f'SELECT DISTINCT group_id FROM {ENTITY} ORDER BY group_id'


def build_group_entities_query() -> str:
    """Every entity of ``@group_id``; the order fixes the label-propagation start labels."""
    return (
        f'SELECT {node_select_columns(ENTITY)} FROM {ENTITY} '
        'WHERE group_id = @group_id ORDER BY uuid DESC'
    )


def build_relates_to_projection_query(graph_name: str) -> str:
    """For every entity of ``@group_id``, its RELATES_TO neighbours in the same group with the
    number of edges between them in either direction (Neo4j's ``count(e)`` per neighbour).
    ``COUNT(e)`` is rejected on a graph element, so the edge uuid is counted."""
    entity = ENTITY_TABLE.label
    return (
        f'GRAPH {graph_name}\n'
        f'MATCH (n:{entity})-[e:{RELATES_TO_TABLE.label}]-(m:{entity})\n'
        'WHERE n.group_id = @group_id AND m.group_id = @group_id\n'
        'RETURN n.uuid AS uuid, m.uuid AS neighbor_uuid, COUNT(e.uuid) AS edge_count\n'
        'GROUP BY uuid, neighbor_uuid'
    )


def build_member_communities_query(graph_name: str) -> str:
    """Communities that have a HAS_MEMBER edge to the entity ``@uuid``."""
    return (
        f'GRAPH {graph_name}\n'
        f'MATCH (c:{COMMUNITY_TABLE.label})-[:{HAS_MEMBER_TABLE.label}]->'
        f'(n:{ENTITY_TABLE.label})\n'
        'WHERE n.uuid = @uuid\n'
        f'RETURN {node_select_columns(COMMUNITY, "c")}'
    )


def build_neighbor_community_votes_query(graph_name: str) -> str:
    """Per community, the number of (HAS_MEMBER, RELATES_TO) paths from it to the entity
    ``@uuid`` through a neighbour: the rows Neo4j's majority vote counts."""
    entity = ENTITY_TABLE.label
    return (
        f'GRAPH {graph_name}\n'
        f'MATCH (c:{COMMUNITY_TABLE.label})-[:{HAS_MEMBER_TABLE.label}]->(m:{entity})'
        f'-[:{RELATES_TO_TABLE.label}]-(n:{entity})\n'
        'WHERE n.uuid = @uuid\n'
        'RETURN c.uuid AS uuid, COUNT(*) AS votes\n'
        'GROUP BY uuid'
    )


def build_community_by_uuid_query() -> str:
    return f'SELECT {node_select_columns(COMMUNITY)} FROM {COMMUNITY} WHERE uuid = @uuid'


def build_mentioned_entities_query() -> str:
    """Entities targeted by a MENTIONS edge from any episode in ``@uuids``.

    MENTIONS rows are interleaved in their episode, so ``uuid IN UNNEST`` is a key-prefix
    lookup; the semi-join returns each entity once (Neo4j's ``RETURN DISTINCT``)."""
    return (
        f'SELECT {node_select_columns(ENTITY)} FROM {ENTITY} '
        f'WHERE uuid IN (SELECT target_node_uuid FROM {MENTIONS} WHERE uuid IN UNNEST(@uuids))'
    )


def build_communities_by_members_query() -> str:
    """Communities with a HAS_MEMBER edge to an existing entity in ``@uuids``, each once."""
    return (
        f'SELECT {node_select_columns(COMMUNITY)} FROM {COMMUNITY} '
        f'WHERE uuid IN (SELECT hm.uuid FROM {HAS_MEMBER} AS hm '
        f'JOIN {ENTITY} AS m ON m.uuid = hm.target_node_uuid '
        'WHERE hm.target_node_uuid IN UNNEST(@uuids))'
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


async def _read_by_uuids(
    driver: SpannerDriver, sql: str, uuids: Sequence[str]
) -> list[dict[str, Any]]:
    """Run ``sql`` (binding ``@uuids``) over ``uuids`` in chunks; rows de-duplicated by uuid."""
    unique = _unique(uuids)
    rows: dict[str, dict[str, Any]] = {}
    for start in range(0, len(unique), _READ_CHUNK):
        records = await driver.read(
            sql, {'uuids': unique[start : start + _READ_CHUNK]}, {'uuids': STRING_ARRAY}
        )
        for record in records:
            rows.setdefault(record['uuid'], record)
    return list(rows.values())


async def _group_clusters(driver: SpannerDriver, group_id: str) -> list[list[EntityNode]]:
    """Label-propagation clusters of one group, as lists of entity nodes."""
    params = {'group_id': group_id}
    node_rows, edge_rows = await semaphore_gather(
        driver.read(build_group_entities_query(), params),
        driver.read(build_relates_to_projection_query(driver.graph_name), params),
    )
    nodes = {row['uuid']: entity_node_from_row(row) for row in node_rows}

    # Entities without neighbours stay in the projection with an empty list, so they come
    # back as their own cluster. The two reads use separate snapshots: rows naming an entity
    # missing from the node snapshot (written in between) are skipped, because
    # label_propagation needs every neighbour to be a projection key.
    projection: dict[str, list[Neighbor]] = {uuid: [] for uuid in nodes}
    for row in edge_rows:
        neighbors = projection.get(row['uuid'])
        if neighbors is not None and row['neighbor_uuid'] in projection:
            neighbors.append(
                Neighbor(node_uuid=row['neighbor_uuid'], edge_count=int(row['edge_count']))
            )

    clusters = label_propagation(projection)
    return [[nodes[uuid] for uuid in cluster] for cluster in clusters if cluster]


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class SpannerGraphMaintenanceOperations(GraphMaintenanceOperations):
    """Spanner graph maintenance: clearing data, schema, communities."""

    async def clear_data(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        """Delete everything (``group_ids=None``) or every node and edge of the groups.

        Partitioned DML on each edge table, then on each node table; not atomic, like
        Neo4j's per-label deletes. Edges are matched by their own ``group_id``, so an edge of
        another group pointing at a deleted node is left dangling (Graphiti writes edges in
        their endpoints' group). An empty list deletes nothing, as in Neo4j.
        """
        driver = spanner_driver(executor)
        if group_ids is None:
            await driver.clear_tables()
            return
        if not group_ids:
            return
        params = {'group_ids': _unique(group_ids)}
        for table in (*EDGE_TABLES, *NODE_TABLES):
            await driver.execute_partitioned_dml(
                build_delete_by_groups_dml(table), params, {'group_ids': STRING_ARRAY}
            )

    async def build_indices_and_constraints(
        self,
        executor: QueryExecutor,
        delete_existing: bool = False,
    ) -> None:
        await spanner_driver(executor).build_indices_and_constraints(delete_existing)

    async def delete_all_indexes(
        self,
        executor: QueryExecutor,
    ) -> None:
        await spanner_driver(executor).delete_all_indexes()

    async def get_community_clusters(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> list[list[EntityNode]]:
        """Entity clusters for community building, one list of nodes per cluster.

        ``group_ids=None`` means every group that has entities. Per group: all its entities
        plus one GQL query for their RELATES_TO neighbour counts within the group, then label
        propagation. Isolated entities come back as single-node clusters.
        """
        driver = spanner_driver(executor)
        if group_ids is None:
            records = await driver.read(build_entity_group_ids_query())
            group_ids = [record['group_id'] for record in records]

        per_group = await semaphore_gather(
            *[_group_clusters(driver, group_id) for group_id in _unique(group_ids)]
        )
        return [cluster for clusters in per_group for cluster in clusters]

    async def remove_communities(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        """Delete the communities of the groups (every community when ``group_ids`` is empty
        or ``None``, as in the base class) with their HAS_MEMBER edges. Entities are kept.

        HAS_MEMBER rows are interleaved in their community, so a row the group filter misses
        is removed by the cascade from its community row.
        """
        driver = spanner_driver(executor)
        for table in (HAS_MEMBER, COMMUNITY):
            if group_ids:
                await driver.execute_partitioned_dml(
                    build_delete_by_groups_dml(table),
                    {'group_ids': _unique(group_ids)},
                    {'group_ids': STRING_ARRAY},
                )
            else:
                await driver.execute_partitioned_dml(build_delete_all_dml(table))

    async def determine_entity_community(
        self,
        executor: QueryExecutor,
        entity: EntityNode,
    ) -> tuple[CommunityNode | None, bool]:
        """The community the entity belongs to, and whether that membership is new.

        ``(community, False)`` if the entity already has a HAS_MEMBER edge. Otherwise the
        community with the most (HAS_MEMBER, RELATES_TO) paths to the entity through its
        neighbours as ``(community, True)``, or ``(None, False)`` when no neighbour has one.
        This is the shape of ``community_operations.determine_entity_community`` and the
        legacy interface; the abstract method is annotated ``None``.
        """
        driver = spanner_driver(executor)
        graph = driver.graph_name
        params = {'uuid': entity.uuid}

        members = await driver.read(build_member_communities_query(graph), params)
        if members:
            # Neo4j takes the first record (arbitrary order); pick the smallest uuid.
            first = min(members, key=lambda row: row['uuid'])
            return community_node_from_row(first), False

        votes = await driver.read(build_neighbor_community_votes_query(graph), params)
        if not votes:
            return None, False
        # Most votes wins; ties (record order in Neo4j) go to the smallest uuid.
        winner = min(votes, key=lambda row: (-int(row['votes']), row['uuid']))
        records = await driver.read(build_community_by_uuid_query(), {'uuid': winner['uuid']})
        if not records:
            return None, False
        return community_node_from_row(records[0]), True

    async def get_mentioned_nodes(
        self,
        executor: QueryExecutor,
        episodes: list[EpisodicNode],
    ) -> list[EntityNode]:
        """Distinct entities mentioned by any of the episodes."""
        uuids = [episode.uuid for episode in episodes]
        if not uuids:
            return []
        records = await _read_by_uuids(
            spanner_driver(executor), build_mentioned_entities_query(), uuids
        )
        return [entity_node_from_row(record) for record in records]

    async def get_communities_by_nodes(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
    ) -> list[CommunityNode]:
        """Distinct communities that have any of the nodes as a member."""
        uuids = [node.uuid for node in nodes]
        if not uuids:
            return []
        records = await _read_by_uuids(
            spanner_driver(executor), build_communities_by_members_query(), uuids
        )
        return [community_node_from_row(record) for record in records]
