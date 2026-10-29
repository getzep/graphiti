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

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast
from uuid import NAMESPACE_URL, uuid5

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.namespaces.edges import EdgeNamespace
from graphiti_core.namespaces.nodes import NodeNamespace
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_filters import SearchFilters

CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


class GraphitiSurface(Protocol):
    driver: GraphDriver
    nodes: NodeNamespace
    edges: EdgeNamespace


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    group_id: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    reachable: list[dict[str, Any]]
    evidence_kind: str = 'hermetic-or-live-backend'


def stable_uuid(group_id: str, name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f'ontology-ai:conformance:{group_id}:{name}'))


def make_node(name: str, group_id: str) -> EntityNode:
    display_name = 'Alpha' if name == 'a' else 'Beta'
    return EntityNode(
        uuid=stable_uuid(group_id, name),
        name=display_name,
        group_id=group_id,
        labels=['Concept'],
        summary=f'{display_name} conformance node.',
        attributes={'fixture': 'conformance', 'rank': 1 if name == 'a' else 2},
        created_at=CREATED_AT,
    )


def make_edge(group_id: str) -> EntityEdge:
    return EntityEdge(
        uuid=stable_uuid(group_id, 'a-relates-b'),
        source_node_uuid=stable_uuid(group_id, 'a'),
        target_node_uuid=stable_uuid(group_id, 'b'),
        group_id=group_id,
        name='RELATES_TO',
        fact='Alpha is related to Beta.',
        episodes=[],
        attributes={'fixture': 'conformance'},
        created_at=CREATED_AT,
        valid_at=CREATED_AT,
        reference_time=CREATED_AT,
    )


def normalize_nodes(nodes: list[EntityNode]) -> list[dict[str, Any]]:
    return [
        {
            'uuid': node.uuid,
            'group_id': node.group_id,
            'name': node.name,
            'labels': sorted(node.labels),
            'summary': node.summary,
            'attributes': dict(sorted(node.attributes.items())),
            'created_at': node.created_at.astimezone(UTC).isoformat(),
        }
        for node in sorted(nodes, key=lambda item: item.uuid)
    ]


def normalize_edges(edges: list[EntityEdge]) -> list[dict[str, Any]]:
    return [
        {
            'uuid': edge.uuid,
            'group_id': edge.group_id,
            'source_uuid': edge.source_node_uuid,
            'target_uuid': edge.target_node_uuid,
            'name': edge.name,
            'fact': edge.fact,
            'episodes': sorted(edge.episodes),
            'attributes': dict(sorted(edge.attributes.items())),
            'created_at': edge.created_at.astimezone(UTC).isoformat(),
            'valid_at': edge.valid_at.astimezone(UTC).isoformat() if edge.valid_at else None,
            'invalid_at': (
                edge.invalid_at.astimezone(UTC).isoformat() if edge.invalid_at else None
            ),
            'expired_at': (
                edge.expired_at.astimezone(UTC).isoformat() if edge.expired_at else None
            ),
            'reference_time': (
                edge.reference_time.astimezone(UTC).isoformat() if edge.reference_time else None
            ),
        }
        for edge in sorted(edges, key=lambda item: item.uuid)
    ]


async def run_core_scenario(graphiti: GraphitiSurface, group_id: str) -> ScenarioResult:
    graph_ops = graphiti.driver.graph_ops
    if graph_ops is None:
        raise AssertionError('conformance backend must provide graph_ops')
    await graph_ops.clear_data(graphiti.driver, [group_id])

    async with graphiti.driver.transaction() as tx:
        await graphiti.nodes.entity.save(make_node('a', group_id), tx=tx)
        await graphiti.nodes.entity.save(make_node('b', group_id), tx=tx)
        await graphiti.edges.entity.save(make_edge(group_id), tx=tx)

    nodes = await graphiti.nodes.entity.get_by_group_ids([group_id])
    edges = await graphiti.edges.entity.get_by_group_ids([group_id])
    search_ops = graphiti.driver.search_ops
    if search_ops is None:
        raise AssertionError('conformance backend must provide search_ops')
    reachable = await search_ops.node_bfs_search(
        graphiti.driver,
        [stable_uuid(group_id, 'a')],
        SearchFilters(),
        2,
        [group_id],
        10,
    )
    return ScenarioResult(
        group_id=group_id,
        nodes=normalize_nodes(nodes),
        edges=normalize_edges(cast(list[EntityEdge], edges)),
        reachable=normalize_nodes(reachable),
    )
