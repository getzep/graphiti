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

from datetime import datetime, timezone

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.nodes import EntityNode

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_edge(uuid: str = 'e1') -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id='g',
        source_node_uuid='a',
        target_node_uuid='b',
        name='WORKS_AT',
        fact='Ana works at Acme',
        created_at=T0,
    )


def test_edges_with_the_same_uuid_are_equal():
    assert make_edge() == make_edge()
    assert make_edge('e1') != make_edge('e2')


def test_an_edge_is_equal_to_itself():
    edge = make_edge()

    assert edge == edge
    assert edge in [edge]


def test_an_edge_never_equals_a_node():
    edge = make_edge('e1')
    node = EntityNode(uuid='e1', name='Acme', group_id='g')

    assert edge != node
    assert node != edge


def test_edges_compare_across_subclasses_by_uuid():
    edge = make_edge('e9')
    episodic_edge = EpisodicEdge(
        uuid='e9',
        group_id='g',
        source_node_uuid='ep1',
        target_node_uuid='n1',
        created_at=T0,
    )

    assert edge == episodic_edge
    assert hash(edge) == hash(episodic_edge)


def test_edge_containers_follow_uuid_identity():
    first = make_edge('e1')
    second = make_edge('e1')
    other = make_edge('e2')

    assert second in [first, other]
    assert len({first, second}) == 1

    items = [first, other]
    items.remove(second)
    assert items == [other]
