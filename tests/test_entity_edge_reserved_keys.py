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

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.entity_edge_ops import FalkorEntityEdgeOperations
from graphiti_core.driver.record_parsers import entity_edge_from_record, entity_node_from_record
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx
from graphiti_core.utils.datetime_utils import utc_now

SOURCE = 'real-source'
TARGET = 'real-target'
POLLUTED = {
    'source_uuid': 'old-s',
    'target_uuid': 'old-t',
    'source_node_uuid': 'evil-s',
    'target_node_uuid': 'evil-t',
    'custom': 'kept',
}


def _edge() -> EntityEdge:
    return EntityEdge(
        source_node_uuid=SOURCE,
        target_node_uuid=TARGET,
        name='RELATES_TO',
        fact='src relates to tgt',
        fact_embedding=[0.1, 0.2],
        group_id='g1',
        created_at=utc_now(),
        attributes=dict(POLLUTED),
    )


def _assert_clean(edge_data: dict[str, Any]) -> None:
    assert 'source_uuid' not in edge_data
    assert 'target_uuid' not in edge_data
    assert edge_data['custom'] == 'kept'


async def test_falkor_ops_save_drops_endpoint_keys_from_attributes():
    executor = MagicMock()
    executor.execute_query = AsyncMock()

    await FalkorEntityEdgeOperations().save(executor, _edge())

    kwargs = executor.execute_query.await_args.kwargs
    _assert_clean(kwargs['edge_data'])
    assert 'source_node_uuid' not in kwargs['edge_data']
    assert 'target_node_uuid' not in kwargs['edge_data']
    assert kwargs['source_uuid'] == SOURCE
    assert kwargs['target_uuid'] == TARGET


async def test_falkor_ops_save_bulk_keeps_trusted_endpoints():
    executor = MagicMock()
    executor.execute_query = AsyncMock()

    await FalkorEntityEdgeOperations().save_bulk(executor, [_edge()])

    edge_data = executor.execute_query.await_args.kwargs['entity_edges'][0]
    _assert_clean(edge_data)
    assert edge_data['source_node_uuid'] == SOURCE
    assert edge_data['target_node_uuid'] == TARGET


async def test_legacy_entity_edge_save_drops_endpoint_keys_on_falkordb():
    driver = MagicMock()
    driver.provider = GraphProvider.FALKORDB
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock()

    await _edge().save(driver)

    kwargs = driver.execute_query.await_args.kwargs
    _assert_clean(kwargs['edge_data'])
    assert 'source_node_uuid' not in kwargs['edge_data']
    assert 'target_node_uuid' not in kwargs['edge_data']
    assert kwargs['source_uuid'] == SOURCE
    assert kwargs['target_uuid'] == TARGET


async def test_bulk_utils_drops_endpoint_keys_from_attributes():
    driver = MagicMock()
    driver.provider = GraphProvider.FALKORDB
    driver.graph_operations_interface = None
    tx = MagicMock()
    tx.run = AsyncMock()

    await add_nodes_and_edges_bulk_tx(tx, [], [], [], [_edge()], MagicMock(), driver)

    edge_data = next(
        c.kwargs['entity_edges'][0] for c in tx.run.await_args_list if 'entity_edges' in c.kwargs
    )
    _assert_clean(edge_data)
    assert edge_data['source_node_uuid'] == SOURCE
    assert edge_data['target_node_uuid'] == TARGET


def test_record_parsers_strip_stray_endpoint_keys():
    def record() -> dict[str, Any]:
        return {
            'uuid': 'e1',
            'source_node_uuid': SOURCE,
            'target_node_uuid': TARGET,
            'name': 'RELATES_TO',
            'fact': 'f',
            'fact_embedding': None,
            'group_id': 'g1',
            'episodes': [],
            'created_at': utc_now(),
            'expired_at': None,
            'valid_at': None,
            'invalid_at': None,
            'attributes': {'uuid': 'e1', **POLLUTED},
        }

    for edge in (
        entity_edge_from_record(record()),
        get_entity_edge_from_record(record(), GraphProvider.FALKORDB),
    ):
        assert edge.attributes == {'custom': 'kept'}
        assert edge.source_node_uuid == SOURCE
        assert edge.target_node_uuid == TARGET


def test_entity_node_parser_still_strips_node_fields():
    node = entity_node_from_record(
        {
            'uuid': 'n1',
            'name': 'A',
            'name_embedding': [0.1],
            'group_id': 'g1',
            'labels': ['Entity'],
            'created_at': utc_now(),
            'summary': 's',
            'attributes': {
                'uuid': 'n1',
                'name': 'A',
                'group_id': 'g1',
                'name_embedding': [0.1],
                'summary': 's',
                'created_at': 'x',
                'labels': ['Entity'],
                'age': 3,
            },
        }
    )
    assert node.attributes == {'age': 3}
