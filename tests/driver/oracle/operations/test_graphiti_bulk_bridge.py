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

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from graphiti_core.driver.oracle.legacy import OracleLegacyGraphOperations
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.nodes import EntityNode, EpisodicNode

NOW = datetime(2026, 10, 6, 12, tzinfo=UTC)
VECTOR = [0.0] * 1023 + [1.0]


class CaptureOps:
    def __init__(self) -> None:
        self.models = []
        self.transaction = None

    async def save_bulk(self, _driver, models, *, tx, batch_size):
        self.models = models
        self.transaction = tx


class CaptureScopedOps:
    def __init__(self) -> None:
        self.group_id = None

    async def delete_by_uuids(self, _driver, _uuids, *, group_id, batch_size):
        self.group_id = group_id

    async def get_by_uuids(self, _driver, _uuids, *, group_id):
        self.group_id = group_id
        return []


@pytest.mark.asyncio
async def test_graphiti_bulk_dicts_become_typed_oracle_models_without_losing_attributes() -> None:
    episode_ops = CaptureOps()
    node_ops = CaptureOps()
    mention_ops = CaptureOps()
    fact_ops = CaptureOps()
    driver = SimpleNamespace(
        episode_node_ops=episode_ops,
        entity_node_ops=node_ops,
        episodic_edge_ops=mention_ops,
        entity_edge_ops=fact_ops,
    )
    tx = object()
    bridge = OracleLegacyGraphOperations()

    await bridge.episodic_node_save_bulk(
        None,
        driver,
        tx,
        [
            {
                'uuid': 'episode-1',
                'name': 'REG-01',
                'group_id': 'demo-telco-fieldservice-v1',
                'source': 'text',
                'content': 'Região Norte',
                'source_description': 'document://demo?page=2',
                'created_at': NOW,
                'valid_at': NOW,
            }
        ],
    )
    await bridge.node_save_bulk(
        None,
        driver,
        tx,
        [
            {
                'uuid': 'node-1',
                'name': 'Carla',
                'group_id': 'demo-telco-fieldservice-v1',
                'summary': 'Técnica Norte',
                'labels': ['Entity', 'Tecnico'],
                'created_at': NOW,
                'name_embedding': VECTOR,
                'codigo': 'T01',
            }
        ],
    )
    await bridge.episodic_edge_save_bulk(
        None,
        driver,
        tx,
        [
            {
                'uuid': 'mention-1',
                'source_node_uuid': 'episode-1',
                'target_node_uuid': 'node-1',
                'group_id': 'demo-telco-fieldservice-v1',
                'created_at': NOW,
            }
        ],
    )
    await bridge.edge_save_bulk(
        None,
        driver,
        tx,
        [
            {
                'uuid': 'edge-1',
                'source_node_uuid': 'node-1',
                'target_node_uuid': 'node-2',
                'name': 'ATRIBUIDO_A',
                'fact': 'OS-104 atribuída a Carla',
                'group_id': 'demo-telco-fieldservice-v1',
                'episodes': ['episode-1'],
                'created_at': NOW,
                'valid_at': NOW,
                'reference_time': NOW,
                'fact_embedding': VECTOR,
                'codigo_fonte': 'AGD-D1-B',
            }
        ],
    )

    assert isinstance(episode_ops.models[0], EpisodicNode)
    assert isinstance(node_ops.models[0], EntityNode)
    assert node_ops.models[0].attributes['codigo'] == 'T01'
    assert isinstance(mention_ops.models[0], EpisodicEdge)
    assert isinstance(fact_ops.models[0], EntityEdge)
    assert fact_ops.models[0].attributes['codigo_fonte'] == 'AGD-D1-B'
    assert all(ops.transaction is tx for ops in (episode_ops, node_ops, mention_ops, fact_ops))


@pytest.mark.asyncio
async def test_legacy_bulk_bridges_preserve_explicit_group_scope() -> None:
    operations = CaptureScopedOps()
    driver = SimpleNamespace(entity_node_ops=operations)
    bridge = OracleLegacyGraphOperations()

    await bridge.node_delete_by_uuids(
        None,
        driver,
        ['node-a'],
        group_id='group-b',
    )
    assert operations.group_id == 'group-b'

    await bridge.node_get_by_uuids(
        None,
        driver,
        ['node-a'],
        group_id='group-c',
    )
    assert operations.group_id == 'group-c'
