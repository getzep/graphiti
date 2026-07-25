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

from types import SimpleNamespace

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.graph_ops import FalkorGraphMaintenanceOperations
from graphiti_core.graphiti import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data


class _FakeFalkorDriver:
    provider = GraphProvider.FALKORDB
    graph_operations_interface = FalkorGraphMaintenanceOperations()

    def __init__(self, database: str, group_routing: str = 'database'):
        self._database = database
        self.group_routing = group_routing
        self.clone_calls: list[str] = []
        self.queries: list[tuple[str, dict]] = []

    def clone(self, database: str) -> '_FakeFalkorDriver':
        self.clone_calls.append(database)
        return _FakeFalkorDriver(database, self.group_routing)

    def should_route_group_id_to_database(self, group_id: str) -> bool:
        return self.group_routing == 'database' and group_id != self._database

    async def execute_query(self, query: str, **kwargs):
        self.queries.append((query, kwargs))
        return [], [], None


def _graphiti_with_driver(driver: _FakeFalkorDriver) -> Graphiti:
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.clients = SimpleNamespace(driver=driver)
    return graphiti


def test_record_group_routing_keeps_episode_write_on_fixed_falkor_graph():
    driver = _FakeFalkorDriver(
        'graphiti_personal',
        group_routing='record',
    )
    graphiti = _graphiti_with_driver(driver)

    group_id = graphiti._resolve_group_id_for_write('harmony-cloud-shared-brain')

    assert group_id == 'harmony-cloud-shared-brain'
    assert graphiti.driver is driver
    assert graphiti.clients.driver is driver
    assert driver.clone_calls == []


def test_default_group_routing_preserves_episode_write_graph_per_group():
    driver = _FakeFalkorDriver(
        'graphiti_personal',
        group_routing='database',
    )
    graphiti = _graphiti_with_driver(driver)

    group_id = graphiti._resolve_group_id_for_write('harmony-cloud-shared-brain')

    assert group_id == 'harmony-cloud-shared-brain'
    assert driver.clone_calls == ['harmony-cloud-shared-brain']
    assert graphiti.driver._database == 'harmony-cloud-shared-brain'
    assert graphiti.clients.driver is graphiti.driver


@pytest.mark.asyncio
async def test_record_group_routing_clears_only_scoped_records_without_cloning():
    driver = _FakeFalkorDriver(
        'graphiti_personal',
        group_routing='record',
    )

    await clear_data(driver, group_ids=['harmony-cloud-shared-brain'])

    assert driver.clone_calls == []
    assert len(driver.queries) == 3
    assert all(
        params == {'group_ids': ['harmony-cloud-shared-brain']} for _, params in driver.queries
    )
    assert all('WHERE n.group_id IN $group_ids' in query for query, _ in driver.queries)
