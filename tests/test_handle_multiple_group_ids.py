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

from graphiti_core.decorators import handle_multiple_group_ids
from graphiti_core.driver.driver import GraphProvider, GroupRouting


class _FakeDriver:
    def __init__(
        self,
        database: str,
        provider: GraphProvider = GraphProvider.FALKORDB,
        group_routing: GroupRouting = GroupRouting.DATABASE,
    ):
        self._database = database
        self.provider = provider
        self.group_routing = group_routing
        self.clone_calls: list[str] = []

    @property
    def routes_group_ids_to_databases(self) -> bool:
        return self.group_routing is GroupRouting.DATABASE

    def clone(self, database: str) -> '_FakeDriver':
        self.clone_calls.append(database)
        if self.group_routing is GroupRouting.RECORD:
            # Mirrors FalkorDriver.clone: record routing never re-points the driver.
            return self
        return _FakeDriver(database, self.provider, self.group_routing)


class _Host:
    def __init__(self, driver: _FakeDriver):
        self.clients = SimpleNamespace(driver=driver)
        self.max_coroutines = None
        self.seen_drivers: list[str | None] = []

    @handle_multiple_group_ids
    async def search(
        self,
        query: str,
        group_ids: list[str] | None = None,
        driver=None,
    ):
        db = getattr(driver, '_database', None) if driver is not None else None
        self.seen_drivers.append(db)
        return [f'{query}:{db}:{group_ids}']


@pytest.mark.asyncio
async def test_falkor_single_group_id_clones_driver_when_database_differs():
    """Single group_id must route to that graph, not the driver's default (#1659)."""
    driver = _FakeDriver('reggraph')
    host = _Host(driver)

    result = await host.search('q', group_ids=['acptprobe'])

    assert driver.clone_calls == ['acptprobe']
    # Shared driver must not be reassigned (call-scoped clone only)
    assert host.clients.driver is driver
    assert host.clients.driver._database == 'reggraph'
    assert host.seen_drivers == ['acptprobe']
    assert result == ["q:acptprobe:['acptprobe']"]


@pytest.mark.asyncio
async def test_falkor_single_group_id_skips_clone_when_already_on_graph():
    driver = _FakeDriver('acptprobe')
    host = _Host(driver)

    result = await host.search('q', group_ids=['acptprobe'])

    assert driver.clone_calls == []
    assert host.seen_drivers == [None]  # uses default path without injecting driver
    assert result == ["q:None:['acptprobe']"]


@pytest.mark.asyncio
async def test_falkor_multi_group_ids_still_clones_each():
    driver = _FakeDriver('reggraph')
    host = _Host(driver)

    result = await host.search('q', group_ids=['a', 'b'])

    assert driver.clone_calls == ['a', 'b']
    assert host.clients.driver is driver
    assert set(host.seen_drivers) == {'a', 'b'}
    assert sorted(result) == ["q:a:['a']", "q:b:['b']"]


@pytest.mark.asyncio
async def test_falkor_record_routing_single_group_id_stays_on_fixed_graph():
    """Record routing keeps the configured graph; group_id only filters records (#1684)."""
    driver = _FakeDriver('graphiti_personal', group_routing=GroupRouting.RECORD)
    host = _Host(driver)

    result = await host.search('q', group_ids=['projecta'])

    assert driver.clone_calls == []
    assert host.clients.driver is driver
    assert host.clients.driver._database == 'graphiti_personal'
    assert host.seen_drivers == [None]
    assert result == ["q:None:['projecta']"]


@pytest.mark.asyncio
async def test_falkor_record_routing_multi_group_ids_run_as_one_query():
    """Record routing needs no per-group fan-out: one query filters all group_ids."""
    driver = _FakeDriver('graphiti_personal', group_routing=GroupRouting.RECORD)
    host = _Host(driver)

    result = await host.search('q', group_ids=['a', 'b'])

    assert driver.clone_calls == []
    assert host.seen_drivers == [None]
    assert result == ["q:None:['a', 'b']"]


@pytest.mark.asyncio
async def test_non_falkor_single_group_id_is_passthrough():
    driver = _FakeDriver('neo4j', provider=GraphProvider.NEO4J)
    host = _Host(driver)

    result = await host.search('q', group_ids=['tenant'])

    assert driver.clone_calls == []
    assert host.seen_drivers == [None]
    assert result == ["q:None:['tenant']"]
