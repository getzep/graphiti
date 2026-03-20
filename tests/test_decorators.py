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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.decorators import handle_multiple_group_ids
from graphiti_core.driver.driver import GraphProvider


def make_falkordb_service(call_tracker):
    """Build a mock service that looks like it uses FalkorDB."""
    driver = MagicMock()
    driver.provider = GraphProvider.FALKORDB
    driver.clone = MagicMock(return_value=driver)

    clients = MagicMock()
    clients.driver = driver

    class FakeService:
        def __init__(self):
            self.clients = clients
            self.max_coroutines = None

        @handle_multiple_group_ids
        async def search(self, query: str, group_ids: list[str] | None = None, driver=None):
            call_tracker.append(group_ids)
            return []

    return FakeService()


def make_neo4j_service(call_tracker):
    """Build a mock service that looks like it uses Neo4j."""
    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J

    clients = MagicMock()
    clients.driver = driver

    class FakeService:
        def __init__(self):
            self.clients = clients
            self.max_coroutines = None

        @handle_multiple_group_ids
        async def search(self, query: str, group_ids: list[str] | None = None, driver=None):
            call_tracker.append(group_ids)
            return []

    return FakeService()


@pytest.mark.asyncio
async def test_falkordb_single_group_id_routes_correctly():
    """Single group_id on FalkorDB must route to the correct graph, not default_db."""
    calls = []
    svc = make_falkordb_service(calls)
    await svc.search('test query', group_ids=['my-group'])
    assert len(calls) == 1, 'Expected exactly one call for a single group_id'
    assert calls[0] == ['my-group'], f'Expected group_id to be passed through, got {calls[0]}'


@pytest.mark.asyncio
async def test_falkordb_multiple_group_ids_routes_each():
    """Multiple group_ids on FalkorDB must produce one call per group_id."""
    calls = []
    svc = make_falkordb_service(calls)
    await svc.search('test query', group_ids=['group-a', 'group-b', 'group-c'])
    assert len(calls) == 3, f'Expected 3 calls (one per group_id), got {len(calls)}'
    passed = {c[0] for c in calls}
    assert passed == {'group-a', 'group-b', 'group-c'}


@pytest.mark.asyncio
async def test_falkordb_no_group_ids_falls_through():
    """No group_ids → normal execution path (no per-group routing)."""
    calls = []
    svc = make_falkordb_service(calls)
    await svc.search('test query', group_ids=None)
    assert len(calls) == 1
    assert calls[0] is None


@pytest.mark.asyncio
async def test_neo4j_single_group_id_falls_through():
    """Neo4j does not use the FalkorDB routing path — should pass through unchanged."""
    calls = []
    svc = make_neo4j_service(calls)
    await svc.search('test query', group_ids=['my-group'])
    assert len(calls) == 1
    assert calls[0] == ['my-group']
