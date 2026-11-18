"""Regression tests: ``EntityNode.get_by_uuids`` must honour its ``group_id`` scope.

The ``graph_operations_interface`` path has always applied the scope, but the Cypher
fallback that every shipped driver uses did not, so a uuid that exists in more than one
group came back from every group even when the caller passed ``group_id``.

Callers rely on the scope. ``resolve_extracted_edges`` fetches dangling edge endpoints
with the comment "Limit the lookup to the edge group", and
``GraphReader.nodes_by_uuids`` keeps a concurrent ingestion commit inside its graph.
"""

from datetime import datetime, timezone

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.nodes import EntityNode


class GroupScopedDriver:
    """A driver with the group semantics of the Cypher that the fallback builds."""

    graph_operations_interface = None
    provider = GraphProvider.NEO4J

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.queries: list[tuple[str, dict]] = []

    async def execute_query(self, query: str, **params):
        self.queries.append((query, params))
        uuids = params.get('uuids') or []
        # The result is restricted to a group only when the query binds `$group_id`.
        scoped = '$group_id' in query
        records = [
            {
                'uuid': row['uuid'],
                'name': row['name'],
                'group_id': row['group_id'],
                'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'summary': '',
                'labels': ['Entity'],
                'attributes': {},
            }
            for row in self.rows
            if row['uuid'] in uuids and (not scoped or row['group_id'] == params.get('group_id'))
        ]
        return records, None, None


ROWS = [
    {'uuid': 'shared-uuid', 'name': 'Alice', 'group_id': 'tenant-a'},
    {'uuid': 'shared-uuid', 'name': 'Alice', 'group_id': 'tenant-b'},
]


@pytest.mark.asyncio
async def test_get_by_uuids_returns_only_the_requested_group():
    driver = GroupScopedDriver(ROWS)

    nodes = await EntityNode.get_by_uuids(driver, ['shared-uuid'], group_id='tenant-a')

    assert [node.group_id for node in nodes] == ['tenant-a']


@pytest.mark.asyncio
async def test_get_by_uuids_binds_the_group_scope_in_the_query():
    driver = GroupScopedDriver(ROWS)

    await EntityNode.get_by_uuids(driver, ['shared-uuid'], group_id='tenant-a')

    query, params = driver.queries[0]
    assert 'n.group_id = $group_id' in query
    assert params['group_id'] == 'tenant-a'


@pytest.mark.asyncio
async def test_get_by_uuids_scopes_the_default_group():
    """The default group id is '' - a real scope, not "no scope"."""
    driver = GroupScopedDriver(
        [
            {'uuid': 'shared-uuid', 'name': 'Alice', 'group_id': ''},
            {'uuid': 'shared-uuid', 'name': 'Alice', 'group_id': 'tenant-b'},
        ]
    )

    nodes = await EntityNode.get_by_uuids(driver, ['shared-uuid'], group_id='')

    assert [node.group_id for node in nodes] == ['']


@pytest.mark.asyncio
async def test_get_by_uuids_without_group_id_keeps_every_group():
    driver = GroupScopedDriver(ROWS)

    nodes = await EntityNode.get_by_uuids(driver, ['shared-uuid'])

    assert sorted(node.group_id for node in nodes) == ['tenant-a', 'tenant-b']
    assert '$group_id' not in driver.queries[0][0]
