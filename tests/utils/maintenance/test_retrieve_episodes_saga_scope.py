"""Saga episode retrieval must search every requested group."""

from datetime import datetime, timezone

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.utils.maintenance.graph_data_operations import retrieve_episodes

REFERENCE_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)


class RecordingDriver:
    """A driver that records the saga query instead of running it."""

    provider = GraphProvider.NEO4J
    graph_operations_interface = None

    def __init__(self) -> None:
        self.query: str | None = None
        self.params: dict = {}

    async def execute_query(self, query: str, **params):
        self.query = query
        self.params = params
        return [], [], None


@pytest.mark.parametrize(
    ('group_ids', 'expected_group_ids'),
    [
        (['g1', 'g2'], ['g1', 'g2']),
        (['g2'], ['g2']),
        (None, None),
    ],
)
async def test_retrieve_episodes_saga_searches_every_requested_group(
    group_ids: list[str] | None, expected_group_ids: list[str] | None
):
    driver = RecordingDriver()

    await retrieve_episodes(
        driver, REFERENCE_TIME, last_n=3, group_ids=group_ids, saga='onboarding'
    )

    assert driver.query is not None
    assert '$saga_name' in driver.query
    if expected_group_ids is None:
        # No group scope: the saga is looked up by name, not by a null group id,
        # because a node-pattern property equal to null matches nothing.
        assert 'group_id: $group_id' not in driver.query
        assert 's.group_id IN $group_ids' not in driver.query
        assert 'group_ids' not in driver.params
        assert 'group_id' not in driver.params
    else:
        assert 's.group_id IN $group_ids' in driver.query
        assert driver.params['group_ids'] == expected_group_ids
