"""Regression tests for Neptune fulltext search group_id scoping (no live database).

node_fulltext_search, episode_fulltext_search and community_fulltext_search all
compute a group_id filter for their non-Neptune (Cypher fulltext procedure) branch,
but the Neptune branch -- which resolves candidate uuids via AOSS and then re-MATCHes
them by uuid -- never spliced that filter into its own Cypher. A caller passing
group_ids therefore got results across every group/tenant on Neptune, even though the
same call correctly scoped to group_ids on every other backend.

That Cypher-side filter alone is not sufficient: run_aoss_query pulls the top-`limit`
hits globally before the Cypher filter runs, so a requested-group hit ranked just below
the cutoff is dropped even though it should have been returned. The AOSS query itself
must also filter by group_id, which is why run_aoss_query below records the group_ids
it was called with.

Mirrors the RecordingExecutor approach of tests/utils/search/test_edge_bfs_query_shape.py:
a recording driver captures the emitted Cypher, so no database connection is required.
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    episode_fulltext_search,
    node_fulltext_search,
)


def _make_row() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        'uuid': 'hit-uuid',
        'name': 'hit name',
        'group_id': 'group-a',
        'created_at': now,
        'summary': 'hit summary',
        'labels': [],
        'attributes': {},
        'name_embedding': None,
        'content': 'hit content',
        'valid_at': now,
        'source': 'text',
        'source_description': 'hit source',
        'entity_edges': [],
    }


class RecordingNeptuneDriver:
    """Captures the AOSS call and the Cypher/params a search function emits."""

    provider = GraphProvider.NEPTUNE
    search_interface = None
    fulltext_syntax = ''

    def __init__(self):
        self.cypher_query = ''
        self.params: dict[str, Any] = {}
        self.aoss_group_ids: list[str] | None = 'not-called'  # type: ignore[assignment]

    def run_aoss_query(
        self, index_name: str, query_text: str, limit: int = 10, group_ids: list[str] | None = None
    ):
        self.aoss_group_ids = group_ids
        return {
            'hits': {
                'total': {'value': 1},
                'hits': [{'_source': {'uuid': 'hit-uuid'}, '_score': 1}],
            }
        }

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.cypher_query = cypher_query_
        self.params = kwargs
        return [_make_row()], None, None


@pytest.mark.asyncio
async def test_neptune_node_fulltext_search_scopes_by_group_id():
    driver = RecordingNeptuneDriver()

    results = await node_fulltext_search(
        driver,  # type: ignore[arg-type]
        'api test system',
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert driver.aoss_group_ids == ['group-a']
    assert 'AND n.group_id IN $group_ids' in driver.cypher_query
    assert driver.params.get('group_ids') == ['group-a']
    assert len(results) == 1
    assert results[0].uuid == 'hit-uuid'


@pytest.mark.asyncio
async def test_neptune_node_fulltext_search_omits_filter_without_group_ids():
    driver = RecordingNeptuneDriver()

    await node_fulltext_search(
        driver,  # type: ignore[arg-type]
        'api test system',
        SearchFilters(),
        group_ids=None,
    )

    assert driver.aoss_group_ids is None
    assert 'n.group_id IN $group_ids' not in driver.cypher_query


@pytest.mark.asyncio
async def test_neptune_episode_fulltext_search_scopes_by_group_id():
    driver = RecordingNeptuneDriver()

    results = await episode_fulltext_search(
        driver,  # type: ignore[arg-type]
        'api test system',
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert driver.aoss_group_ids == ['group-a']
    assert 'AND e.group_id IN $group_ids' in driver.cypher_query
    assert 'e.uuid=i.id' in driver.cypher_query
    assert driver.params.get('group_ids') == ['group-a']
    assert driver.params.get('ids') == [{'id': 'hit-uuid', 'score': 1}]
    assert len(results) == 1
    assert results[0].uuid == 'hit-uuid'


@pytest.mark.asyncio
async def test_neptune_community_fulltext_search_scopes_by_group_id():
    driver = RecordingNeptuneDriver()

    results = await community_fulltext_search(
        driver,  # type: ignore[arg-type]
        'api test system',
        group_ids=['group-a'],
    )

    assert driver.aoss_group_ids == ['group-a']
    # The MATCH in this branch binds `comm`, not `c` -- the filter must use the
    # alias the query actually defines, not the one the non-Neptune branch uses.
    assert 'AND comm.group_id IN $group_ids' in driver.cypher_query
    assert driver.params.get('group_ids') == ['group-a']
    assert len(results) == 1
    assert results[0].uuid == 'hit-uuid'
