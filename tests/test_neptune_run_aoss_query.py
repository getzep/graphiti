"""Regression tests for NeptuneDriver.run_aoss_query's group_id filtering (no live database).

run_aoss_query fetches the top-`limit` hits from OpenSearch before any Cypher-side
group_id filter runs downstream. If the AOSS query itself doesn't scope by group_id,
a caller can get zero or too-few results back even when matching documents exist in
the requested group, just ranked below the global top-`limit` cutoff. These tests
build a NeptuneDriver without a live cluster (via __new__) and assert the query body
sent to the OpenSearch client carries an exact-match group_id filter.
"""

from typing import Any

from graphiti_core.driver.neptune_driver import NeptuneDriver


class RecordingAossClient:
    def __init__(self):
        self.body: dict[str, Any] = {}
        self.index: str = ''

    def search(self, body: dict[str, Any], index: str):
        self.body = body
        self.index = index
        return {'hits': {'total': {'value': 0}, 'hits': []}}


def _make_driver() -> NeptuneDriver:
    driver = NeptuneDriver.__new__(NeptuneDriver)
    driver.aoss_client = RecordingAossClient()  # type: ignore[attr-defined]
    return driver


def test_run_aoss_query_filters_by_group_id():
    driver = _make_driver()

    driver.run_aoss_query(
        'node_name_and_summary', 'api test system', limit=5, group_ids=['group-a']
    )

    body = driver.aoss_client.body  # type: ignore[attr-defined]
    assert body['size'] == 5
    assert body['query']['bool']['filter'] == {'terms': {'group_id.keyword': ['group-a']}}
    assert body['query']['bool']['must']['multi_match']['query'] == 'api test system'


def test_run_aoss_query_omits_filter_without_group_ids():
    driver = _make_driver()

    driver.run_aoss_query('node_name_and_summary', 'api test system', limit=5, group_ids=None)

    body = driver.aoss_client.body  # type: ignore[attr-defined]
    assert 'bool' not in body['query']
    assert body['query']['multi_match']['query'] == 'api test system'
