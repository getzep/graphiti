from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neptune.operations.search_ops import NeptuneSearchOperations
from graphiti_core.driver.neptune_driver import NeptuneDriver
from graphiti_core.nodes import CommunityNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    episode_fulltext_search,
    node_fulltext_search,
)


class StubNeptuneDriver:
    provider = GraphProvider.NEPTUNE
    search_interface = None
    fulltext_syntax = ''
    graph_operations_interface = None

    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.aoss_calls: list[tuple[str, list[dict[str, str]]]] = []

    def run_aoss_query(self, index_name: str, query: str, limit: int = 10):
        return {
            'hits': {
                'total': {'value': 1},
                'hits': [{'_source': {'uuid': 'result-uuid'}, '_score': 1.0}],
            }
        }

    async def execute_query(self, cypher_query: str, **params: Any):
        self.calls.append((cypher_query, params))
        return [], None, None

    async def save_to_aoss(self, index_name: str, data: list[dict[str, str]]):
        self.aoss_calls.append((index_name, data))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('search_kind', 'group_filter'),
    [
        ('node', 'n.group_id IN $group_ids'),
        ('episode', 'e.group_id IN $group_ids'),
        ('community', 'comm.group_id IN $group_ids'),
    ],
)
async def test_legacy_fulltext_search_filters_group_ids(search_kind: str, group_filter: str):
    driver = StubNeptuneDriver()
    group_ids = ['group-a']

    if search_kind == 'node':
        await node_fulltext_search(driver, 'query', SearchFilters(), group_ids=group_ids)  # type: ignore[arg-type]
    elif search_kind == 'episode':
        await episode_fulltext_search(driver, 'query', SearchFilters(), group_ids=group_ids)  # type: ignore[arg-type]
    else:
        await community_fulltext_search(driver, 'query', group_ids=group_ids)  # type: ignore[arg-type]

    query, params = driver.calls[0]
    assert group_filter in query
    assert params['group_ids'] == group_ids
    if search_kind == 'episode':
        assert 'e.uuid=i.id' in query
        assert 'e.uuid=i.uuid' not in query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('search_kind', 'group_filter'),
    [
        ('node', 'n.group_id IN $group_ids'),
        ('episode', 'e.group_id IN $group_ids'),
        ('community', 'n.group_id IN $group_ids'),
    ],
)
async def test_neptune_search_operations_fulltext_filters_group_ids(
    search_kind: str, group_filter: str
):
    driver = StubNeptuneDriver()
    executor = StubNeptuneDriver()
    search_ops = NeptuneSearchOperations(driver)  # type: ignore[arg-type]
    group_ids = ['group-a']

    if search_kind == 'node':
        await search_ops.node_fulltext_search(
            executor,
            'query',
            SearchFilters(),
            group_ids=group_ids,  # type: ignore[arg-type]
        )
    elif search_kind == 'episode':
        await search_ops.episode_fulltext_search(
            executor,
            'query',
            SearchFilters(),
            group_ids=group_ids,  # type: ignore[arg-type]
        )
    else:
        await search_ops.community_fulltext_search(executor, 'query', group_ids=group_ids)  # type: ignore[arg-type]

    query, params = executor.calls[0]
    assert group_filter in query
    assert params['group_ids'] == group_ids
    if search_kind == 'episode':
        assert 'e.uuid=i.id' in query
        assert 'e.uuid=i.uuid' not in query


def test_neptune_parameter_sanitizer_keeps_malicious_edge_types_bound():
    driver = NeptuneDriver.__new__(NeptuneDriver)
    query = 'MATCH (e) WHERE e.name IN $edge_types RETURN e'
    edge_types = ['WORKS_AT"}) MATCH (n) DETACH DELETE n //']
    params = {'edge_types': edge_types.copy()}

    sanitized_query = driver._sanitize_parameters(query, params)

    assert sanitized_query == query
    assert params['edge_types'] == edge_types


def test_neptune_parameter_sanitizer_keeps_edge_type_values_bound():
    driver = NeptuneDriver.__new__(NeptuneDriver)
    query = 'MATCH (e) WHERE e.name IN $edge_types RETURN e'
    params = {'edge_types': ['WORKS_AT']}

    sanitized_query = driver._sanitize_parameters(query, params)

    assert sanitized_query == query
    assert params['edge_types'] == ['WORKS_AT']


def test_neptune_parameter_sanitizer_inlines_parsed_datetime_strings():
    driver = NeptuneDriver.__new__(NeptuneDriver)
    query = 'MATCH (n) WHERE n.created_at IN $created_at RETURN n'
    params = {'created_at': ['2026-01-01T00:00:00+00:00']}

    sanitized_query = driver._sanitize_parameters(query, params)

    assert sanitized_query == (
        'MATCH (n) WHERE n.created_at IN [datetime("2026-01-01T00:00:00+00:00")] RETURN n'
    )


def test_neptune_parameter_sanitizer_matches_full_parameter_names():
    driver = NeptuneDriver.__new__(NeptuneDriver)
    query = 'MATCH (n) WHERE n.uuid = $id AND n.uuid IN $ids RETURN n'
    params = {'id': 'node-uuid', 'ids': ['2026-01-01T00:00:00+00:00']}

    sanitized_query = driver._sanitize_parameters(query, params)

    assert sanitized_query == (
        'MATCH (n) WHERE n.uuid = $id AND n.uuid IN '
        '[datetime("2026-01-01T00:00:00+00:00")] RETURN n'
    )


@pytest.mark.asyncio
async def test_community_node_save_uses_community_name_index():
    driver = StubNeptuneDriver()
    community = CommunityNode(uuid='community-uuid', name='community', group_id='group-a')

    await community.save(driver)  # type: ignore[arg-type]

    assert driver.aoss_calls[0][0] == 'community_name'
