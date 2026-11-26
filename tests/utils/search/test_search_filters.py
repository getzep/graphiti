"""Unit tests for search filter query construction."""

from datetime import datetime, timezone

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
    edge_search_filter_query_constructor,
)

TEMPORAL_FIELDS = ('valid_at', 'invalid_at', 'created_at', 'expired_at')
PROVIDERS = (
    GraphProvider.NEO4J,
    GraphProvider.FALKORDB,
    GraphProvider.NEPTUNE,
    GraphProvider.RELATE,
)


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize('field', TEMPORAL_FIELDS)
@pytest.mark.parametrize('empty_value', [[], [[]], [[], []]])
def test_edge_search_filter_empty_temporal_groups_produce_no_predicate(
    provider: GraphProvider,
    field: str,
    empty_value: list[list[DateFilter]],
):
    filters = SearchFilters(**{field: empty_value})

    filter_queries, filter_params = edge_search_filter_query_constructor(filters, provider)

    assert filter_queries == []
    assert filter_params == {}


@pytest.mark.parametrize('provider', PROVIDERS)
def test_edge_search_filter_all_empty_temporal_fields_with_other_filters(provider: GraphProvider):
    filters = SearchFilters(
        edge_types=['WORKS_AT'],
        valid_at=[],
        invalid_at=[[]],
        created_at=[[], []],
        expired_at=[],
    )

    filter_queries, filter_params = edge_search_filter_query_constructor(filters, provider)

    assert filter_queries == ['e.name in $edge_types']
    assert filter_params == {'edge_types': ['WORKS_AT']}


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize('field', TEMPORAL_FIELDS)
@pytest.mark.parametrize('layout', ['leading_empty', 'trailing_empty', 'surrounding_empty'])
def test_edge_search_filter_mixed_empty_and_nonempty_temporal_groups(
    provider: GraphProvider,
    field: str,
    layout: str,
):
    dt = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
    date_filter = DateFilter(date=dt, comparison_operator=ComparisonOperator.greater_than)

    if layout == 'leading_empty':
        groups: list[list[DateFilter]] = [[], [date_filter]]
    elif layout == 'trailing_empty':
        groups = [[date_filter], []]
    else:
        groups = [[], [date_filter], []]

    filters = SearchFilters(**{field: groups})

    filter_queries, filter_params = edge_search_filter_query_constructor(filters, provider)

    assert filter_queries == [f'((e.{field} > ${field}_0))']
    assert filter_params == {f'{field}_0': dt}


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize('field', TEMPORAL_FIELDS)
def test_edge_search_filter_empty_group_between_nonempty_or_groups(
    provider: GraphProvider,
    field: str,
):
    dt = datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc)
    filters = SearchFilters(
        **{
            field: [
                [DateFilter(date=None, comparison_operator=ComparisonOperator.is_null)],
                [],
                [DateFilter(date=dt, comparison_operator=ComparisonOperator.less_than)],
            ]
        }
    )

    filter_queries, filter_params = edge_search_filter_query_constructor(filters, provider)

    assert filter_queries == [f'((e.{field} IS NULL) OR (e.{field} < ${field}_0))']
    assert filter_params == {f'{field}_0': dt}


@pytest.mark.parametrize('provider', (GraphProvider.NEO4J, GraphProvider.FALKORDB))
def test_edge_search_filter_preserves_nonempty_temporal_predicates_and_params(
    provider: GraphProvider,
):
    start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc)
    filters = SearchFilters(
        valid_at=[
            [
                DateFilter(date=start, comparison_operator=ComparisonOperator.greater_than_equal),
                DateFilter(date=end, comparison_operator=ComparisonOperator.less_than_equal),
            ],
            [DateFilter(date=None, comparison_operator=ComparisonOperator.is_null)],
        ],
        invalid_at=[
            [DateFilter(date=None, comparison_operator=ComparisonOperator.is_not_null)],
        ],
    )

    filter_queries, filter_params = edge_search_filter_query_constructor(filters, provider)

    assert filter_queries == [
        '((e.valid_at >= $valid_at_0) AND (e.valid_at <= $valid_at_1) OR (e.valid_at IS NULL))',
        '((e.invalid_at IS NOT NULL))',
    ]
    assert filter_params == {
        'valid_at_0': start,
        'valid_at_1': end,
    }
