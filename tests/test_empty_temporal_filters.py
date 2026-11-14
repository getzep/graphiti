from datetime import datetime, timezone

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
    edge_search_filter_query_constructor,
)


@pytest.mark.parametrize(
    'field',
    ['valid_at', 'invalid_at', 'created_at', 'expired_at'],
)
@pytest.mark.parametrize(
    'value',
    [[], [[]]],
    ids=['empty-outer-list', 'empty-inner-group'],
)
@pytest.mark.parametrize(
    'provider',
    list(GraphProvider),
)
def test_empty_temporal_filters_do_not_generate_predicates(field, value, provider):
    filters = SearchFilters(**{field: value})

    queries, params = edge_search_filter_query_constructor(filters, provider)

    assert queries == []
    assert params == {}


@pytest.mark.parametrize(
    'field',
    ['valid_at', 'invalid_at', 'created_at', 'expired_at'],
)
@pytest.mark.parametrize('provider', list(GraphProvider))
@pytest.mark.parametrize(
    'placement',
    ['before', 'after', 'both'],
)
def test_mixed_empty_temporal_groups(field, provider, placement):
    """Empty groups must not corrupt populated filters."""

    populated_group = [
        DateFilter(
            date=datetime(2026, 1, 11, tzinfo=timezone.utc),
            comparison_operator=(ComparisonOperator.greater_than),
        )
    ]

    control = SearchFilters(**{field: [populated_group]})

    expected_queries, expected_params = edge_search_filter_query_constructor(control, provider)

    if placement == 'before':
        groups = [[], populated_group]
    elif placement == 'after':
        groups = [populated_group, []]
    else:
        groups = [[], populated_group, []]

    filters = SearchFilters(**{field: groups})

    actual_queries, actual_params = edge_search_filter_query_constructor(filters, provider)

    assert actual_queries == expected_queries
    assert actual_params == expected_params
    assert len(actual_queries) == 1
    assert actual_params
