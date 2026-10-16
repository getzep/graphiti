from datetime import datetime, timedelta

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
    edge_search_filter_query_constructor,
)

NOW = datetime(2026, 10, 6)
D_OLD = NOW - timedelta(days=10)
D_MID = NOW - timedelta(days=5)
D_NEW = NOW - timedelta(days=1)


@pytest.mark.parametrize(
    'field',
    ['valid_at', 'invalid_at', 'created_at', 'expired_at'],
)
def test_edge_temporal_filter_does_not_collide_across_or_groups(field):
    """Two OR groups sharing inner index 0 must not overwrite each other's params.

    Regression: each group's `j` was used as the parameter suffix on its own,
    so group 1's `j=0` overwrote group 0's `j=0` in ``filter_params``. The
    first OR branch then evaluated against the wrong date, silently dropping
    results.
    """
    filters = SearchFilters.model_construct(
        **{
            field: [
                [
                    DateFilter(date=D_OLD, comparison_operator=ComparisonOperator.greater_than),
                    DateFilter(date=D_MID, comparison_operator=ComparisonOperator.less_than),
                ],
                [
                    DateFilter(date=D_NEW, comparison_operator=ComparisonOperator.greater_than),
                ],
            ]
        }
    )

    queries, params = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)
    query_text = queries[0]

    # Group 0's first condition must bind to D_OLD, group 1's only condition to D_NEW.
    # Distinct parameter keys per (group, inner index):
    key_g0_j0 = f'{field}_0_0'
    key_g1_j0 = f'{field}_1_0'
    assert key_g0_j0 in params, f'missing {key_g0_j0} in {params}'
    assert key_g1_j0 in params, f'missing {key_g1_j0} in {params}'
    assert params[key_g0_j0] == D_OLD, f'{key_g0_j0} should be D_OLD, got {params[key_g0_j0]}'
    assert params[key_g1_j0] == D_NEW, f'{key_g1_j0} should be D_NEW, got {params[key_g1_j0]}'

    # The query text must reference both distinct placeholders.
    assert f'${key_g0_j0}' in query_text, f'${key_g0_j0} missing in query: {query_text}'
    assert f'${key_g1_j0}' in query_text, f'${key_g1_j0} missing in query: {query_text}'


@pytest.mark.parametrize(
    'field',
    ['valid_at', 'invalid_at', 'created_at', 'expired_at'],
)
def test_edge_temporal_filter_single_or_group_keeps_predictable_keys(field):
    """A single OR group must still produce a coherent query and params map.

    Backward-compat guard: the internal parameter key format may change, but
    every placeholder in the emitted query must have a matching entry in params.
    """
    filters = SearchFilters.model_construct(
        **{
            field: [
                [
                    DateFilter(
                        date=D_OLD, comparison_operator=ComparisonOperator.greater_than_equal
                    ),
                    DateFilter(date=D_NEW, comparison_operator=ComparisonOperator.less_than_equal),
                ],
            ]
        }
    )

    queries, params = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)
    query_text = queries[0]

    # Every $<key> referenced in the query must exist in params.
    import re

    referenced = set(re.findall(r'\$(\w+)', query_text))
    missing = referenced - set(params.keys())
    assert not missing, f'query references params not provided: {missing}; query={query_text}'
    assert params, 'expected at least one bound parameter'
