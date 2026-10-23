from datetime import datetime, timezone

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
    edge_search_filter_query_constructor,
)


@pytest.mark.parametrize('field_name', ['valid_at', 'invalid_at', 'created_at', 'expired_at'])
def test_date_filter_params_are_unique_across_or_groups(field_name):
    earlier = datetime(2025, 1, 1, tzinfo=timezone.utc)
    later = datetime(2025, 2, 1, tzinfo=timezone.utc)
    filters = SearchFilters(
        **{
            field_name: [
                [
                    DateFilter(
                        date=earlier,
                        comparison_operator=ComparisonOperator.greater_than_equal,
                    )
                ],
                [
                    DateFilter(
                        date=later,
                        comparison_operator=ComparisonOperator.less_than,
                    )
                ],
                [DateFilter(comparison_operator=ComparisonOperator.is_null)],
                [DateFilter(comparison_operator=ComparisonOperator.is_not_null)],
            ]
        }
    )

    filter_queries, filter_params = edge_search_filter_query_constructor(
        filters, GraphProvider.NEO4J
    )

    assert len(filter_queries) == 1
    assert len(filter_params) == 2
    assert set(filter_params.values()) == {earlier, later}

    query = filter_queries[0]
    earlier_param = next(name for name, value in filter_params.items() if value == earlier)
    later_param = next(name for name, value in filter_params.items() if value == later)
    assert f'e.{field_name} >= ${earlier_param}' in query
    assert f'e.{field_name} < ${later_param}' in query
    assert f'e.{field_name} IS NULL' in query
    assert f'e.{field_name} IS NOT NULL' in query
