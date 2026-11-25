"""SearchFilters.property_filters is declared but never read.

A caller who sets it expects a WHERE clause and silently gets an unfiltered
search — every row, regardless of the property they filtered on. For a
multi-tenant caller filtering on a tenant id, that is a cross-tenant read.

These tests pin the contract for both constructors.
"""

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    PropertyFilter,
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

CONSTRUCTORS = [
    pytest.param(node_search_filter_query_constructor, 'n', id='node'),
    pytest.param(edge_search_filter_query_constructor, 'e', id='edge'),
]


def _sf(**kwargs) -> SearchFilters:
    return SearchFilters(property_filters=[PropertyFilter(**kwargs)])


@pytest.mark.parametrize('construct,alias', CONSTRUCTORS)
def test_equality_filter_becomes_a_predicate(construct, alias):
    queries, params = construct(
        _sf(
            property_name='index_org_id',
            property_value='org_1',
            comparison_operator=ComparisonOperator.equals,
        ),
        GraphProvider.FALKORDB,
    )

    assert any(f'{alias}.index_org_id' in q for q in queries)
    assert 'org_1' in params.values()


@pytest.mark.parametrize('construct,alias', CONSTRUCTORS)
def test_value_is_bound_not_interpolated(construct, alias):
    """A value spliced into the query text is an injection primitive."""
    queries, params = construct(
        _sf(
            property_name='index_org_id',
            property_value="' OR 1=1 --",
            comparison_operator=ComparisonOperator.equals,
        ),
        GraphProvider.FALKORDB,
    )

    assert not any('OR 1=1' in q for q in queries)
    assert "' OR 1=1 --" in params.values()


@pytest.mark.parametrize('construct,alias', CONSTRUCTORS)
def test_is_null_binds_no_parameter(construct, alias):
    """IS NULL takes no right-hand operand; binding one is a syntax error."""
    queries, params = construct(
        _sf(
            property_name='archived_at',
            property_value=None,
            comparison_operator=ComparisonOperator.is_null,
        ),
        GraphProvider.FALKORDB,
    )

    assert any('IS NULL' in q for q in queries)
    assert params == {}


@pytest.mark.parametrize('construct,alias', CONSTRUCTORS)
def test_is_not_null_binds_no_parameter(construct, alias):
    queries, params = construct(
        _sf(
            property_name='archived_at',
            property_value=None,
            comparison_operator=ComparisonOperator.is_not_null,
        ),
        GraphProvider.FALKORDB,
    )

    assert any('IS NOT NULL' in q for q in queries)
    assert params == {}


@pytest.mark.parametrize('construct,alias', CONSTRUCTORS)
def test_multiple_filters_all_applied_without_collision(construct, alias):
    """Two filters must each survive — a shared parameter name silently drops one."""
    filters = SearchFilters(
        property_filters=[
            PropertyFilter(
                property_name='index_org_id',
                property_value='o1',
                comparison_operator=ComparisonOperator.equals,
            ),
            PropertyFilter(
                property_name='index_layer',
                property_value='org',
                comparison_operator=ComparisonOperator.equals,
            ),
        ]
    )
    queries, params = construct(filters, GraphProvider.FALKORDB)

    assert any(f'{alias}.index_org_id' in q for q in queries)
    assert any(f'{alias}.index_layer' in q for q in queries)
    assert sorted(params.values()) == ['o1', 'org']


@pytest.mark.parametrize('construct,alias', CONSTRUCTORS)
def test_same_property_twice_keeps_both_values(construct, alias):
    """A range (x > 1 AND x < 9) reuses one property name across two filters."""
    filters = SearchFilters(
        property_filters=[
            PropertyFilter(
                property_name='score',
                property_value=1,
                comparison_operator=ComparisonOperator.greater_than,
            ),
            PropertyFilter(
                property_name='score',
                property_value=9,
                comparison_operator=ComparisonOperator.less_than,
            ),
        ]
    )
    _, params = construct(filters, GraphProvider.FALKORDB)

    assert sorted(params.values()) == [1, 9]


@pytest.mark.parametrize('construct,alias', CONSTRUCTORS)
@pytest.mark.parametrize(
    'operator,expected',
    [
        (ComparisonOperator.equals, '='),
        (ComparisonOperator.not_equals, '<>'),
        (ComparisonOperator.greater_than, '>'),
        (ComparisonOperator.less_than, '<'),
        (ComparisonOperator.greater_than_equal, '>='),
        (ComparisonOperator.less_than_equal, '<='),
    ],
)
def test_each_comparison_operator_is_emitted(construct, alias, operator, expected):
    queries, _ = construct(
        _sf(property_name='score', property_value=5, comparison_operator=operator),
        GraphProvider.FALKORDB,
    )

    assert any(expected in q for q in queries)


@pytest.mark.parametrize('construct,alias', CONSTRUCTORS)
def test_absent_property_filters_changes_nothing(construct, alias):
    """Additive by construction: callers that never set the field are untouched."""
    assert construct(SearchFilters(), GraphProvider.FALKORDB) == construct(
        SearchFilters(property_filters=None), GraphProvider.FALKORDB
    )


@pytest.mark.parametrize('construct,alias', CONSTRUCTORS)
def test_property_filters_compose_with_existing_filters(construct, alias):
    """They must ADD to the other filters, not replace them."""
    plain = SearchFilters(node_labels=['Entity'], edge_types=['RELATES_TO'])
    combined = SearchFilters(
        node_labels=['Entity'],
        edge_types=['RELATES_TO'],
        property_filters=[
            PropertyFilter(
                property_name='index_org_id',
                property_value='o1',
                comparison_operator=ComparisonOperator.equals,
            )
        ],
    )

    plain_queries, _ = construct(plain, GraphProvider.FALKORDB)
    combined_queries, _ = construct(combined, GraphProvider.FALKORDB)

    assert len(combined_queries) == len(plain_queries) + 1
    for q in plain_queries:
        assert q in combined_queries
