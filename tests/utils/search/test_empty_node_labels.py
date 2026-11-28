"""An unselected label filter must not produce an empty Cypher label expression."""

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)


@pytest.mark.parametrize('provider', list(GraphProvider))
@pytest.mark.parametrize(
    'constructor',
    [node_search_filter_query_constructor, edge_search_filter_query_constructor],
)
def test_empty_node_labels_match_absent_label_filter(provider, constructor):
    assert constructor(SearchFilters(node_labels=[]), provider) == constructor(
        SearchFilters(), provider
    )


@pytest.mark.parametrize('provider', list(GraphProvider))
def test_empty_node_labels_preserve_other_edge_filters(provider):
    filters = SearchFilters(node_labels=[], edge_types=['WORKS_AT'], edge_uuids=['edge-1'])

    queries, params = edge_search_filter_query_constructor(filters, provider)

    assert queries == ['e.name in $edge_types', 'e.uuid in $edge_uuids']
    assert params == {'edge_types': ['WORKS_AT'], 'edge_uuids': ['edge-1']}


@pytest.mark.parametrize('provider', list(GraphProvider))
def test_populated_node_labels_still_filter(provider):
    filters = SearchFilters(node_labels=['Person'])

    node_queries, node_params = node_search_filter_query_constructor(filters, provider)
    edge_queries, edge_params = edge_search_filter_query_constructor(filters, provider)

    if provider == GraphProvider.KUZU:
        assert node_queries == ['list_has_all(n.labels, $labels)']
        assert edge_queries == [
            'list_has_all(n.labels, $labels) AND list_has_all(m.labels, $labels)'
        ]
        assert node_params == edge_params == {'labels': ['Person']}
    else:
        assert node_queries == ['n:Person']
        assert edge_queries == ['n:Person AND m:Person']
        assert node_params == edge_params == {}
