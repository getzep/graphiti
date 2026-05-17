"""Tests for search_filters.py — node/edge label filter query construction."""

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)


# ── node_search_filter_query_constructor ──────────────────────────────


class TestNodeSearchFilterSingleLabel:
    """Single-label filter should work identically across providers."""

    @pytest.mark.parametrize(
        "provider",
        [GraphProvider.NEO4J, GraphProvider.FALKORDB, GraphProvider.NEPTUNE],
    )
    def test_single_label(self, provider: GraphProvider):
        filters = SearchFilters(node_labels=["Aircraft"])
        queries, _ = node_search_filter_query_constructor(filters, provider)
        assert len(queries) == 1
        assert "Aircraft" in queries[0]

    def test_single_label_kuzu(self):
        filters = SearchFilters(node_labels=["Aircraft"])
        queries, params = node_search_filter_query_constructor(filters, GraphProvider.KUZU)
        assert queries == ["list_has_all(n.labels, $labels)"]
        assert params["labels"] == ["Aircraft"]


class TestNodeSearchFilterMultiLabel:
    """Multi-label filter must use provider-appropriate syntax."""

    def test_neo4j_uses_pipe_syntax(self):
        filters = SearchFilters(node_labels=["Occurrence", "Operator"])
        queries, params = node_search_filter_query_constructor(filters, GraphProvider.NEO4J)
        assert queries == ["n:Occurrence|Operator"]
        assert params == {}

    def test_falkordb_uses_or_syntax(self):
        filters = SearchFilters(node_labels=["Occurrence", "Operator"])
        queries, params = node_search_filter_query_constructor(filters, GraphProvider.FALKORDB)
        assert len(queries) == 1
        assert queries[0] == "(n:Occurrence OR n:Operator)"
        assert params == {}

    def test_falkordb_three_labels(self):
        filters = SearchFilters(node_labels=["Aircraft", "Occurrence", "Operator"])
        queries, _ = node_search_filter_query_constructor(filters, GraphProvider.FALKORDB)
        assert queries[0] == "(n:Aircraft OR n:Occurrence OR n:Operator)"

    def test_kuzu_uses_list_has_all(self):
        filters = SearchFilters(node_labels=["Occurrence", "Operator"])
        queries, params = node_search_filter_query_constructor(filters, GraphProvider.KUZU)
        assert queries == ["list_has_all(n.labels, $labels)"]
        assert params["labels"] == ["Occurrence", "Operator"]


class TestNodeSearchFilterEmpty:
    """No node_labels → no filter."""

    def test_none_labels(self):
        filters = SearchFilters(node_labels=None)
        queries, params = node_search_filter_query_constructor(filters, GraphProvider.FALKORDB)
        assert queries == []
        assert params == {}


# ── edge_search_filter_query_constructor ──────────────────────────────


class TestEdgeSearchFilterMultiLabel:
    """Multi-label node filter within edge queries."""

    def test_neo4j_uses_pipe_syntax(self):
        filters = SearchFilters(node_labels=["Occurrence", "Operator"])
        queries, _ = edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)
        assert len(queries) == 1
        assert queries[0] == "n:Occurrence|Operator AND m:Occurrence|Operator"

    def test_falkordb_uses_or_syntax(self):
        filters = SearchFilters(node_labels=["Occurrence", "Operator"])
        queries, params = edge_search_filter_query_constructor(filters, GraphProvider.FALKORDB)
        assert len(queries) == 1
        assert queries[0] == "(n:Occurrence OR n:Operator) AND (m:Occurrence OR m:Operator)"
        assert params == {}

    def test_falkordb_three_labels(self):
        filters = SearchFilters(node_labels=["Aircraft", "Occurrence", "Operator"])
        queries, _ = edge_search_filter_query_constructor(filters, GraphProvider.FALKORDB)
        assert queries[0] == (
            "(n:Aircraft OR n:Occurrence OR n:Operator) AND "
            "(m:Aircraft OR m:Occurrence OR m:Operator)"
        )

    def test_kuzu_uses_list_has_all(self):
        filters = SearchFilters(node_labels=["Occurrence", "Operator"])
        queries, params = edge_search_filter_query_constructor(filters, GraphProvider.KUZU)
        assert "list_has_all" in queries[0]
        assert params["labels"] == ["Occurrence", "Operator"]


class TestEdgeSearchFilterSingleLabel:
    """Single-label node filter in edge queries works across providers."""

    @pytest.mark.parametrize(
        "provider",
        [GraphProvider.NEO4J, GraphProvider.FALKORDB, GraphProvider.NEPTUNE],
    )
    def test_single_label(self, provider: GraphProvider):
        filters = SearchFilters(node_labels=["Aircraft"])
        queries, _ = edge_search_filter_query_constructor(filters, provider)
        assert len(queries) == 1
        assert "Aircraft" in queries[0]


class TestEdgeSearchFilterEdgeTypes:
    """edge_types filter is provider-agnostic."""

    def test_edge_types_filter(self):
        filters = SearchFilters(edge_types=["OPERATED_BY", "LOCATED_IN"])
        queries, params = edge_search_filter_query_constructor(filters, GraphProvider.FALKORDB)
        assert "e.name in $edge_types" in queries
        assert params["edge_types"] == ["OPERATED_BY", "LOCATED_IN"]
