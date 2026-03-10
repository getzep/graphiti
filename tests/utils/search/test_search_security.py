from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neo4j.operations.search_ops import _build_neo4j_fulltext_query
from graphiti_core.errors import GroupIdValidationError, NodeLabelValidationError
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)
from graphiti_core.search.search_utils import fulltext_query


def test_search_filters_reject_unsafe_node_labels():
    with pytest.raises(ValidationError, match='node_labels must start with a letter or underscore'):
        SearchFilters(node_labels=['Entity`) WITH n MATCH (x) DETACH DELETE x //'])


def test_node_search_filter_constructor_keeps_valid_label_expression():
    filters = SearchFilters(node_labels=['Person', 'Organization'])

    filter_queries, filter_params = node_search_filter_query_constructor(
        filters, GraphProvider.NEO4J
    )

    assert filter_queries == ['n:Person|Organization']
    assert filter_params == {}


def test_node_search_filter_constructor_rejects_unsafe_labels_bypassing_pydantic():
    filters = SearchFilters.model_construct(node_labels=['Entity`) DETACH DELETE x //'])

    with pytest.raises(NodeLabelValidationError, match='node_labels must start with a letter or underscore'):
        node_search_filter_query_constructor(filters, GraphProvider.NEO4J)


def test_edge_search_filter_constructor_rejects_unsafe_labels_bypassing_pydantic():
    filters = SearchFilters.model_construct(node_labels=['Entity`) DETACH DELETE x //'])

    with pytest.raises(NodeLabelValidationError, match='node_labels must start with a letter or underscore'):
        edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)


def test_fulltext_query_rejects_invalid_group_ids():
    driver = SimpleNamespace(provider=GraphProvider.NEO4J, fulltext_syntax='')

    with pytest.raises(GroupIdValidationError, match='must contain only alphanumeric'):
        fulltext_query('test', ['bad"group'], driver)


def test_build_neo4j_fulltext_query_rejects_invalid_group_ids():
    with pytest.raises(GroupIdValidationError, match='must contain only alphanumeric'):
        _build_neo4j_fulltext_query('test', ['bad"group'])


def test_falkordb_fulltext_query_rejects_invalid_group_ids():
    # Import inside the test so collection still works when FalkorDB extras are unavailable.
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    driver = MagicMock(spec=FalkorDriver)
    driver.sanitize.return_value = 'test'

    with pytest.raises(GroupIdValidationError, match='must contain only alphanumeric'):
        FalkorDriver.build_fulltext_query(driver, 'test', ['bad"group'])


@pytest.mark.asyncio
async def test_shared_search_rejects_invalid_group_ids():
    clients = SimpleNamespace(
        driver=SimpleNamespace(),
        embedder=SimpleNamespace(),
        cross_encoder=SimpleNamespace(),
    )

    with pytest.raises(GroupIdValidationError, match='must contain only alphanumeric'):
        await search(
            clients,
            query='test',
            group_ids=['bad"group'],
            config=SearchConfig(),
            search_filter=SearchFilters(),
        )
