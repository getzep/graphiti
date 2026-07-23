from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.search_ops import _build_falkor_fulltext_query
from graphiti_core.driver.neo4j.operations.search_ops import (
    Neo4jSearchOperations,
    _build_neo4j_fulltext_query,
)
from graphiti_core.errors import GroupIdValidationError, NodeLabelValidationError
from graphiti_core.helpers import get_default_group_id, validate_group_id
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)
from graphiti_core.search.search_utils import fulltext_query


class RecordingExecutor:
    def __init__(self):
        self.cypher = ''

    async def execute_query(self, cypher, **kwargs):
        self.cypher = cypher
        return [], None, None


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

    with pytest.raises(
        NodeLabelValidationError, match='node_labels must start with a letter or underscore'
    ):
        node_search_filter_query_constructor(filters, GraphProvider.NEO4J)


def test_edge_search_filter_constructor_rejects_unsafe_labels_bypassing_pydantic():
    filters = SearchFilters.model_construct(node_labels=['Entity`) DETACH DELETE x //'])

    with pytest.raises(
        NodeLabelValidationError, match='node_labels must start with a letter or underscore'
    ):
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


def test_falkordb_fulltext_query_strips_backticks():
    """Backtick characters should be sanitized to prevent RediSearch syntax errors."""
    from graphiti_core.driver.falkordb.operations.search_ops import _build_falkor_fulltext_query

    result = _build_falkor_fulltext_query('`add_episode`', ['group1'])
    # Backticks should be removed; remaining tokens joined with OR
    assert '`' not in result
    assert result != ''


def test_falkordb_fulltext_query_returns_empty_on_stopwords_only():
    """When all tokens are stopwords, return empty string instead of malformed query."""
    from graphiti_core.driver.falkordb.operations.search_ops import _build_falkor_fulltext_query

    result = _build_falkor_fulltext_query('the and is', ['group1'])
    assert result == ''


def test_falkordb_fulltext_query_returns_empty_on_punctuation_only():
    """When input is all special characters, return empty string."""
    from graphiti_core.driver.falkordb.operations.search_ops import _build_falkor_fulltext_query

    result = _build_falkor_fulltext_query('!!!...???', ['group1'])
    assert result == ''


def test_falkordb_driver_build_fulltext_query_strips_backticks():
    """FalkorDriver.build_fulltext_query should also strip backticks."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    driver = MagicMock(spec=FalkorDriver)
    driver.sanitize = FalkorDriver.sanitize.__get__(driver, FalkorDriver)

    result = FalkorDriver.build_fulltext_query(driver, '`add_episode`', ['group1'])
    assert '`' not in result
    assert result != ''


def test_falkordb_driver_build_fulltext_query_returns_empty_on_stopwords_only():
    """FalkorDriver.build_fulltext_query returns empty on stopword-only input."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    driver = MagicMock(spec=FalkorDriver)
    driver.sanitize = FalkorDriver.sanitize.__get__(driver, FalkorDriver)

    result = FalkorDriver.build_fulltext_query(driver, 'the and is', ['group1'])
    assert result == ''


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


def test_falkordb_default_group_id_passes_validation():
    # Regression: the FalkorDB default group_id must satisfy validate_group_id.
    # Otherwise add_episode/search with the default group_id raises
    # GroupIdValidationError once search re-validates the group_ids.
    default = get_default_group_id(GraphProvider.FALKORDB)
    assert validate_group_id(default) is True


def test_falkordb_fulltext_query_escapes_default_group_id():
    # The default group_id ('_') must be escaped for RediSearch (-> '\\_').
    # An unescaped '_' produces a RediSearch fulltext syntax error.
    default = get_default_group_id(GraphProvider.FALKORDB)
    built = _build_falkor_fulltext_query('hello', [default])
    assert '@group_id:"\\_"' in built
    assert '@group_id:"_"' not in built


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('method_name', 'expected_guard'),
    [
        (
            'node_similarity_search',
            'n.name_embedding IS NOT NULL AND size(n.name_embedding) = size($search_vector)',
        ),
        (
            'edge_similarity_search',
            'e.fact_embedding IS NOT NULL AND size(e.fact_embedding) = size($search_vector)',
        ),
        (
            'community_similarity_search',
            'c.name_embedding IS NOT NULL AND size(c.name_embedding) = size($search_vector)',
        ),
    ],
)
async def test_neo4j_similarity_search_filters_invalid_embeddings_before_cosine(
    method_name, expected_guard
):
    executor = RecordingExecutor()
    operations = Neo4jSearchOperations()
    method = getattr(operations, method_name)

    if method_name == 'edge_similarity_search':
        await method(executor, [0.1, 0.2], None, None, SearchFilters(), group_ids=['tenant'])
    elif method_name == 'community_similarity_search':
        await method(executor, [0.1, 0.2], group_ids=['tenant'])
    else:
        await method(executor, [0.1, 0.2], SearchFilters(), group_ids=['tenant'])

    cosine_index = executor.cypher.index('vector.similarity.cosine')
    guard_index = executor.cypher.index(expected_guard)
    assert guard_index < cosine_index
