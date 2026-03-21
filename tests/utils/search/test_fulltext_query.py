from unittest.mock import Mock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_utils import fulltext_query


@pytest.fixture
def neo4j_driver():
    driver = Mock()
    driver.provider = GraphProvider.NEO4J
    driver.fulltext_syntax = ''
    return driver


class TestFulltextQueryGroupIds:
    """Regression tests for issue #1249: BM25 query must wrap group_id OR
    clauses in parentheses so that all group_ids are applied as a filter."""

    def test_single_group_id(self, neo4j_driver):
        result = fulltext_query('alice', ['g1'], neo4j_driver)
        assert result.startswith('(group_id:"g1") AND ')

    def test_multiple_group_ids_wrapped_in_parentheses(self, neo4j_driver):
        result = fulltext_query('alice', ['g1', 'g2', 'g3'], neo4j_driver)
        # The OR clause must be wrapped so AND binds correctly
        assert result.startswith('(group_id:"g1" OR group_id:"g2" OR group_id:"g3") AND ')

    def test_no_group_ids(self, neo4j_driver):
        result = fulltext_query('alice', None, neo4j_driver)
        # No group filter prefix
        assert not result.startswith('(group_id')
        assert '(alice)' in result

    def test_empty_group_ids(self, neo4j_driver):
        result = fulltext_query('alice', [], neo4j_driver)
        assert not result.startswith('(group_id')
