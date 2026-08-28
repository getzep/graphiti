from types import SimpleNamespace

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_utils import fulltext_query


def _neo4j_driver():
    return SimpleNamespace(provider=GraphProvider.NEO4J, fulltext_syntax='')


def test_fulltext_query_groups_multiple_group_ids_before_query_clause():
    result = fulltext_query('hello world', ['group1', 'group2', 'group3'], _neo4j_driver())

    assert result == '(group_id:"group1" OR group_id:"group2" OR group_id:"group3") AND (hello world)'


def test_fulltext_query_keeps_single_group_filter_semantics():
    result = fulltext_query('hello world', ['group1'], _neo4j_driver())

    assert result == '(group_id:"group1") AND (hello world)'


def test_fulltext_query_without_group_ids_is_unchanged():
    result = fulltext_query('hello world', None, _neo4j_driver())

    assert result == '(hello world)'
