import pytest

from graphiti_core.driver.falkordb.fulltext import (
    build_falkor_fulltext_query,
    sanitize_falkor_fulltext_query,
)


@pytest.mark.parametrize('token', ['_', '__', '___'])
def test_underscore_only_query_returns_empty(token):
    assert build_falkor_fulltext_query(token) == ''


def test_standalone_underscore_is_removed_from_path_query():
    query = 'sessions/<enc-cwd>/<ISO-ts>_<uuid>.jsonl'
    built = build_falkor_fulltext_query(query)

    assert built
    assert ' | _ | ' not in built
    assert ' | _)' not in built


@pytest.mark.parametrize('token', ['a_b', 'session_jsonl', 'a_b_c'])
def test_internal_underscores_are_preserved(token):
    sanitized = sanitize_falkor_fulltext_query(token)

    assert sanitized == token
    assert token in build_falkor_fulltext_query(token)
