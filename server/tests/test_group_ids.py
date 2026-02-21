import re

import pytest

from graph_service.group_ids import resolve_group_id


def test_resolve_group_id_is_deterministic():
    assert resolve_group_id('user', 'github_login:octocat') == resolve_group_id(
        'user', 'github_login:octocat'
    )


def test_resolve_group_id_changes_with_scope():
    assert resolve_group_id('user', 'k') != resolve_group_id('workspace', 'k')


def test_resolve_group_id_matches_group_id_charset():
    group_id = resolve_group_id('user', 'github_login:octocat')
    assert re.match(r'^[a-zA-Z0-9_-]+$', group_id)


def test_resolve_group_id_rejects_empty_key():
    with pytest.raises(ValueError):
        resolve_group_id('user', '')
