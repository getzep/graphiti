"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Hermetic tests for the Spanner rquery builder.

import pytest

from graphiti_core.driver.spanner.fulltext import (
    MAX_QUERY_LENGTH,
    build_fulltext_query,
    fulltext_terms,
)
from graphiti_core.driver.spanner.operations.search_ops import SpannerSearchOperations


def test_joins_terms_with_or():
    assert build_fulltext_query('alice founder acme') == 'alice OR founder OR acme'


def test_single_term():
    assert build_fulltext_query('alice') == 'alice'


@pytest.mark.parametrize(
    ('query', 'expected'),
    [
        ('Alice (founder) of "Acme"!', 'alice OR founder OR of OR acme'),
        ('c++ && java || go', 'c OR java OR go'),
        ('name:alice AND -bob', 'name OR alice OR and OR bob'),
        (
            'a/b\\c `d` [e] {f} <g> ~h ^i *j ?k',
            'a OR b OR c OR d OR e OR f OR g OR h OR i OR j OR k',
        ),
        ("it's $5, 100% @home #1; x=y | z.", 'it OR s OR 5 OR 100 OR home OR 1 OR x OR y OR z'),
        ('snake_case stays', 'snake_case OR stays'),
        ('nul\x00byte', 'nul OR byte'),
    ],
)
def test_strips_special_characters(query: str, expected: str):
    assert build_fulltext_query(query) == expected


def test_rquery_operators_in_text_are_lowercased():
    # Upper-case OR/AND/AROUND would be parsed as rquery operators.
    assert build_fulltext_query('cats OR dogs AND AROUND') == 'cats OR or OR dogs OR and OR around'


def test_deduplicates_case_insensitively():
    assert fulltext_terms('Alice alice ALICE bob') == ['alice', 'bob']


@pytest.mark.parametrize('query', ['', '   ', '()[]{}"!?', '- + | & ~'])
def test_empty_after_sanitising(query: str):
    assert build_fulltext_query(query) == ''


def test_respects_max_query_length():
    words = [f'word{i:03d}' for i in range(100)]
    query = ' '.join(words)
    full = ' OR '.join(words)
    for limit in (7, 20, 64, MAX_QUERY_LENGTH, 500):
        result = build_fulltext_query(query, max_query_length=limit)
        assert len(result) <= limit
        assert full.startswith(result)
        # Only whole terms are kept, and as many as fit.
        kept = result.split(' OR ')
        assert kept == words[: len(kept)]
        assert len(' OR '.join(words[: len(kept) + 1])) > limit
    assert build_fulltext_query('abcdefghij', max_query_length=5) == ''


def test_group_ids_do_not_change_the_query():
    assert build_fulltext_query('alice bob', ['g1', 'g2']) == build_fulltext_query('alice bob')


def test_search_ops_build_fulltext_query_delegates():
    ops = SpannerSearchOperations()
    assert ops.build_fulltext_query('Alice & Bob', ['g'], 128) == 'alice OR bob'
