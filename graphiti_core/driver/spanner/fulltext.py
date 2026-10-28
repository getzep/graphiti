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

# Graphiti search text -> Spanner ``rquery`` for ``SEARCH(search_tokens, @q)``.
#
# rquery joins bare terms with AND by default; Graphiti's Lucene queries behave like OR, so
# terms are joined with ``OR`` (verified on the emulator). Group filtering is NOT part of the
# query string: the search code filters with predicates on the source columns, which the
# search indexes cover through their ``TOKEN()`` columns. Verified on the emulator:
#
#     WHERE SEARCH(search_tokens, @q)
#       AND group_id IN UNNEST(@group_ids)          -- covered by group_tokens
#       AND ARRAY_INCLUDES_ANY(labels, @labels)     -- covered by label_tokens (Entity only)
#
# ``ARRAY_INCLUDES_ANY(group_tokens, ...)`` is rejected ("expected array type but found
# TOKENLIST"), and ``SEARCH()`` only accepts TOKENIZE_FULLTEXT columns.

from __future__ import annotations

MAX_QUERY_LENGTH = 128

# Union of the characters Graphiti strips or escapes for Lucene (helpers.lucene_sanitize) and
# FalkorDB (driver/falkordb/fulltext.py). In rquery, '-' negates, '"' quotes a phrase, '|' and
# parentheses are operators; none of them should come from user text.
_SPECIAL_CHARACTERS = ',.<>{}[]"\':;!@#$%^&*()-+=~?|/\\`'
_TRANSLATION = str.maketrans(dict.fromkeys(_SPECIAL_CHARACTERS + '\x00', ' '))


def fulltext_terms(query: str) -> list[str]:
    """Sanitised, lower-cased, de-duplicated search terms in their original order.

    Lower-casing is lossless (``TOKENIZE_FULLTEXT`` is case-insensitive) and stops words such
    as ``OR``, ``AND`` or ``AROUND`` from being parsed as rquery operators.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for raw in query.translate(_TRANSLATION).split():
        term = raw.lower()
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def build_fulltext_query(
    query: str,
    group_ids: list[str] | None = None,
    max_query_length: int = MAX_QUERY_LENGTH,
) -> str:
    """Build an rquery string: sanitised terms joined with ``' OR '``.

    Terms are added in order while the result stays within ``max_query_length`` characters;
    later terms are dropped. Returns ``''`` when no term remains (callers skip the search).
    ``group_ids`` is accepted for interface parity and ignored; see the module comment.
    """
    del group_ids  # group filtering happens in SQL, not in the rquery string
    result = ''
    for term in fulltext_terms(query):
        candidate = term if not result else f'{result} OR {term}'
        if len(candidate) > max_query_length:
            break
        result = candidate
    return result
