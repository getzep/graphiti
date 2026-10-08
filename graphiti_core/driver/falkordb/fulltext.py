"""Shared FalkorDB fulltext-query construction."""

import re

from graphiti_core.driver.falkordb import STOPWORDS
from graphiti_core.helpers import validate_group_ids

MAX_QUERY_LENGTH = 128

# FalkorDB separator characters that break text into tokens.
_SEPARATOR_MAP = str.maketrans(
    {
        ',': ' ',
        '.': ' ',
        '<': ' ',
        '>': ' ',
        '{': ' ',
        '}': ' ',
        '[': ' ',
        ']': ' ',
        '"': ' ',
        "'": ' ',
        ':': ' ',
        ';': ' ',
        '!': ' ',
        '@': ' ',
        '#': ' ',
        '$': ' ',
        '%': ' ',
        '^': ' ',
        '&': ' ',
        '*': ' ',
        '(': ' ',
        ')': ' ',
        '-': ' ',
        '+': ' ',
        '=': ' ',
        '~': ' ',
        '?': ' ',
        '|': ' ',
        '/': ' ',
        '\\': ' ',
        '`': ' ',
    }
)


def sanitize_falkor_fulltext_query(query: str) -> str:
    """Replace FalkorDB special characters with whitespace."""
    return ' '.join(query.translate(_SEPARATOR_MAP).split())


def _escape_fulltext_group_id(group_id: str) -> str:
    """Escape a validated group ID for RediSearch fulltext syntax."""
    return re.sub(r'([^a-zA-Z0-9])', r'\\\1', group_id)


def _group_id_fulltext_term(group_id: str) -> str:
    """Render a group ID as a fulltext term that matches how FalkorDB indexes it.

    FalkorDB tokenizes TEXT fields on separators (hyphen et al.) but keeps
    underscores inside tokens: ``personal-prefs`` indexes as ``personal`` +
    ``prefs`` (phrase query), while ``personal_prefs`` stays one token
    (literal term). Splitting must mirror that tokenizer, or one of the two
    id styles silently stops matching.
    Re-joining the tokens as a phrase (``"personal prefs"``) matches the indexed
    form while keeping token order and adjacency, which stays tighter than an
    OR of loose tokens.
    """
    # FalkorDB's tokenizer treats [a-zA-Z0-9_] as word characters: hyphens and
    # other separators split tokens, underscores do NOT. Split accordingly so a
    # 'graphiti-core' id becomes the phrase "graphiti core" while a
    # 'graphiti_core' id stays the single token "graphiti_core".
    tokens = [t for t in re.split(r'[^a-zA-Z0-9_]+', group_id) if t]
    if not tokens:
        return f'"{_escape_fulltext_group_id(group_id)}"'
    # Escape inside every token: RediSearch rejects unescaped special chars in
    # quoted terms (a bare '_' raises a syntax error) while an escaped word
    # char still tokenizes back to itself and matches the index.
    escaped = [_escape_fulltext_group_id(t) for t in tokens]
    if len(escaped) == 1:
        return f'"{escaped[0]}"'
    return '"' + ' '.join(escaped) + '"'


def build_falkor_fulltext_query(
    query: str,
    group_ids: list[str] | None = None,
    max_query_length: int = MAX_QUERY_LENGTH,
) -> str:
    """Build a FalkorDB RedisSearch fulltext query."""
    validate_group_ids(group_ids)

    group_filter = ''
    if group_ids:
        group_id_terms = [_group_id_fulltext_term(group_id) for group_id in group_ids]
        group_filter = f'(@group_id:{"|".join(group_id_terms)})'

    filtered_words = [
        word
        for word in sanitize_falkor_fulltext_query(query).split()
        # Drop punctuation-only tokens (a bare '_' from identifiers like
        # '<ts>_<uuid>' makes RediSearch reject the whole query — #1820)
        # and stopword tokens.
        if word.lower() not in STOPWORDS and any(c.isalnum() for c in word)
    ]
    if not filtered_words:
        return ''

    sanitized_query = ' | '.join(filtered_words)
    if len(sanitized_query.split(' ')) + len(group_ids or []) >= max_query_length:
        return ''

    return f'{group_filter} ({sanitized_query})'
