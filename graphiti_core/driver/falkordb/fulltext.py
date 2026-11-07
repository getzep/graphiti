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


def _group_id_phrase(group_id: str) -> str:
    """Quote a validated group ID as the phrase RediSearch indexed it as.

    RediSearch tokenizes TEXT fields on separators, so a stored group_id such as
    ``ko-verify`` is the token sequence ``ko verify``. Escaping the hyphen
    (``"ko\\-verify"``) matched nothing; the phrase ``"ko verify"`` does. Characters that
    are not separators but still special to the query syntax - the default group id ``_``
    above all - do need escaping (``"\\_"``), or the query silently matches nothing.
    """
    phrase = sanitize_falkor_fulltext_query(group_id)
    # Escape only what RediSearch treats specially; word characters of any script pass through
    # (an escaped Hangul syllable such as \\한 no longer matches the indexed token).
    phrase = re.sub(r'([^\w ]|_)', r'\\\1', phrase)
    return f'"{phrase}"'


def build_falkor_fulltext_query(
    query: str,
    group_ids: list[str] | None = None,
    max_query_length: int = MAX_QUERY_LENGTH,
) -> str:
    """Build a FalkorDB RedisSearch fulltext query."""
    validate_group_ids(group_ids)

    group_filter = ''
    if group_ids:
        phrases = [_group_id_phrase(group_id) for group_id in group_ids if group_id.strip()]
        group_filter = f'(@group_id:{"|".join(phrases)})' if phrases else ''

    filtered_words = [
        word
        for word in sanitize_falkor_fulltext_query(query).split()
        if word.lower() not in STOPWORDS
    ]
    if not filtered_words:
        return ''

    sanitized_query = ' | '.join(filtered_words)
    if len(sanitized_query.split(' ')) + len(group_ids or []) >= max_query_length:
        return ''

    return f'{group_filter} ({sanitized_query})'
