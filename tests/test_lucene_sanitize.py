"""
Tests for lucene_sanitize() — verifies that:

1. Lucene special characters are properly backslash-escaped.
2. Lucene boolean keywords (AND, OR, NOT) are escaped only when they appear
   as standalone uppercase words, NOT when they are substrings of other words.
3. Normal text — including words containing the letters O, R, N, T, A, D —
   passes through unmodified.

Regression tests for https://github.com/getzep/graphiti/issues/1302
"""

import pytest

from graphiti_core.helpers import lucene_sanitize


# ---------------------------------------------------------------------------
# 1. Uppercase letters are NOT corrupted
# ---------------------------------------------------------------------------
class TestUppercaseLettersPreserved:
    """The previous implementation escaped individual chars O, R, N, T, A, D
    via str.maketrans, corrupting every word containing those letters."""

    @pytest.mark.parametrize(
        "query",
        [
            "Donald Trump",
            "ORACLE",
            "NODE",
            "Android",
            "Data Science",
            "Toronto",
            "NASA",
            "OpenAI",
            "Amazon Web Services",
            "TORNADO",
            "ANDROID",
            "ORPHAN",
            "NORMANDY",
            "RANDOM",
            "STANDARD",
            "Doctor",
            "Robert",
        ],
    )
    def test_normal_words_unchanged(self, query: str):
        """Words containing O, R, N, T, A, D must NOT be escaped."""
        result = lucene_sanitize(query)
        assert result == query, f"Expected {query!r}, got {result!r}"


# ---------------------------------------------------------------------------
# 2. Lucene boolean keywords ARE escaped (whole-word only)
# ---------------------------------------------------------------------------
class TestBooleanKeywordsEscaped:
    """AND, OR, NOT as standalone uppercase words must be backslash-escaped."""

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("cats AND dogs", r"cats \AND dogs"),
            ("cats OR dogs", r"cats \OR dogs"),
            ("NOT cats", r"\NOT cats"),
            ("AND", r"\AND"),
            ("OR", r"\OR"),
            ("NOT", r"\NOT"),
            ("a AND b OR c NOT d", r"a \AND b \OR c \NOT d"),
        ],
    )
    def test_keywords_escaped(self, query: str, expected: str):
        result = lucene_sanitize(query)
        assert result == expected, f"Expected {expected!r}, got {result!r}"


# ---------------------------------------------------------------------------
# 3. Keywords inside words are NOT escaped
# ---------------------------------------------------------------------------
class TestKeywordsInsideWordsNotEscaped:
    """AND/OR/NOT embedded in larger words must not be touched."""

    @pytest.mark.parametrize(
        "query",
        [
            "ANDROID",       # contains AND
            "TORNADO",       # contains OR, AND, NOT as substrings
            "ORPHAN",        # contains OR
            "ANNOTATE",      # contains NOT
            "RANDOM",        # contains AND
            "NORMANDY",      # contains OR, AND
            "STANDARD",      # contains AND
            "Sandor",        # contains AND
            "ornament",      # lowercase, no match anyway
            "nothing",       # contains NOT in lowercase
        ],
    )
    def test_substrings_unchanged(self, query: str):
        result = lucene_sanitize(query)
        assert result == query, f"Expected {query!r}, got {result!r}"


# ---------------------------------------------------------------------------
# 4. Special characters are escaped
# ---------------------------------------------------------------------------
class TestSpecialCharsEscaped:
    @pytest.mark.parametrize(
        "char, escaped",
        [
            ("+", r"\+"),
            ("-", r"\-"),
            ("&", r"\&"),
            ("|", r"\|"),
            ("!", r"\!"),
            ("(", r"\("),
            (")", r"\)"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("[", r"\["),
            ("]", r"\]"),
            ("^", r"\^"),
            ('"', r'\"'),
            ("~", r"\~"),
            ("*", r"\*"),
            ("?", r"\?"),
            (":", r"\:"),
            ("/", r"\/"),
        ],
    )
    def test_individual_special_char(self, char: str, escaped: str):
        result = lucene_sanitize(char)
        assert result == escaped, f"Expected {escaped!r}, got {result!r}"


# ---------------------------------------------------------------------------
# 5. Combined: keywords + special chars + normal text
# ---------------------------------------------------------------------------
class TestCombined:
    def test_keyword_with_special_chars(self):
        result = lucene_sanitize("NOT (a OR b) AND c")
        assert result == r"\NOT \(a \OR b\) \AND c"

    def test_real_world_entity_query(self):
        """Realistic knowledge graph entity search."""
        result = lucene_sanitize("Donald Trump AND ORACLE")
        assert result == r"Donald Trump \AND ORACLE"

    def test_mixed_case_keywords_not_escaped(self):
        """Only uppercase AND, OR, NOT are Lucene keywords."""
        for query in ["cats and dogs", "cats or dogs", "not cats", "And", "Or", "Not"]:
            result = lucene_sanitize(query)
            assert result == query, f"Expected {query!r}, got {result!r}"

    def test_empty_string(self):
        assert lucene_sanitize("") == ""

    def test_whitespace_only(self):
        assert lucene_sanitize("   ") == "   "

    def test_no_special_chars(self):
        query = "simple query with no special characters"
        assert lucene_sanitize(query) == query
