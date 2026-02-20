"""Tests for fuzzy dedup safety guards.

Validates that the raised Jaccard threshold and edit-distance guard prevent
false-positive merges when names share long prefixes but differ in
semantically significant suffixes (e.g., sequential identifiers).

Background: 3-gram Jaccard similarity at the previous 0.9 threshold
falsely auto-merged entities like "SafetyRecommendation SR-2023-052"
with "SafetyRecommendation SR-2023-053" (Jaccard=0.93) because the long
shared prefix dominated the shingle set. Raising the threshold to 0.95
and adding a Levenshtein edit-distance guard prevents this class of
false merges while preserving auto-merge for genuine duplicates.
"""

from unittest.mock import MagicMock

import pytest

from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupResolutionState,
    _FUZZY_JACCARD_THRESHOLD,
    _build_candidate_indexes,
    _resolve_with_similarity,
)


def _make_node(name: str, uuid: str = '', labels: list[str] | None = None):
    """Create a minimal EntityNode mock for dedup testing."""
    node = MagicMock()
    node.name = name
    node.uuid = uuid or f'uuid-{name.lower().replace(" ", "-")}'
    node.labels = labels or ['Entity']
    node.group_id = 'test'
    node.attributes = {}
    node.summary = ''
    return node


class TestJaccardThresholdValue:
    """Verify the threshold is set high enough to prevent known false merges."""

    def test_threshold_is_at_least_095(self):
        """Threshold must be >= 0.95 to prevent false merges in the 0.90-0.94 range.

        At 0.9, these entities are falsely auto-merged:
        - "SafetyRecommendation SR-2023-052" vs "...053" (Jaccard=0.93)
        - "ICAO Annex 13 Amendment 16" vs "...17" (Jaccard=0.92)
        - "Service Bulletin SB-2024-001 Rev A" vs "...Rev B" (Jaccard=0.94)
        """
        assert _FUZZY_JACCARD_THRESHOLD >= 0.95


class TestSequentialIdentifiersNotAutoMerged:
    """Names with sequential identifiers must NOT be auto-merged.

    These have high 3-gram Jaccard similarity (0.90-0.94) due to long shared
    prefixes but represent completely different entities.
    """

    @pytest.mark.parametrize(
        'name_a, name_b',
        [
            # Type-prefixed sequential IDs
            ('SafetyRecommendation SR-2023-052', 'SafetyRecommendation SR-2023-053'),
            ('SafetyRecommendation SR-2023-074', 'SafetyRecommendation SR-2023-079'),
            # Regulatory documents with sequential amendments
            ('ICAO Annex 13 Amendment 16', 'ICAO Annex 13 Amendment 17'),
            # Versioned documents
            ('Service Bulletin SB-2024-001 Rev A', 'Service Bulletin SB-2024-001 Rev B'),
            # Sequential directive numbers
            ('Safety Directive SD-2024-0031', 'Safety Directive SD-2024-0032'),
        ],
    )
    def test_sequential_ids_deferred_to_llm(self, name_a: str, name_b: str):
        """Entities with sequential identifiers must be deferred to LLM, not auto-merged."""
        existing = _make_node(name_a, uuid='existing-uuid')
        extracted = _make_node(name_b, uuid='new-uuid')

        indexes = _build_candidate_indexes([existing])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_with_similarity([extracted], indexes, state)

        assert state.resolved_nodes[0] is None, (
            f'{name_b!r} was falsely auto-merged with {name_a!r}'
        )
        assert 0 in state.unresolved_indices


class TestGenuineDuplicatesStillAutoMerge:
    """Names that ARE the same entity should still auto-merge.

    Exact matches (after normalization) bypass fuzzy matching entirely and
    resolve deterministically regardless of entropy or Jaccard thresholds.
    """

    def test_exact_match_still_works(self):
        """Identical names still auto-merge via exact match."""
        existing = _make_node('Barcelona-El Prat Airport', uuid='existing-uuid')
        extracted = _make_node('Barcelona-El Prat Airport', uuid='new-uuid')

        indexes = _build_candidate_indexes([existing])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_with_similarity([extracted], indexes, state)

        assert state.resolved_nodes[0] is existing

    def test_case_variation_still_works(self):
        """Case differences still auto-merge via exact match normalization."""
        existing = _make_node('European Aviation Safety Agency', uuid='existing-uuid')
        extracted = _make_node('european aviation safety agency', uuid='new-uuid')

        indexes = _build_candidate_indexes([existing])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_with_similarity([extracted], indexes, state)

        assert state.resolved_nodes[0] is existing


class TestEditDistanceFunction:
    """Unit tests for the Levenshtein distance implementation."""

    def test_import_exists(self):
        """The _levenshtein_distance function must be importable."""
        from graphiti_core.utils.maintenance.dedup_helpers import _levenshtein_distance

        assert callable(_levenshtein_distance)

    def test_identical_strings(self):
        from graphiti_core.utils.maintenance.dedup_helpers import _levenshtein_distance

        assert _levenshtein_distance('hello', 'hello') == 0

    def test_single_substitution(self):
        from graphiti_core.utils.maintenance.dedup_helpers import _levenshtein_distance

        assert _levenshtein_distance('cat', 'bat') == 1

    def test_single_insertion(self):
        from graphiti_core.utils.maintenance.dedup_helpers import _levenshtein_distance

        assert _levenshtein_distance('cat', 'cats') == 1

    def test_single_deletion(self):
        from graphiti_core.utils.maintenance.dedup_helpers import _levenshtein_distance

        assert _levenshtein_distance('cats', 'cat') == 1

    def test_empty_strings(self):
        from graphiti_core.utils.maintenance.dedup_helpers import _levenshtein_distance

        assert _levenshtein_distance('', '') == 0
        assert _levenshtein_distance('abc', '') == 3
        assert _levenshtein_distance('', 'abc') == 3

    def test_sequential_ids(self):
        from graphiti_core.utils.maintenance.dedup_helpers import _levenshtein_distance

        assert _levenshtein_distance('sr 2023 052', 'sr 2023 053') == 1
        assert _levenshtein_distance('amendment 16', 'amendment 17') == 1

    def test_multiple_differences(self):
        from graphiti_core.utils.maintenance.dedup_helpers import _levenshtein_distance

        # 'kitten' -> 'sitting': 3 operations (k->s, e->i, +g)
        assert _levenshtein_distance('kitten', 'sitting') == 3


class TestEditDistanceGuard:
    """Test that the edit-distance guard prevents auto-merge for structurally different names.

    When Jaccard >= threshold but edit distance > 2, defer to LLM.
    """

    def test_high_jaccard_high_edit_distance_defers_to_llm(self):
        """Names with 3+ character differences should defer to LLM even with high Jaccard.

        This requires very long names where 3+ scattered changes still yield
        high Jaccard similarity.
        """
        # Construct names long enough that 3 changes still produce Jaccard >= 0.95
        base = 'A' * 50 + 'International Aviation Safety Board Report Number '
        name_a = base + 'XYZ'
        name_b = base + 'ABC'  # 3 character differences at end

        existing = _make_node(name_a, uuid='existing-uuid')
        extracted = _make_node(name_b, uuid='new-uuid')

        indexes = _build_candidate_indexes([existing])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_with_similarity([extracted], indexes, state)

        # Should defer to LLM due to edit distance > 2
        assert state.resolved_nodes[0] is None, (
            'Names with edit distance > 2 should not be auto-merged'
        )
