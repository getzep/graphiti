"""
Tests for resolve_edge_contradictions() and rrf().

Both are pure functions that had zero test coverage prior to this file.
"""

from datetime import datetime, timezone
from uuid import uuid4

from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_utils import rrf
from graphiti_core.utils.maintenance.edge_operations import resolve_edge_contradictions


def make_edge(
    uuid: str | None = None,
    fact: str = 'test fact',
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
    expired_at: datetime | None = None,
) -> EntityEdge:
    return EntityEdge(
        uuid=uuid or str(uuid4()),
        source_node_uuid=str(uuid4()),
        target_node_uuid=str(uuid4()),
        name='test_relation',
        fact=fact,
        group_id='test',
        created_at=datetime.now(timezone.utc),
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
    )


class TestResolveEdgeContradictions:
    """resolve_edge_contradictions() — edge_operations.py:538

    Coverage before this test: 0%.
    Three branches:
      1. Empty candidates → empty result
      2. Temporal gap (no overlap) → skip
      3. Candidate older than resolved → invalidate
    """

    def test_empty_candidates_returns_empty(self):
        resolved = make_edge(valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert resolve_edge_contradictions(resolved, []) == []

    def test_candidate_invalid_before_resolved_valid_no_invalidation(self):
        resolved = make_edge(valid_at=datetime(2025, 6, 1, tzinfo=timezone.utc))
        candidate = make_edge(
            valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            invalid_at=datetime(2025, 5, 1, tzinfo=timezone.utc),
        )
        assert resolve_edge_contradictions(resolved, [candidate]) == []

    def test_resolved_invalid_before_candidate_valid_no_invalidation(self):
        resolved = make_edge(
            valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            invalid_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
        )
        candidate = make_edge(valid_at=datetime(2025, 6, 1, tzinfo=timezone.utc))
        assert resolve_edge_contradictions(resolved, [candidate]) == []

    def test_candidate_older_than_resolved_invalidates(self):
        resolved = make_edge(valid_at=datetime(2025, 6, 1, tzinfo=timezone.utc))
        candidate = make_edge(
            uuid='candidate-1',
            valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        result = resolve_edge_contradictions(resolved, [candidate])
        assert len(result) == 1
        assert result[0].uuid == 'candidate-1'
        assert result[0].invalid_at == resolved.valid_at
        assert result[0].expired_at is not None

    def test_mixed_candidates_only_older_invalidated(self):
        resolved = make_edge(valid_at=datetime(2025, 6, 1, tzinfo=timezone.utc))
        older = make_edge(uuid='older', valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        newer = make_edge(
            uuid='newer',
            valid_at=datetime(2025, 12, 1, tzinfo=timezone.utc),
            invalid_at=datetime(2025, 11, 1, tzinfo=timezone.utc),
        )
        result = resolve_edge_contradictions(resolved, [older, newer])
        uuids = [e.uuid for e in result]
        assert 'older' in uuids
        assert 'newer' not in uuids

    def test_preserves_existing_expired_at(self):
        resolved = make_edge(valid_at=datetime(2025, 6, 1, tzinfo=timezone.utc))
        existing_expiry = datetime(2025, 3, 1, tzinfo=timezone.utc)
        candidate = make_edge(
            valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            expired_at=existing_expiry,
        )
        result = resolve_edge_contradictions(resolved, [candidate])
        assert result[0].expired_at == existing_expiry

    def test_multiple_invalidations_in_order(self):
        resolved = make_edge(valid_at=datetime(2025, 6, 1, tzinfo=timezone.utc))
        c1 = make_edge(uuid='c1', valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        c2 = make_edge(uuid='c2', valid_at=datetime(2025, 3, 1, tzinfo=timezone.utc))
        result = resolve_edge_contradictions(resolved, [c1, c2])
        assert len(result) == 2
        assert {e.uuid for e in result} == {'c1', 'c2'}


class TestRRF:
    """rrf() — search_utils.py:1780

    Coverage before this test: 0% (only tested implicitly via search integration tests).
    Formula: score[uuid] += 1 / (position + rank_const)
    """

    def test_empty_lists(self):
        uuids, scores = rrf([])
        assert uuids == []
        assert scores == []

    def test_single_list_is_identity_order(self):
        uuids, scores = rrf([['a', 'b', 'c']])
        assert uuids == ['a', 'b', 'c']
        assert scores[0] == 1.0
        assert scores[1] == 1 / 2
        assert scores[2] == 1 / 3

    def test_two_lists_with_overlap_sum_scores(self):
        uuids, scores = rrf([['a', 'b'], ['b', 'a']])
        assert abs(scores[0] - 1.5) < 0.001

    def test_min_score_filters_low_scoring_items(self):
        uuids, scores = rrf([['a', 'b', 'c']], min_score=0.4)
        assert 'c' not in uuids

    def test_rank_const_changes_score_distribution(self):
        _, scores_low = rrf([['x', 'y']], rank_const=1)
        _, scores_high = rrf([['x', 'y']], rank_const=5)
        assert scores_low[0] > scores_high[0]

    def test_empty_inner_list_ignored(self):
        uuids, scores = rrf([['a'], [], ['b']])
        assert set(uuids[:2]) == {'a', 'b'}

    def test_three_ranked_lists_rrf_fusion(self):
        uuids, scores = rrf(
            [
                ['x', 'y', 'z'],
                ['z', 'y', 'x'],
                ['y', 'z', 'x'],
            ]
        )
        # y: 1/2 + 1/2 + 1/1 = 2.0 (highest)
        # z: 1/3 + 1/1 + 1/2 = 1.833...
        # x: 1/1 + 1/3 + 1/3 = 1.666...
        assert uuids[0] == 'y'
        assert abs(scores[0] - 2.0) < 0.001
