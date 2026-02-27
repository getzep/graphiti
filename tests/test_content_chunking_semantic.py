"""Tests for Smart Cutter: chunk_conversation_semantic() and lane adapters."""

from __future__ import annotations

import hashlib
import unittest
from datetime import datetime, timedelta, timezone

from graphiti_core.utils.content_chunking import (
    ChunkBoundary,
    SmartCutterConfig,
    chunk_conversation_semantic,
    estimate_tokens,
    graphiti_lane_merge,
    om_lane_split,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def _make_msg(
    msg_id: str,
    content: str,
    created_at: str,
    embedding: list[float] | None = None,
) -> dict:
    return {
        'message_id': msg_id,
        'content': content,
        'created_at': created_at,
        'content_embedding': embedding or [1.0, 0.0, 0.0],
    }


def _make_messages(
    count: int,
    base_time: datetime | None = None,
    gap_minutes: int = 5,
    embedding: list[float] | None = None,
    content_prefix: str = 'Message content for testing purposes number',
) -> list[dict]:
    """Generate a list of test messages with sequential timestamps."""
    base = base_time or datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    emb = embedding or [1.0, 0.0, 0.0]
    msgs = []
    for i in range(count):
        ts = base + timedelta(minutes=i * gap_minutes)
        msgs.append({
            'message_id': f'msg_{i:04d}',
            'content': f'{content_prefix} {i}',
            'created_at': _iso(ts),
            'content_embedding': emb,
        })
    return msgs


class TestSmartCutterConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SmartCutterConfig()
        self.assertEqual(cfg.hard_gap_hours, 48.0)
        self.assertEqual(cfg.semantic_drift_threshold, 0.50)
        self.assertEqual(cfg.substantive_min_tokens, 10)
        self.assertEqual(cfg.substantive_min_chars, 40)
        self.assertEqual(cfg.max_chunk_tokens, 4000)
        self.assertEqual(cfg.lookback_window_messages, 25)
        self.assertEqual(cfg.lookback_min_head_tokens, 600)
        self.assertEqual(cfg.lookback_min_tail_tokens, 300)


class TestChunkConversationSemantic(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        result = chunk_conversation_semantic([])
        self.assertEqual(result, [])

    def test_single_message(self):
        msgs = _make_messages(1)
        result = chunk_conversation_semantic(msgs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].boundary_reason, 'end_of_stream')
        self.assertEqual(result[0].message_ids, ['msg_0000'])
        self.assertEqual(result[0].chunk_index, 0)

    def test_deterministic_chunk_ids(self):
        msgs = _make_messages(5)
        r1 = chunk_conversation_semantic(msgs)
        r2 = chunk_conversation_semantic(msgs)
        self.assertEqual(len(r1), len(r2))
        for a, b in zip(r1, r2):
            self.assertEqual(a.chunk_id, b.chunk_id)
            self.assertEqual(a.message_ids, b.message_ids)

    def test_chunk_id_formula(self):
        msgs = _make_messages(3)
        result = chunk_conversation_semantic(msgs)
        self.assertEqual(len(result), 1)
        chunk = result[0]
        expected_id = hashlib.sha256(
            f'smartcut|msg_0000|msg_0002|3'.encode('utf-8')
        ).hexdigest()
        self.assertEqual(chunk.chunk_id, expected_id)

    def test_hard_gap_produces_cut(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msgs = [
            _make_msg('m1', 'Hello this is a long enough message for testing.', _iso(base)),
            _make_msg(
                'm2',
                'After a long gap this is also a long enough message.',
                _iso(base + timedelta(hours=49)),
            ),
        ]
        result = chunk_conversation_semantic(msgs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].boundary_reason, 'hard_gap')
        self.assertEqual(result[0].message_ids, ['m1'])
        self.assertEqual(result[1].message_ids, ['m2'])

    def test_no_cut_within_48h(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msgs = [
            _make_msg(
                'm1', 'First message with enough content for testing.', _iso(base),
                [1.0, 0.0, 0.0],
            ),
            _make_msg(
                'm2', 'Second message also with enough content for testing.',
                _iso(base + timedelta(hours=47)),
                [1.0, 0.0, 0.0],
            ),
        ]
        result = chunk_conversation_semantic(msgs)
        self.assertEqual(len(result), 1)

    def test_semantic_drift_cut(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msgs = [
            _make_msg(
                'm1',
                'A long enough message about topic A with many words to be substantive.',
                _iso(base),
                [1.0, 0.0, 0.0],
            ),
            _make_msg(
                'm2',
                'A long enough message about topic A continued with more words and substance.',
                _iso(base + timedelta(minutes=5)),
                [0.99, 0.01, 0.0],
            ),
            _make_msg(
                'm3',
                'A completely different topic B message with sufficient words to be substantive.',
                _iso(base + timedelta(minutes=10)),
                [0.0, 0.0, 1.0],  # orthogonal = cosine ~0
            ),
        ]
        cfg = SmartCutterConfig(semantic_drift_threshold=0.50)
        result = chunk_conversation_semantic(msgs, cfg)
        self.assertGreaterEqual(len(result), 2)
        # First chunk should contain m1+m2 (similar), m3 should start new chunk
        found_drift = any(c.boundary_reason == 'semantic_drift' for c in result)
        self.assertTrue(found_drift)

    def test_non_substantive_messages_no_semantic_cut(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msgs = [
            _make_msg(
                'm1',
                'A substantive message about topic A with many many words for testing.',
                _iso(base),
                [1.0, 0.0, 0.0],
            ),
            _make_msg(
                'm2', 'ok',  # too short to be substantive
                _iso(base + timedelta(minutes=5)),
                [0.0, 0.0, 1.0],  # very different but non-substantive
            ),
            _make_msg(
                'm3',
                'Another substantive message continuing topic A with enough words for testing.',
                _iso(base + timedelta(minutes=10)),
                [0.95, 0.05, 0.0],
            ),
        ]
        result = chunk_conversation_semantic(msgs)
        # Non-substantive m2 should not trigger a semantic cut
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].message_ids), 3)

    def test_token_overflow_cut(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Each message ~1000 tokens = 4000 chars
        big_content = 'x' * 4000
        msgs = [
            _make_msg(f'm{i}', big_content, _iso(base + timedelta(minutes=i)))
            for i in range(6)
        ]
        cfg = SmartCutterConfig(max_chunk_tokens=4000)
        result = chunk_conversation_semantic(msgs, cfg)
        self.assertGreater(len(result), 1)
        # Check that each chunk respects max_chunk_tokens approximately
        for chunk in result[:-1]:
            self.assertLessEqual(chunk.token_count, cfg.max_chunk_tokens + 1100)

    def test_missing_message_id_raises(self):
        msgs = [{'content': 'hi', 'created_at': '2025-01-01T00:00:00Z',
                 'content_embedding': [1.0]}]
        with self.assertRaises(ValueError):
            chunk_conversation_semantic(msgs)

    def test_missing_content_raises(self):
        msgs = [{'message_id': 'm1', 'created_at': '2025-01-01T00:00:00Z',
                 'content_embedding': [1.0]}]
        with self.assertRaises(ValueError):
            chunk_conversation_semantic(msgs)

    def test_missing_created_at_raises(self):
        msgs = [{'message_id': 'm1', 'content': 'hi', 'content_embedding': [1.0]}]
        with self.assertRaises(ValueError):
            chunk_conversation_semantic(msgs)

    def test_missing_embedding_raises(self):
        msgs = [{'message_id': 'm1', 'content': 'hi',
                 'created_at': '2025-01-01T00:00:00Z'}]
        with self.assertRaises(ValueError):
            chunk_conversation_semantic(msgs)

    def test_sorts_by_created_at(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msgs = [
            _make_msg('m2', 'Second message content.', _iso(base + timedelta(minutes=10))),
            _make_msg('m1', 'First message content.', _iso(base)),
        ]
        result = chunk_conversation_semantic(msgs)
        # Should be sorted, so m1 comes first
        self.assertEqual(result[0].message_ids[0], 'm1')

    def test_boundary_score_populated(self):
        msgs = _make_messages(3)
        result = chunk_conversation_semantic(msgs)
        for chunk in result:
            self.assertIsInstance(chunk.boundary_score, float)

    def test_time_range_populated(self):
        msgs = _make_messages(3)
        result = chunk_conversation_semantic(msgs)
        for chunk in result:
            self.assertTrue(chunk.time_range_start)
            self.assertTrue(chunk.time_range_end)

    def test_zero_norm_embedding(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msgs = [
            _make_msg('m1', 'A long enough message for testing here.', _iso(base), [0.0, 0.0]),
            _make_msg(
                'm2', 'Another long enough message for testing here.',
                _iso(base + timedelta(minutes=5)),
                [0.0, 0.0],
            ),
        ]
        result = chunk_conversation_semantic(msgs)
        # Should not crash; zero norm treated as similarity 0.0
        self.assertGreaterEqual(len(result), 1)


class TestGraphitiLaneMerge(unittest.TestCase):
    def test_merge_small_adjacent_chunks(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Create messages that produce small chunks
        msgs = [
            _make_msg('m1', 'Short.', _iso(base)),
            _make_msg('m2', 'Also short.', _iso(base + timedelta(minutes=5))),
        ]
        # Manually create small base chunks
        chunks = [
            ChunkBoundary(
                chunk_index=0, chunk_id='c0', message_ids=['m1'],
                token_count=50, time_range_start=_iso(base),
                time_range_end=_iso(base),
                boundary_reason='semantic_drift', boundary_score=0.4,
            ),
            ChunkBoundary(
                chunk_index=1, chunk_id='c1', message_ids=['m2'],
                token_count=50, time_range_start=_iso(base + timedelta(minutes=5)),
                time_range_end=_iso(base + timedelta(minutes=5)),
                boundary_reason='end_of_stream', boundary_score=0.0,
            ),
        ]
        result = graphiti_lane_merge(chunks, msgs)
        # Both chunks < 800 tokens and no hard_gap → should merge
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].message_ids, ['m1', 'm2'])

    def test_no_merge_across_hard_gap(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msgs = _make_messages(2, base)
        chunks = [
            ChunkBoundary(
                chunk_index=0, chunk_id='c0', message_ids=['msg_0000'],
                token_count=50, time_range_start=_iso(base),
                time_range_end=_iso(base),
                boundary_reason='hard_gap', boundary_score=1.0,
            ),
            ChunkBoundary(
                chunk_index=1, chunk_id='c1', message_ids=['msg_0001'],
                token_count=50,
                time_range_start=_iso(base + timedelta(minutes=5)),
                time_range_end=_iso(base + timedelta(minutes=5)),
                boundary_reason='end_of_stream', boundary_score=0.0,
            ),
        ]
        result = graphiti_lane_merge(chunks, msgs)
        self.assertEqual(len(result), 2)

    def test_no_merge_if_either_chunk_large(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msgs = _make_messages(2, base)
        chunks = [
            ChunkBoundary(
                chunk_index=0, chunk_id='c0', message_ids=['msg_0000'],
                token_count=900, time_range_start=_iso(base),
                time_range_end=_iso(base),
                boundary_reason='semantic_drift', boundary_score=0.4,
            ),
            ChunkBoundary(
                chunk_index=1, chunk_id='c1', message_ids=['msg_0001'],
                token_count=50,
                time_range_start=_iso(base + timedelta(minutes=5)),
                time_range_end=_iso(base + timedelta(minutes=5)),
                boundary_reason='end_of_stream', boundary_score=0.0,
            ),
        ]
        result = graphiti_lane_merge(chunks, msgs)
        self.assertEqual(len(result), 2)


class TestOMLaneSplit(unittest.TestCase):
    def test_no_split_under_ceiling(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msgs = _make_messages(2, base)
        chunks = [
            ChunkBoundary(
                chunk_index=0, chunk_id='c0', message_ids=['msg_0000', 'msg_0001'],
                token_count=3000,
                time_range_start=_iso(base),
                time_range_end=_iso(base + timedelta(minutes=5)),
                boundary_reason='end_of_stream', boundary_score=0.0,
            ),
        ]
        result = om_lane_split(chunks, msgs)
        self.assertEqual(len(result), 1)

    def test_split_over_ceiling(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        msgs = _make_messages(10, base)
        msg_ids = [f'msg_{i:04d}' for i in range(10)]
        chunks = [
            ChunkBoundary(
                chunk_index=0, chunk_id='c0', message_ids=msg_ids,
                token_count=5000,
                time_range_start=_iso(base),
                time_range_end=_iso(base + timedelta(minutes=45)),
                boundary_reason='end_of_stream', boundary_score=0.0,
            ),
        ]
        result = om_lane_split(chunks, msgs)
        self.assertGreaterEqual(len(result), 2)
        # All message IDs should be covered
        all_ids = []
        for c in result:
            all_ids.extend(c.message_ids)
        self.assertEqual(sorted(all_ids), sorted(msg_ids))


if __name__ == '__main__':
    unittest.main()
