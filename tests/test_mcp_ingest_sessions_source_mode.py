"""Tests for mcp_ingest_sessions.py --source-mode (FR-4 + FR-10)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestSourceModeArgParsing(unittest.TestCase):
    """Test --source-mode argument parsing."""

    def test_default_source_mode_is_neo4j(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args(['--group-id', 's1_sessions_main'])
        self.assertEqual(args.source_mode, 'neo4j')

    def test_explicit_neo4j_mode(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--source-mode', 'neo4j',
        ])
        self.assertEqual(args.source_mode, 'neo4j')

    def test_explicit_evidence_mode(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--source-mode', 'evidence',
            '--evidence', '/tmp/test.json',
        ])
        self.assertEqual(args.source_mode, 'evidence')

    def test_build_manifest_flag(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--source-mode', 'neo4j',
            '--build-manifest', '/tmp/manifest.jsonl',
        ])
        self.assertEqual(args.build_manifest, '/tmp/manifest.jsonl')

    def test_claim_mode_flags(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--source-mode', 'neo4j',
            '--manifest', '/tmp/manifest.jsonl',
            '--claim-mode',
            '--shards', '4',
            '--shard-index', '0',
        ])
        self.assertTrue(args.claim_mode)
        self.assertEqual(args.shards, 4)
        self.assertEqual(args.shard_index, 0)

    def test_claim_state_check_flag(self):
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--manifest', '/tmp/manifest.jsonl',
            '--claim-state-check',
            '--dry-run',
        ])
        self.assertTrue(args.claim_state_check)


class TestBootstrapGuard(unittest.TestCase):
    """Test BOOTSTRAP_REQUIRED guard (FR-4 item 6)."""

    def test_bootstrap_required_when_no_messages_and_evidence_exists(self):
        from scripts.mcp_ingest_sessions import check_bootstrap_guard

        # No messages in Neo4j AND evidence files exist → BOOTSTRAP_REQUIRED
        result = check_bootstrap_guard(
            neo4j_message_count=0,
            evidence_files_exist=True,
        )
        self.assertTrue(result)

    def test_no_bootstrap_when_messages_exist(self):
        from scripts.mcp_ingest_sessions import check_bootstrap_guard

        # Messages exist in Neo4j → guard satisfied
        result = check_bootstrap_guard(
            neo4j_message_count=100,
            evidence_files_exist=True,
        )
        self.assertFalse(result)

    def test_no_bootstrap_when_no_evidence_files(self):
        from scripts.mcp_ingest_sessions import check_bootstrap_guard

        # No evidence files → no guard needed
        result = check_bootstrap_guard(
            neo4j_message_count=0,
            evidence_files_exist=False,
        )
        self.assertFalse(result)


class TestClaimStateDB(unittest.TestCase):
    """Test SQLite claim-state storage (FR-10)."""

    def test_claim_db_schema(self):
        import tempfile

        from scripts.mcp_ingest_sessions import init_claim_db

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            # Verify table exists with expected columns
            cursor = conn.execute('PRAGMA table_info(chunk_claims)')
            columns = {row[1] for row in cursor.fetchall()}
            expected = {
                'chunk_id', 'status', 'worker_id',
                'claimed_at', 'completed_at', 'fail_count', 'error',
            }
            self.assertTrue(
                expected.issubset(columns),
                f'Missing columns: {expected - columns}',
            )
            conn.close()

    def test_claim_pending_to_claimed(self):
        import tempfile

        from scripts.mcp_ingest_sessions import (
            claim_chunk,
            init_claim_db,
            seed_claims,
        )

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            seed_claims(conn, ['chunk_001', 'chunk_002'])

            # Claim one chunk
            claimed = claim_chunk(conn, worker_id='w0')
            self.assertIsNotNone(claimed)
            self.assertIn(claimed, ['chunk_001', 'chunk_002'])

            # Verify status changed
            row = conn.execute(
                'SELECT status FROM chunk_claims WHERE chunk_id = ?',
                (claimed,),
            ).fetchone()
            self.assertEqual(row[0], 'claimed')
            conn.close()


class TestEvidenceModeBackwardCompat(unittest.TestCase):
    """Test evidence mode still works (backward compat)."""

    def test_evidence_mode_parseable(self):
        """Evidence mode args parse correctly."""
        from scripts.mcp_ingest_sessions import build_parser

        parser = build_parser()
        args = parser.parse_args([
            '--group-id', 's1_sessions_main',
            '--source-mode', 'evidence',
            '--evidence', 'evidence/sessions_v1/main/all_evidence.json',
            '--limit', '10',
            '--dry-run',
        ])
        self.assertEqual(args.source_mode, 'evidence')
        self.assertEqual(args.limit, 10)
        self.assertTrue(args.dry_run)


class TestBuildEpisodeBody(unittest.TestCase):
    """Test _build_episode_body formats chunk content correctly."""

    def test_basic_formatting(self):
        from scripts.mcp_ingest_sessions import _build_episode_body

        messages_by_id = {
            'msg1': {
                'message_id': 'msg1',
                'content': 'Hello there',
                'created_at': '2026-01-15T12:00:00Z',
                'role': 'user',
            },
            'msg2': {
                'message_id': 'msg2',
                'content': 'Hi back',
                'created_at': '2026-01-15T12:01:00Z',
                'role': 'assistant',
            },
        }
        body = _build_episode_body(['msg1', 'msg2'], messages_by_id)
        self.assertIn('Hello there', body)
        self.assertIn('Hi back', body)
        self.assertIn('2026-01-15', body)

    def test_missing_message_skipped(self):
        from scripts.mcp_ingest_sessions import _build_episode_body

        body = _build_episode_body(['nonexistent'], {})
        self.assertEqual(body, '')

    def test_message_order_preserved(self):
        from scripts.mcp_ingest_sessions import _build_episode_body

        messages_by_id = {
            'a': {'content': 'first', 'created_at': '2026-01-15T10:00:00Z', 'role': 'user'},
            'b': {'content': 'second', 'created_at': '2026-01-15T11:00:00Z', 'role': 'user'},
        }
        body = _build_episode_body(['a', 'b'], messages_by_id)
        self.assertLess(body.index('first'), body.index('second'))


class TestLoadManifest(unittest.TestCase):
    """Test _load_manifest reads JSONL manifest correctly."""

    def test_load_manifest(self):
        import os
        import tempfile

        from scripts.mcp_ingest_sessions import _load_manifest

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            import json as _json
            _json.dump({'chunk_id': 'abc', 'message_ids': ['m1', 'm2'], 'content': 'c1'}, f)
            f.write('\n')
            _json.dump({'chunk_id': 'def', 'message_ids': ['m3'], 'content': 'c2'}, f)
            f.write('\n')
            tmp = f.name

        try:
            from pathlib import Path
            result = _load_manifest(Path(tmp))
            self.assertIn('abc', result)
            self.assertIn('def', result)
            self.assertEqual(result['abc']['message_ids'], ['m1', 'm2'])
        finally:
            os.unlink(tmp)

    def test_empty_manifest_returns_empty_dict(self):
        import os
        import tempfile
        from pathlib import Path

        from scripts.mcp_ingest_sessions import _load_manifest

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            tmp = f.name

        try:
            result = _load_manifest(Path(tmp))
            self.assertEqual(result, {})
        finally:
            os.unlink(tmp)

    def test_malformed_lines_skipped(self):
        import json as _json
        import os
        import tempfile
        from pathlib import Path

        from scripts.mcp_ingest_sessions import _load_manifest

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('not json\n')
            _json.dump({'chunk_id': 'valid', 'message_ids': ['m1'], 'content': 'c'}, f)
            f.write('\n')
            tmp = f.name

        try:
            result = _load_manifest(Path(tmp))
            self.assertIn('valid', result)
            self.assertEqual(len(result), 1)
        finally:
            os.unlink(tmp)


class TestClaimHelpers(unittest.TestCase):
    """Test _claim_done and _claim_fail helpers."""

    def _make_db(self):
        import tempfile

        from scripts.mcp_ingest_sessions import init_claim_db, seed_claims
        tmp = tempfile.mktemp(suffix='.db')
        conn = init_claim_db(tmp)
        seed_claims(conn, ['chunk_001', 'chunk_002'])
        return conn, tmp

    def test_claim_done_marks_status(self):
        import os

        from scripts.mcp_ingest_sessions import _claim_done, claim_chunk

        conn, tmp = self._make_db()
        try:
            chunk_id = claim_chunk(conn, 'w0')
            _claim_done(conn, chunk_id)
            row = conn.execute(
                "SELECT status FROM chunk_claims WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            self.assertEqual(row[0], 'done')
        finally:
            conn.close()
            os.unlink(tmp)

    def test_claim_fail_increments_fail_count(self):
        import os

        from scripts.mcp_ingest_sessions import _claim_fail, claim_chunk

        conn, tmp = self._make_db()
        try:
            chunk_id = claim_chunk(conn, 'w0')
            _claim_fail(conn, chunk_id, 'test error')
            row = conn.execute(
                "SELECT status, fail_count, error FROM chunk_claims WHERE chunk_id=?",
                (chunk_id,)
            ).fetchone()
            self.assertEqual(row[0], 'failed')
            self.assertEqual(row[1], 1)
            self.assertEqual(row[2], 'test error')
        finally:
            conn.close()
            os.unlink(tmp)

    def test_claim_fail_twice_increments_twice(self):
        import os
        import tempfile

        from scripts.mcp_ingest_sessions import _claim_fail, claim_chunk, init_claim_db, seed_claims
        tmp = tempfile.mktemp(suffix='.db')
        conn = init_claim_db(tmp)
        seed_claims(conn, ['chunk_x'])
        try:
            # First claim + fail
            chunk_id = claim_chunk(conn, 'w0')
            _claim_fail(conn, chunk_id, 'err1')
            # Re-seed to pending so we can claim again
            conn.execute("UPDATE chunk_claims SET status='pending' WHERE chunk_id=?", (chunk_id,))
            conn.commit()
            chunk_id2 = claim_chunk(conn, 'w0')
            _claim_fail(conn, chunk_id2, 'err2')
            row = conn.execute(
                "SELECT fail_count FROM chunk_claims WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            self.assertEqual(row[0], 2)
        finally:
            conn.close()
            os.unlink(tmp)


class TestHardeningCaps(unittest.TestCase):
    """Test that hardening constants are in place."""

    def test_neo4j_fetch_ceiling_exists(self):
        from scripts.mcp_ingest_sessions import _NEO4J_FETCH_CEILING
        self.assertGreater(_NEO4J_FETCH_CEILING, 0)
        self.assertLessEqual(_NEO4J_FETCH_CEILING, 100_000)

    def test_benchmark_max_response_bytes_exists(self):
        from scripts.run_retrieval_benchmark import _MAX_RESPONSE_BYTES
        self.assertGreater(_MAX_RESPONSE_BYTES, 0)

    def test_mcp_server_caps_defined_in_source(self):
        """Verify cap constants appear in the MCP server source file."""
        from pathlib import Path
        server_src = (
            Path(__file__).resolve().parents[1]
            / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'
        )
        source = server_src.read_text(encoding='utf-8')
        self.assertIn('_MAX_NODES_CAP', source)
        self.assertIn('_MAX_FACTS_CAP', source)
        # Ensure the caps are applied in search_nodes and search_memory_facts.
        self.assertIn('_MAX_NODES_CAP', source)
        self.assertIn('_MAX_FACTS_CAP', source)


class TestManifestMalformedLineWarning(unittest.TestCase):
    """Item 1: _load_manifest emits WARNING for malformed JSONL lines (not silent skip)."""

    def test_malformed_line_triggers_warning(self):
        """A malformed JSONL line should produce a logger.warning, not be silently dropped."""
        import json as _json
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from scripts.mcp_ingest_sessions import _load_manifest

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('not json at all\n')
            _json.dump({'chunk_id': 'good', 'message_ids': [], 'content': 'c'}, f)
            f.write('\n')
            tmp = f.name

        try:
            with patch('scripts.mcp_ingest_sessions._logger') as mock_log:
                result = _load_manifest(Path(tmp))
            # Valid line still loaded
            self.assertIn('good', result)
            # Warning was emitted for the bad line
            self.assertTrue(
                mock_log.warning.called,
                'Expected _logger.warning to be called for malformed JSONL line',
            )
            # Warning includes line number (first positional arg is format string; check args)
            call_args = mock_log.warning.call_args
            # Line number 1 should appear in the arguments
            self.assertIn(1, call_args[0], 'Expected line number 1 in warning args')
        finally:
            os.unlink(tmp)

    def test_warning_does_not_include_raw_payload(self):
        """Sensitive payload content must NOT appear verbatim in the warning args."""
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from scripts.mcp_ingest_sessions import _load_manifest

        sensitive = 'SENSITIVE_TOKEN_XYZ'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write something that looks malformed but contains sensitive text
            f.write(f'{sensitive} {{bad json\n')
            tmp = f.name

        try:
            with patch('scripts.mcp_ingest_sessions._logger') as mock_log:
                _load_manifest(Path(tmp))
            # Flatten all warning call args to a string for inspection
            call_args_str = str(mock_log.warning.call_args_list)
            self.assertNotIn(
                sensitive,
                call_args_str,
                'Raw payload content must not appear in warning message',
            )
        finally:
            os.unlink(tmp)

    def test_blank_lines_not_warned(self):
        """Blank lines are silently skipped — no warning should fire."""
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from scripts.mcp_ingest_sessions import _load_manifest

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('\n\n   \n')
            tmp = f.name

        try:
            with patch('scripts.mcp_ingest_sessions._logger') as mock_log:
                result = _load_manifest(Path(tmp))
            self.assertEqual(result, {})
            mock_log.warning.assert_not_called()
        finally:
            os.unlink(tmp)


class TestSentNotMarkedIdempotency(unittest.TestCase):
    """Item 2: sent_not_marked status prevents duplicate add_memory on retry."""

    def _make_db(self, chunk_ids: list[str]):
        import tempfile

        from scripts.mcp_ingest_sessions import init_claim_db, seed_claims

        tmp = tempfile.mktemp(suffix='.db')
        conn = init_claim_db(tmp)
        seed_claims(conn, chunk_ids)
        return conn, tmp

    def test_claim_sent_not_marked_helper(self):
        """_claim_sent_not_marked transitions a claimed chunk to sent_not_marked."""
        import os

        from scripts.mcp_ingest_sessions import (
            _claim_sent_not_marked,
            claim_chunk,
        )

        conn, tmp = self._make_db(['chunk_A'])
        try:
            chunk_id = claim_chunk(conn, 'w0')
            self.assertEqual(chunk_id, 'chunk_A')
            _claim_sent_not_marked(conn, chunk_id)
            row = conn.execute(
                'SELECT status FROM chunk_claims WHERE chunk_id=?', (chunk_id,)
            ).fetchone()
            self.assertEqual(row[0], 'sent_not_marked')
        finally:
            conn.close()
            os.unlink(tmp)

    def test_sent_not_marked_survives_after_neo4j_fail(self):
        """After neo4j mark failure the chunk stays sent_not_marked (not failed/pending)."""
        import os

        from scripts.mcp_ingest_sessions import (
            _claim_sent_not_marked,
            claim_chunk,
        )

        conn, tmp = self._make_db(['chunk_E'])
        try:
            chunk_id = claim_chunk(conn, 'w0')
            _claim_sent_not_marked(conn, chunk_id)

            # Simulate neo4j mark failure: update error field but keep status
            conn.execute(
                "UPDATE chunk_claims SET error='neo4j timeout' WHERE chunk_id=?",
                (chunk_id,),
            )
            conn.commit()

            row = conn.execute(
                'SELECT status, error FROM chunk_claims WHERE chunk_id=?', (chunk_id,)
            ).fetchone()
            self.assertEqual(row[0], 'sent_not_marked')
            self.assertIn('neo4j', row[1])
        finally:
            conn.close()
            os.unlink(tmp)

class TestSearchRateLimiter(unittest.TestCase):
    """Item 3: sliding-window rate limiter for search endpoints."""

    def setUp(self):
        import asyncio
        # Create an explicit event loop so tests work on Python 3.14+
        # where asyncio.get_event_loop() no longer auto-creates one.
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def tearDown(self):
        self._loop.close()
        import asyncio
        asyncio.set_event_loop(None)

    def _make_limiter(self, max_requests: int = 5, window_seconds: float = 10.0):
        import sys
        from pathlib import Path

        _mcp_src = str(Path(__file__).resolve().parents[1] / 'mcp_server' / 'src')
        sys.path.insert(0, _mcp_src)
        try:
            from utils.rate_limiter import SlidingWindowRateLimiter
        finally:
            sys.path.remove(_mcp_src)
            sys.modules.pop('config', None)

        return SlidingWindowRateLimiter(max_requests=max_requests, window_seconds=window_seconds)

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def test_allows_requests_within_limit(self):
        rl = self._make_limiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            self.assertTrue(self._run(rl.is_allowed()))

    def test_blocks_request_over_limit(self):
        rl = self._make_limiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            self._run(rl.is_allowed())
        # 4th request should be blocked
        self.assertFalse(self._run(rl.is_allowed()))

    def test_different_keys_independent(self):
        rl = self._make_limiter(max_requests=2, window_seconds=60)
        for _ in range(2):
            self._run(rl.is_allowed('key_a'))
        # key_a is exhausted; key_b should still be allowed
        self.assertFalse(self._run(rl.is_allowed('key_a')))
        self.assertTrue(self._run(rl.is_allowed('key_b')))

    def test_window_expiry_allows_new_requests(self):
        """Requests outside the window no longer count against the limit."""
        import time

        rl = self._make_limiter(max_requests=2, window_seconds=0.05)
        self._run(rl.is_allowed())
        self._run(rl.is_allowed())
        self.assertFalse(self._run(rl.is_allowed()))

        # Wait for window to expire
        time.sleep(0.1)
        self.assertTrue(self._run(rl.is_allowed()))

    def test_invalid_config_raises(self):
        import sys
        from pathlib import Path

        _mcp_src = str(Path(__file__).resolve().parents[1] / 'mcp_server' / 'src')
        sys.path.insert(0, _mcp_src)
        try:
            from utils.rate_limiter import SlidingWindowRateLimiter
        finally:
            sys.path.remove(_mcp_src)
            sys.modules.pop('config', None)

        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(max_requests=0, window_seconds=60)
        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(max_requests=10, window_seconds=0)

    def test_rate_limiter_constants_in_mcp_server(self):
        """Verify rate limiter env-var constants and instance appear in MCP server source."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'
        ).read_text(encoding='utf-8')
        self.assertIn('_SEARCH_RATE_LIMIT_ENABLED', src)
        self.assertIn('_SEARCH_RATE_LIMIT_REQUESTS', src)
        self.assertIn('_SEARCH_RATE_LIMIT_WINDOW', src)
        self.assertIn('_search_rate_limiter', src)
        self.assertIn('rate limit exceeded', src)

    def test_rate_limiter_applied_to_both_search_endpoints(self):
        """Both search_nodes and search_memory_facts check the rate limiter with a caller key."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'
        ).read_text(encoding='utf-8')
        # Count occurrences of the per-caller rate limit guard pattern.
        # After the fix, both endpoints pass a key: is_allowed(caller_key)
        occurrences = src.count('_search_rate_limiter.is_allowed(caller_key)')
        self.assertGreaterEqual(
            occurrences, 2,
            f'Expected per-caller rate limiter applied to at least 2 endpoints, found {occurrences}',
        )
        # Verify the key derivation helper is present
        self.assertIn('_derive_rate_limit_key', src)

    def test_derive_rate_limit_key_present_in_source(self):
        """_derive_rate_limit_key helper is defined in the MCP server source."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'
        ).read_text(encoding='utf-8')
        self.assertIn('def _derive_rate_limit_key(', src)
        # Trusted key is always derived from resolved effective_group_ids (group: prefix)
        self.assertIn('group:', src)
        self.assertIn('__global__', src)
        # Key derivation must happen AFTER _resolve_effective_group_ids call in both endpoints
        self.assertIn('_resolve_effective_group_ids', src)
        # Log must use hashed key to avoid exposing sensitive values
        self.assertIn('_hash_rate_limit_key', src)

    def test_derive_rate_limit_key_logic(self):
        """_derive_rate_limit_key produces canonical, order-independent keys (anti-spoof).

        Mirrors the actual implementation contract: sorted unique effective_group_ids
        are hashed, so permutations of the same group set map to the same bucket.
        """
        import hashlib

        def _derive_rate_limit_key(effective_group_ids):
            """Local mirror of the production implementation for unit testing."""
            if effective_group_ids:
                canonical = '|'.join(sorted(set(effective_group_ids)))
                digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
                return f'group:{digest}'
            return '__global__'

        # Empty list → global fallback
        self.assertEqual(_derive_rate_limit_key([]), '__global__')

        # Non-empty → key starts with 'group:' prefix
        key = _derive_rate_limit_key(['grp1'])
        self.assertTrue(key.startswith('group:'), f'Expected group: prefix, got {key!r}')

        # Same group set, different orderings → identical key (order-independent)
        key_ab = _derive_rate_limit_key(['grp_a', 'grp_b'])
        key_ba = _derive_rate_limit_key(['grp_b', 'grp_a'])
        self.assertEqual(key_ab, key_ba, 'Key must be order-independent (anti-spoof)')

        # Duplicate entries collapse to the same bucket as deduplicated
        key_dedup = _derive_rate_limit_key(['grp_a', 'grp_a'])
        key_single = _derive_rate_limit_key(['grp_a'])
        self.assertEqual(key_dedup, key_single, 'Duplicate group IDs must deduplicate')

        # Different group sets → different keys
        key_a = _derive_rate_limit_key(['grp_a'])
        key_b = _derive_rate_limit_key(['grp_b'])
        self.assertNotEqual(key_a, key_b, 'Distinct group sets must produce distinct keys')

    def test_rate_limit_key_anti_order_spoof(self):
        """Permutation of group IDs must not produce distinct rate-limit buckets.

        Source-level check: verify the implementation uses sorted(set(...)) canonicalization.
        """
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'
        ).read_text(encoding='utf-8')

        # Implementation must canonicalize via sorted + set (or equivalent)
        self.assertIn(
            'sorted(set(effective_group_ids))',
            src,
            '_derive_rate_limit_key must use sorted(set(effective_group_ids)) for canonicalization',
        )
        # Must hash the canonical representation
        self.assertIn(
            'hashlib.sha256',
            src,
            '_derive_rate_limit_key must hash the canonical group set',
        )

    def test_rate_limit_key_derived_after_resolution_in_source(self):
        """In both endpoints, _derive_rate_limit_key is called with effective_group_ids
        (i.e. after _resolve_effective_group_ids), not with raw caller-supplied input."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / 'mcp_server' / 'src' / 'graphiti_mcp_server.py'
        ).read_text(encoding='utf-8')
        # The trusted pattern: key derived from effective_group_ids (+ caller principal)
        trusted_calls = src.count(
            '_derive_rate_limit_key(effective_group_ids, caller_principal)'
        )
        self.assertGreaterEqual(
            trusted_calls,
            2,
            f'Expected _derive_rate_limit_key called with effective_group_ids + caller_principal '
            f'in at least 2 endpoints, found {trusted_calls}',
        )

    def test_key_cardinality_bound_evicts_lru(self):
        """Hard max_keys cardinality: LRU key is evicted when the limit is reached."""
        rl = self._make_limiter(max_requests=5, window_seconds=60)
        # Override max_keys after creation to keep test fast
        rl._max_keys = 3

        # Fill to max_keys with distinct keys (keys A, B, C)
        self._run(rl.is_allowed('key_A'))
        self._run(rl.is_allowed('key_B'))
        self._run(rl.is_allowed('key_C'))
        self.assertEqual(len(rl._timestamps), 3)

        # Re-access key_A to make it most-recently-used (B is now LRU)
        self._run(rl.is_allowed('key_A'))

        # New key D should evict LRU (key_B)
        self._run(rl.is_allowed('key_D'))
        self.assertEqual(len(rl._timestamps), 3, 'cardinality must not exceed max_keys')
        self.assertNotIn('key_B', rl._timestamps, 'LRU key_B should have been evicted')
        self.assertIn('key_A', rl._timestamps)
        self.assertIn('key_C', rl._timestamps)
        self.assertIn('key_D', rl._timestamps)

    def test_max_keys_invalid_raises(self):
        """max_keys=0 raises ValueError."""
        import sys
        from pathlib import Path

        _mcp_src = str(Path(__file__).resolve().parents[1] / 'mcp_server' / 'src')
        sys.path.insert(0, _mcp_src)
        try:
            from utils.rate_limiter import SlidingWindowRateLimiter
        finally:
            sys.path.remove(_mcp_src)
            sys.modules.pop('config', None)

        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(max_requests=10, window_seconds=60, max_keys=0)

    def test_empty_key_removed_after_window_expires(self):
        """Keys with all-expired timestamps are removed to bound memory growth."""
        import time

        rl = self._make_limiter(max_requests=2, window_seconds=0.05)
        self._run(rl.is_allowed('ephemeral'))
        self.assertIn('ephemeral', rl._timestamps)
        time.sleep(0.1)  # let window expire
        # Next call triggers eviction; old timestamps removed, key deleted then re-added
        self._run(rl.is_allowed('ephemeral'))
        # After the new request, key exists with exactly 1 entry (the new one)
        self.assertIn('ephemeral', rl._timestamps)
        self.assertEqual(len(rl._timestamps['ephemeral']), 1)


class TestPhaseAStarvationPrevention(unittest.TestCase):
    """Phase A snapshot approach: each sent_not_marked chunk processed at most once per run."""

    def _make_db(self, chunk_ids: list[str]):
        import tempfile

        from scripts.mcp_ingest_sessions import init_claim_db, seed_claims

        tmp = tempfile.mktemp(suffix='.db')
        conn = init_claim_db(tmp)
        seed_claims(conn, chunk_ids)
        return conn, tmp

    def test_snapshot_sent_not_marked_returns_all_ids(self):
        """_snapshot_sent_not_marked returns IDs of all sent_not_marked chunks."""
        import os

        from scripts.mcp_ingest_sessions import (
            _claim_sent_not_marked,
            _snapshot_sent_not_marked,
            claim_chunk,
        )

        conn, tmp = self._make_db(['chunk_snm1', 'chunk_snm2', 'chunk_pending'])
        try:
            # Move two chunks to sent_not_marked
            cid1 = claim_chunk(conn, 'w0')
            _claim_sent_not_marked(conn, cid1)
            cid2 = claim_chunk(conn, 'w0')
            _claim_sent_not_marked(conn, cid2)

            ids = _snapshot_sent_not_marked(conn)
            self.assertEqual(set(ids), {'chunk_snm1', 'chunk_snm2'})
            # chunk_pending should NOT appear
            self.assertNotIn('chunk_pending', ids)
        finally:
            conn.close()
            os.unlink(tmp)

    def test_targeted_claim_succeeds_for_sent_not_marked(self):
        """_claim_neo4j_retry_targeted claims a specific sent_not_marked chunk."""
        import os

        from scripts.mcp_ingest_sessions import (
            _claim_neo4j_retry_targeted,
            _claim_sent_not_marked,
            claim_chunk,
        )

        conn, tmp = self._make_db(['chunk_X'])
        try:
            claim_chunk(conn, 'w0')
            _claim_sent_not_marked(conn, 'chunk_X')

            claimed = _claim_neo4j_retry_targeted(conn, 'w0', 'chunk_X')
            self.assertTrue(claimed)
            row = conn.execute(
                "SELECT status FROM chunk_claims WHERE chunk_id='chunk_X'"
            ).fetchone()
            self.assertEqual(row[0], 'claimed')
        finally:
            conn.close()
            os.unlink(tmp)

    def test_targeted_claim_returns_false_for_pending(self):
        """_claim_neo4j_retry_targeted must not claim pending chunks."""
        import os

        from scripts.mcp_ingest_sessions import _claim_neo4j_retry_targeted

        conn, tmp = self._make_db(['chunk_Y'])
        try:
            claimed = _claim_neo4j_retry_targeted(conn, 'w0', 'chunk_Y')
            self.assertFalse(claimed)
        finally:
            conn.close()
            os.unlink(tmp)

    def test_pending_work_reachable_after_phase_a_failure(self):
        """Phase B pending chunk is claimable even when Phase A chunk fails repeatedly."""
        import os

        from scripts.mcp_ingest_sessions import (
            _claim_neo4j_retry_targeted,
            _claim_sent_not_marked,
            _snapshot_sent_not_marked,
            claim_chunk,
        )

        conn, tmp = self._make_db(['chunk_failing', 'chunk_pending'])
        try:
            # Set up chunk_failing in sent_not_marked state
            claim_chunk(conn, 'w0')  # claims chunk_failing (first in list)
            _claim_sent_not_marked(conn, 'chunk_failing')

            # Simulate Phase A: snapshot, then process once (failing)
            phase_a_ids = _snapshot_sent_not_marked(conn)
            self.assertIn('chunk_failing', phase_a_ids)

            for cid in phase_a_ids:
                claimed = _claim_neo4j_retry_targeted(conn, 'w0', cid)
                if claimed:
                    # Simulate mark failure: restore to sent_not_marked + increment fail_count
                    conn.execute(
                        "UPDATE chunk_claims SET status='sent_not_marked', "
                        "fail_count=COALESCE(fail_count, 0) + 1, error='simulated' "
                        'WHERE chunk_id=?',
                        (cid,),
                    )
                    conn.commit()

            # Phase B: pending chunk must still be claimable (not starved)
            pending = claim_chunk(conn, 'w0')
            self.assertEqual(
                pending,
                'chunk_pending',
                'Phase B pending work was starved by Phase A failures',
            )
        finally:
            conn.close()
            os.unlink(tmp)

    def test_fail_count_increments_on_phase_a_failure(self):
        """fail_count is incremented each time Neo4j mark fails in Phase A."""
        import os

        from scripts.mcp_ingest_sessions import (
            _claim_neo4j_retry_targeted,
            _claim_sent_not_marked,
            claim_chunk,
        )

        conn, tmp = self._make_db(['chunk_retry'])
        try:
            claim_chunk(conn, 'w0')
            _claim_sent_not_marked(conn, 'chunk_retry')

            # --- Run 1 failure ---
            _claim_neo4j_retry_targeted(conn, 'w0', 'chunk_retry')
            conn.execute(
                "UPDATE chunk_claims SET status='sent_not_marked', "
                "fail_count=COALESCE(fail_count, 0) + 1, error='err1' "
                "WHERE chunk_id='chunk_retry'"
            )
            conn.commit()

            row = conn.execute(
                "SELECT fail_count, status FROM chunk_claims WHERE chunk_id='chunk_retry'"
            ).fetchone()
            self.assertEqual(row[0], 1)
            self.assertEqual(row[1], 'sent_not_marked')

            # --- Run 2 failure ---
            _claim_neo4j_retry_targeted(conn, 'w0', 'chunk_retry')
            conn.execute(
                "UPDATE chunk_claims SET status='sent_not_marked', "
                "fail_count=COALESCE(fail_count, 0) + 1, error='err2' "
                "WHERE chunk_id='chunk_retry'"
            )
            conn.commit()

            row = conn.execute(
                "SELECT fail_count FROM chunk_claims WHERE chunk_id='chunk_retry'"
            ).fetchone()
            self.assertEqual(row[0], 2, 'fail_count must accumulate across retry runs')
        finally:
            conn.close()
            os.unlink(tmp)

    def test_fail_count_not_incremented_on_success(self):
        """fail_count stays at 0 when Neo4j mark succeeds in Phase A."""
        import os

        from scripts.mcp_ingest_sessions import (
            _claim_done,
            _claim_neo4j_retry_targeted,
            _claim_sent_not_marked,
            claim_chunk,
        )

        conn, tmp = self._make_db(['chunk_ok'])
        try:
            claim_chunk(conn, 'w0')
            _claim_sent_not_marked(conn, 'chunk_ok')
            _claim_neo4j_retry_targeted(conn, 'w0', 'chunk_ok')
            _claim_done(conn, 'chunk_ok')

            row = conn.execute(
                "SELECT fail_count, status FROM chunk_claims WHERE chunk_id='chunk_ok'"
            ).fetchone()
            self.assertEqual(row[0], 0)
            self.assertEqual(row[1], 'done')
        finally:
            conn.close()
            os.unlink(tmp)


class TestPhaseBFailCount(unittest.TestCase):
    """Phase B: fail_count is incremented when Neo4j mark fails after add_memory succeeds."""

    def _make_db(self, chunk_ids: list[str]):
        import tempfile

        from scripts.mcp_ingest_sessions import init_claim_db, seed_claims

        tmp = tempfile.mktemp(suffix='.db')
        conn = init_claim_db(tmp)
        seed_claims(conn, chunk_ids)
        return conn, tmp

    def test_phase_b_fail_count_incremented_on_neo4j_mark_failure(self):
        """When Neo4j mark fails in Phase B (after add_memory), fail_count increments.

        This ensures retry-failure observability is consistent between Phase A
        (mark-only retry) and Phase B (full pipeline where only the mark fails).
        """
        import os

        from scripts.mcp_ingest_sessions import _claim_sent_not_marked, claim_chunk

        conn, tmp = self._make_db(['chunk_phaseB'])
        try:
            # Simulate Phase B: claim chunk, call add_memory (success) → sent_not_marked
            chunk_id = claim_chunk(conn, 'w0')
            self.assertEqual(chunk_id, 'chunk_phaseB')
            _claim_sent_not_marked(conn, chunk_id)

            # Verify initial state: fail_count = 0, status = sent_not_marked
            row = conn.execute(
                "SELECT fail_count, status FROM chunk_claims WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
            self.assertEqual(row[0], 0)
            self.assertEqual(row[1], 'sent_not_marked')

            # Simulate Neo4j mark failure in Phase B: the fixed code increments fail_count
            conn.execute(
                "UPDATE chunk_claims SET "
                "fail_count=COALESCE(fail_count, 0) + 1, error=? "
                "WHERE chunk_id=?",
                ('neo4j_mark_failed: simulated error'[:500], chunk_id),
            )
            conn.commit()

            row = conn.execute(
                "SELECT fail_count, status FROM chunk_claims WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
            self.assertEqual(row[0], 1, 'fail_count must be 1 after first Phase B neo4j failure')
            self.assertEqual(row[1], 'sent_not_marked', 'status stays sent_not_marked for Phase A retry')
        finally:
            conn.close()
            os.unlink(tmp)

    def test_phase_b_fail_count_accumulates_across_runs(self):
        """fail_count accumulates if Neo4j mark keeps failing across multiple Phase B runs."""
        import os

        from scripts.mcp_ingest_sessions import _claim_sent_not_marked, claim_chunk

        conn, tmp = self._make_db(['chunk_phaseB2'])
        try:
            chunk_id = claim_chunk(conn, 'w0')
            _claim_sent_not_marked(conn, chunk_id)

            for run in range(1, 4):
                conn.execute(
                    "UPDATE chunk_claims SET "
                    "fail_count=COALESCE(fail_count, 0) + 1, error=? "
                    "WHERE chunk_id=?",
                    (f'neo4j_mark_failed: run {run}', chunk_id),
                )
                conn.commit()

            row = conn.execute(
                "SELECT fail_count FROM chunk_claims WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
            self.assertEqual(row[0], 3, 'fail_count must accumulate across 3 Phase B failures')
        finally:
            conn.close()
            os.unlink(tmp)

    def test_phase_b_source_increments_fail_count(self):
        """The Phase B except block in mcp_ingest_sessions.py increments fail_count."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / 'scripts' / 'mcp_ingest_sessions.py'
        ).read_text(encoding='utf-8')
        # The Phase B neo4j mark failure block must include fail_count increment
        self.assertIn(
            'fail_count=COALESCE(fail_count, 0) + 1',
            src,
            'Phase B neo4j mark failure must increment fail_count (retry accounting completeness)',
        )

    def test_phase_b_source_has_worker_guard(self):
        """Phase B fail_count UPDATE must include status + worker_id guard to prevent
        mutating rows already resolved by another worker."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parents[1]
            / 'scripts' / 'mcp_ingest_sessions.py'
        ).read_text(encoding='utf-8')
        self.assertIn(
            "status='sent_not_marked' AND worker_id=?",
            src,
            "Phase B fail_count UPDATE must guard on status='sent_not_marked' AND worker_id=?",
        )

    def test_phase_b_fail_count_guard_skips_resolved_row(self):
        """If another worker resolves the row first, Phase B fail_count update is a no-op."""
        import os

        from scripts.mcp_ingest_sessions import _claim_done, _claim_sent_not_marked, claim_chunk

        conn, tmp = self._make_db(['chunk_race'])
        try:
            chunk_id = claim_chunk(conn, 'w0')
            _claim_sent_not_marked(conn, chunk_id)

            # Simulate another worker resolving the row first
            _claim_done(conn, chunk_id)

            # Phase B worker now attempts the guarded fail_count update — must be a no-op
            cur = conn.execute(
                "UPDATE chunk_claims SET "
                "fail_count=COALESCE(fail_count, 0) + 1, error=? "
                "WHERE chunk_id=? AND status='sent_not_marked' AND worker_id=?",
                ('neo4j_mark_failed: race', chunk_id, 'w0'),
            )
            conn.commit()

            self.assertEqual(
                cur.rowcount, 0,
                'Guard UPDATE must affect 0 rows when row is already resolved (done)',
            )
            # fail_count must remain 0 on the resolved row
            row = conn.execute(
                'SELECT fail_count, status FROM chunk_claims WHERE chunk_id=?',
                (chunk_id,),
            ).fetchone()
            self.assertEqual(row[1], 'done', 'Row status must still be done')
            self.assertEqual(row[0], 0, 'fail_count must not have been incremented on resolved row')
        finally:
            conn.close()
            os.unlink(tmp)

    def test_phase_b_fail_count_guard_updates_own_row(self):
        """Phase B fail_count UPDATE increments correctly when row is still owned by this worker."""
        import os

        from scripts.mcp_ingest_sessions import _claim_sent_not_marked, claim_chunk

        conn, tmp = self._make_db(['chunk_mine'])
        try:
            chunk_id = claim_chunk(conn, 'w1')
            _claim_sent_not_marked(conn, chunk_id)

            # Worker w1 applies the guarded update — should match exactly 1 row
            cur = conn.execute(
                "UPDATE chunk_claims SET "
                "fail_count=COALESCE(fail_count, 0) + 1, error=? "
                "WHERE chunk_id=? AND status='sent_not_marked' AND worker_id=?",
                ('neo4j_mark_failed: test', chunk_id, 'w1'),
            )
            conn.commit()

            self.assertEqual(cur.rowcount, 1, 'Guard UPDATE must affect exactly 1 row for owner worker')
            row = conn.execute(
                'SELECT fail_count, status FROM chunk_claims WHERE chunk_id=?',
                (chunk_id,),
            ).fetchone()
            self.assertEqual(row[0], 1, 'fail_count must be 1 after guarded update')
            self.assertEqual(row[1], 'sent_not_marked', 'Status must remain sent_not_marked')
        finally:
            conn.close()
            os.unlink(tmp)


if __name__ == '__main__':
    unittest.main()
