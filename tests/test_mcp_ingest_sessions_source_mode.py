"""Tests for mcp_ingest_sessions.py --source-mode (FR-4 + FR-10)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        import sqlite3
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
        import sqlite3
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


if __name__ == '__main__':
    unittest.main()
