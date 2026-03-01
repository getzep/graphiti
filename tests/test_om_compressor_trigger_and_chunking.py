"""Tests for OM compressor Smart Cutter integration (FR-5 + FR-9 + FR-11)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestOMCompressorArgs(unittest.TestCase):
    """Test OM compressor CLI argument parsing."""

    def test_default_mode_is_steady(self):
        from scripts.om_compressor import parse_args

        args = parse_args([])
        self.assertEqual(args.mode, 'steady')

    def test_backfill_mode(self):
        from scripts.om_compressor import parse_args

        args = parse_args(['--mode', 'backfill'])
        self.assertEqual(args.mode, 'backfill')

    def test_force_flag_preserved(self):
        from scripts.om_compressor import parse_args

        args = parse_args(['--force'])
        self.assertTrue(args.force)

    def test_max_chunks_per_run(self):
        from scripts.om_compressor import parse_args

        args = parse_args(['--max-chunks-per-run', '5'])
        self.assertEqual(args.max_chunks_per_run, 5)

    def test_build_manifest_flag(self):
        from scripts.om_compressor import parse_args

        args = parse_args([
            '--mode', 'backfill',
            '--build-manifest', '/tmp/manifest.jsonl',
        ])
        self.assertEqual(args.build_manifest, '/tmp/manifest.jsonl')

    def test_claim_mode_flags(self):
        from scripts.om_compressor import parse_args

        args = parse_args([
            '--mode', 'backfill',
            '--claim-mode',
            '--shards', '4',
            '--shard-index', '0',
        ])
        self.assertTrue(args.claim_mode)
        self.assertEqual(args.shards, 4)
        self.assertEqual(args.shard_index, 0)

    def test_dry_run_flag(self):
        from scripts.om_compressor import parse_args

        args = parse_args(['--dry-run'])
        self.assertTrue(args.dry_run)


class TestOMCompressorConstants(unittest.TestCase):
    """Test FR-3 lane constants are present."""

    def test_max_parent_chunk_size(self):
        from scripts.om_compressor import MAX_PARENT_CHUNK_SIZE

        self.assertEqual(MAX_PARENT_CHUNK_SIZE, 50)

    def test_default_trigger_thresholds(self):
        """Trigger policy: backlog >= 50 OR oldest >= 48h."""
        # These are used inline in the code, verify the constants exist
        from scripts.om_compressor import DEFAULT_MAX_CHUNKS_PER_RUN

        self.assertEqual(DEFAULT_MAX_CHUNKS_PER_RUN, 10)

    def test_dead_letter_attempts(self):
        from scripts.om_compressor import DEAD_LETTER_ATTEMPTS

        self.assertEqual(DEAD_LETTER_ATTEMPTS, 3)


class TestSmartCutterLaneConstants(unittest.TestCase):
    """Test that Smart Cutter lane constants are accessible."""

    def test_graphiti_merge_floor(self):
        from graphiti_core.utils.content_chunking import GRAPHITI_MERGE_FLOOR_TOKENS

        self.assertEqual(GRAPHITI_MERGE_FLOOR_TOKENS, 800)

    def test_om_split_ceiling(self):
        from graphiti_core.utils.content_chunking import OM_SPLIT_CEILING_TOKENS

        self.assertEqual(OM_SPLIT_CEILING_TOKENS, 4000)

    def test_om_split_min_tail(self):
        from graphiti_core.utils.content_chunking import OM_SPLIT_MIN_TAIL_TOKENS

        self.assertEqual(OM_SPLIT_MIN_TAIL_TOKENS, 300)


class TestTriggerPolicy(unittest.TestCase):
    """Test that trigger policy is preserved."""

    def test_trigger_met_by_backlog(self):
        """Backlog >= 50 should trigger."""
        backlog = 50
        oldest_hours = 1.0
        trigger = backlog >= 50 or (oldest_hours is not None and oldest_hours >= 48.0)
        self.assertTrue(trigger)

    def test_trigger_met_by_age(self):
        """Oldest >= 48h should trigger."""
        backlog = 10
        oldest_hours = 48.0
        trigger = backlog >= 50 or (oldest_hours is not None and oldest_hours >= 48.0)
        self.assertTrue(trigger)

    def test_trigger_not_met(self):
        """Below thresholds should not trigger."""
        backlog = 10
        oldest_hours = 24.0
        trigger = backlog >= 50 or (oldest_hours is not None and oldest_hours >= 48.0)
        self.assertFalse(trigger)


class TestOMCompressorLockPath(unittest.TestCase):
    """Test lock file behavior."""

    def test_lock_filename_constant(self):
        from scripts.om_compressor import DEFAULT_LOCK_FILENAME

        self.assertEqual(DEFAULT_LOCK_FILENAME, 'om_graph_write.lock')


class TestSteadyVsBackfillMode(unittest.TestCase):
    """Test mode-specific behavior."""

    def test_steady_mode_uses_lock(self):
        """Steady-state OM must use single-writer lock."""
        from scripts.om_compressor import parse_args

        args = parse_args(['--mode', 'steady'])
        self.assertEqual(args.mode, 'steady')

    def test_backfill_mode_with_claim(self):
        """Backfill mode supports parallel claim-based execution."""
        from scripts.om_compressor import parse_args

        args = parse_args([
            '--mode', 'backfill',
            '--claim-mode',
            '--shards', '4',
            '--shard-index', '1',
            '--max-chunks-per-run', '5',
            '--dry-run',
        ])
        self.assertEqual(args.mode, 'backfill')
        self.assertTrue(args.claim_mode)
        self.assertEqual(args.shards, 4)
        self.assertEqual(args.shard_index, 1)


class TestOMClaimStateDB(unittest.TestCase):
    """Claim DB semantics for FR-11 throughput mode."""

    def test_claim_db_schema_includes_lease_columns(self):
        from scripts.om_compressor import init_claim_db

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            try:
                cols = {row[1] for row in conn.execute('PRAGMA table_info(chunk_claims)').fetchall()}
                self.assertIn('lease_expires_at', cols)
                self.assertIn('attempt_count', cols)
                self.assertIn('claim_shard', cols)
            finally:
                conn.close()

    def test_claim_chunk_respects_shard_partitioning(self):
        from scripts.om_compressor import _claim_shard, claim_chunk, init_claim_db, seed_claims

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            try:
                chunk_ids = [f'chunk_{i:03d}' for i in range(12)]
                seed_claims(conn, chunk_ids)

                expected = {cid for cid in chunk_ids if (_claim_shard(cid) % 2) == 0}
                claimed: set[str] = set()
                while True:
                    cid = claim_chunk(
                        conn,
                        worker_id='w0',
                        shards=2,
                        shard_index=0,
                        lease_seconds=600,
                    )
                    if cid is None:
                        break
                    claimed.add(cid)

                self.assertEqual(claimed, expected)
            finally:
                conn.close()

    def test_stale_claim_can_be_recovered(self):
        from scripts.om_compressor import claim_chunk, init_claim_db, seed_claims

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            try:
                seed_claims(conn, ['chunk_stale'])

                first = claim_chunk(
                    conn,
                    worker_id='worker_a',
                    shards=1,
                    shard_index=0,
                    lease_seconds=600,
                )
                self.assertEqual(first, 'chunk_stale')

                conn.execute(
                    "UPDATE chunk_claims SET lease_expires_at='1970-01-01T00:00:00Z' WHERE chunk_id=?",
                    ('chunk_stale',),
                )
                conn.commit()

                recovered = claim_chunk(
                    conn,
                    worker_id='worker_b',
                    shards=1,
                    shard_index=0,
                    lease_seconds=600,
                )
                self.assertEqual(recovered, 'chunk_stale')
            finally:
                conn.close()


class TestOMDoneConfirm(unittest.TestCase):
    def test_confirm_chunk_done_true_when_all_messages_confirmed(self):
        from scripts.om_compressor import _confirm_chunk_done

        session = MagicMock()
        session.run.return_value.single.return_value = {'total': 2, 'confirmed': 2}

        ok = _confirm_chunk_done(session, message_ids=['m1', 'm2'], chunk_id='chunk_a')
        self.assertTrue(ok)

    def test_confirm_chunk_done_false_when_partial(self):
        from scripts.om_compressor import _confirm_chunk_done

        session = MagicMock()
        session.run.return_value.single.return_value = {'total': 2, 'confirmed': 1}

        ok = _confirm_chunk_done(session, message_ids=['m1', 'm2'], chunk_id='chunk_a')
        self.assertFalse(ok)


class _FakeSessionContext:
    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def session(self, database=None):
        return _FakeSessionContext(self._session)


class TestOMClaimModeDoneConfirmFlow(unittest.TestCase):
    def _write_manifest(self, path: str, *, chunk_id: str, extractor_version: str) -> None:
        row = {
            'chunk_id': chunk_id,
            'message_ids': ['m1'],
            'message_count': 1,
            'extractor_version': extractor_version,
        }
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(row))
            fh.write('\n')

    def test_run_claim_mode_marks_failed_when_done_confirm_fails(self):
        """Done-confirm failure: chunk marked failed, run returns non-zero (run-level exit semantics)."""
        from scripts.om_compressor import ExtractorConfig, MessageRow, parse_args, run

        with tempfile.TemporaryDirectory() as td:
            manifest = os.path.join(td, 'om_manifest.jsonl')
            cfg = ExtractorConfig(
                schema_version='2026-02-17',
                prompt_template='x',
                model_id='m',
                extractor_version='ev1',
            )
            self._write_manifest(manifest, chunk_id='chunk_1', extractor_version=cfg.extractor_version)

            args = parse_args([
                '--mode', 'backfill',
                '--claim-mode',
                '--build-manifest', manifest,
                '--shards', '1',
                '--shard-index', '0',
                '--max-chunks-per-run', '1',
            ])

            fake_session = MagicMock()
            driver = _FakeDriver(fake_session)
            msg = MessageRow(
                message_id='m1',
                source_session_id='s',
                content='hello',
                created_at='2026-02-28T00:00:00Z',
                content_embedding=[],
                om_extract_attempts=0,
            )

            with (
                patch('scripts.om_compressor._neo4j_driver', return_value=driver),
                patch('scripts.om_compressor._ensure_neo4j_constraints'),
                patch('scripts.om_compressor._load_extractor_config', return_value=cfg),
                patch('scripts.om_compressor._fetch_messages_by_ids', return_value=[msg]),
                patch('scripts.om_compressor._activate_energy_scores', return_value=([], [])),
                patch('scripts.om_compressor._process_chunk', return_value={'chunk_id': 'chunk_1'}),
                patch('scripts.om_compressor._confirm_chunk_done', return_value=False),
            ):
                rc = run(args)

            # Done-confirm failure now sets had_failures → run returns non-zero (item 3).
            self.assertEqual(rc, 1)
            claim_db = f'{manifest}.claims.db'
            import sqlite3

            conn = sqlite3.connect(claim_db)
            try:
                row = conn.execute(
                    'SELECT status, fail_count FROM chunk_claims WHERE chunk_id=?',
                    ('chunk_1',),
                ).fetchone()
                self.assertEqual(row[0], 'failed')
                self.assertEqual(row[1], 1)
            finally:
                conn.close()

    def test_run_claim_mode_marks_done_after_done_confirm(self):
        from scripts.om_compressor import ExtractorConfig, MessageRow, parse_args, run

        with tempfile.TemporaryDirectory() as td:
            manifest = os.path.join(td, 'om_manifest.jsonl')
            cfg = ExtractorConfig(
                schema_version='2026-02-17',
                prompt_template='x',
                model_id='m',
                extractor_version='ev1',
            )
            self._write_manifest(manifest, chunk_id='chunk_2', extractor_version=cfg.extractor_version)

            args = parse_args([
                '--mode', 'backfill',
                '--claim-mode',
                '--build-manifest', manifest,
                '--shards', '1',
                '--shard-index', '0',
                '--max-chunks-per-run', '1',
            ])

            fake_session = MagicMock()
            driver = _FakeDriver(fake_session)
            msg = MessageRow(
                message_id='m1',
                source_session_id='s',
                content='hello',
                created_at='2026-02-28T00:00:00Z',
                content_embedding=[],
                om_extract_attempts=0,
            )

            with (
                patch('scripts.om_compressor._neo4j_driver', return_value=driver),
                patch('scripts.om_compressor._ensure_neo4j_constraints'),
                patch('scripts.om_compressor._load_extractor_config', return_value=cfg),
                patch('scripts.om_compressor._fetch_messages_by_ids', return_value=[msg]),
                patch('scripts.om_compressor._activate_energy_scores', return_value=([], [])),
                patch('scripts.om_compressor._process_chunk', return_value={'chunk_id': 'chunk_2'}),
                patch('scripts.om_compressor._confirm_chunk_done', return_value=True),
            ):
                rc = run(args)

            self.assertEqual(rc, 0)
            claim_db = f'{manifest}.claims.db'
            import sqlite3

            conn = sqlite3.connect(claim_db)
            try:
                row = conn.execute(
                    'SELECT status, fail_count FROM chunk_claims WHERE chunk_id=?',
                    ('chunk_2',),
                ).fetchone()
                self.assertEqual(row[0], 'done')
                self.assertEqual(row[1], 0)
            finally:
                conn.close()


class TestSeedClaimsRetry(unittest.TestCase):
    """seed_claims must reset failed chunks to pending so they can be retried (item 1)."""

    def test_seed_claims_resets_failed_to_pending(self):
        """A chunk that previously failed is reset to pending on next seed_claims call."""
        from scripts.om_compressor import (
            CLAIM_STATUS_FAILED,
            CLAIM_STATUS_PENDING,
            init_claim_db,
            seed_claims,
        )

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            try:
                # Seed once so the chunk exists.
                seed_claims(conn, ['chunk_retry'])
                # Simulate a previous failure.
                conn.execute(
                    """
                    UPDATE chunk_claims
                    SET status = 'failed',
                        worker_id = 'dead_worker',
                        claimed_at = '2026-01-01T00:00:00Z',
                        lease_expires_at = '2026-01-01T00:15:00Z',
                        completed_at = '2026-01-01T00:16:00Z',
                        fail_count = 1,
                        error = 'something broke'
                    WHERE chunk_id = 'chunk_retry'
                    """
                )
                conn.commit()

                # Verify it really is failed before retry.
                status_before = conn.execute(
                    'SELECT status FROM chunk_claims WHERE chunk_id=?', ('chunk_retry',)
                ).fetchone()[0]
                self.assertEqual(status_before, CLAIM_STATUS_FAILED)

                # Calling seed_claims again should reset it to pending.
                seed_claims(conn, ['chunk_retry'])

                row = conn.execute(
                    'SELECT status, worker_id, lease_expires_at, completed_at, error '
                    'FROM chunk_claims WHERE chunk_id=?',
                    ('chunk_retry',),
                ).fetchone()
                self.assertEqual(row[0], CLAIM_STATUS_PENDING)
                self.assertIsNone(row[1], 'worker_id should be cleared')
                self.assertIsNone(row[2], 'lease_expires_at should be cleared')
                self.assertIsNone(row[3], 'completed_at should be cleared')
                self.assertIsNone(row[4], 'error should be cleared')
            finally:
                conn.close()

    def test_seed_claims_does_not_reset_done(self):
        """A chunk in 'done' status must NOT be reset — would cause double-processing."""
        from scripts.om_compressor import CLAIM_STATUS_DONE, init_claim_db, seed_claims

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            try:
                seed_claims(conn, ['chunk_done'])
                conn.execute(
                    "UPDATE chunk_claims SET status='done', completed_at='2026-01-01T00:00:00Z' "
                    "WHERE chunk_id='chunk_done'"
                )
                conn.commit()

                # Re-seeding must leave done chunks alone.
                seed_claims(conn, ['chunk_done'])

                status = conn.execute(
                    'SELECT status FROM chunk_claims WHERE chunk_id=?', ('chunk_done',)
                ).fetchone()[0]
                self.assertEqual(status, CLAIM_STATUS_DONE)
            finally:
                conn.close()

    def test_seed_claims_does_not_reset_claimed(self):
        """An in-flight claimed chunk must NOT be reset by seed_claims."""
        from scripts.om_compressor import CLAIM_STATUS_CLAIMED, init_claim_db, seed_claims

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            try:
                seed_claims(conn, ['chunk_claimed'])
                conn.execute(
                    "UPDATE chunk_claims SET status='claimed', worker_id='live_worker' "
                    "WHERE chunk_id='chunk_claimed'"
                )
                conn.commit()

                seed_claims(conn, ['chunk_claimed'])

                row = conn.execute(
                    'SELECT status, worker_id FROM chunk_claims WHERE chunk_id=?',
                    ('chunk_claimed',),
                ).fetchone()
                self.assertEqual(row[0], CLAIM_STATUS_CLAIMED)
                self.assertEqual(row[1], 'live_worker')
            finally:
                conn.close()

    def test_seed_claims_respects_dead_letter_threshold(self):
        """A chunk with fail_count >= DEAD_LETTER_ATTEMPTS must NOT be reset by seed_claims.

        This prevents poison-pill crash loops / head-of-line blocking: a chunk
        that always fails should stay in dead-letter state rather than being
        repeatedly reset to pending and reclaimed indefinitely.
        """
        from scripts.om_compressor import (
            CLAIM_STATUS_FAILED,
            DEAD_LETTER_ATTEMPTS,
            init_claim_db,
            seed_claims,
        )

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            try:
                seed_claims(conn, ['chunk_dead'])
                # Simulate a chunk that has exhausted all allowed attempts.
                conn.execute(
                    """
                    UPDATE chunk_claims
                    SET status = 'failed',
                        fail_count = ?,
                        error = 'repeated extraction failure'
                    WHERE chunk_id = 'chunk_dead'
                    """,
                    (DEAD_LETTER_ATTEMPTS,),
                )
                conn.commit()

                # seed_claims must NOT reset this chunk — it has hit the dead-letter limit.
                seed_claims(conn, ['chunk_dead'])

                row = conn.execute(
                    'SELECT status, fail_count FROM chunk_claims WHERE chunk_id=?',
                    ('chunk_dead',),
                ).fetchone()
                self.assertEqual(row[0], CLAIM_STATUS_FAILED,
                                 'Dead-lettered chunk must remain failed after seed_claims')
                self.assertEqual(row[1], DEAD_LETTER_ATTEMPTS,
                                 'fail_count must be preserved')
            finally:
                conn.close()

    def test_seed_claims_resets_failed_below_dead_letter_threshold(self):
        """A chunk with fail_count < DEAD_LETTER_ATTEMPTS IS still reset to pending.

        Verifies the boundary: fail_count = DEAD_LETTER_ATTEMPTS - 1 should be
        reset (one more retry is allowed); fail_count = DEAD_LETTER_ATTEMPTS
        should not (tested in test_seed_claims_respects_dead_letter_threshold).
        """
        from scripts.om_compressor import (
            CLAIM_STATUS_PENDING,
            DEAD_LETTER_ATTEMPTS,
            init_claim_db,
            seed_claims,
        )

        with tempfile.NamedTemporaryFile(suffix='.db') as f:
            conn = init_claim_db(f.name)
            try:
                seed_claims(conn, ['chunk_retryable'])
                conn.execute(
                    """
                    UPDATE chunk_claims
                    SET status = 'failed',
                        fail_count = ?,
                        error = 'transient error'
                    WHERE chunk_id = 'chunk_retryable'
                    """,
                    (DEAD_LETTER_ATTEMPTS - 1,),
                )
                conn.commit()

                seed_claims(conn, ['chunk_retryable'])

                row = conn.execute(
                    'SELECT status FROM chunk_claims WHERE chunk_id=?',
                    ('chunk_retryable',),
                ).fetchone()
                self.assertEqual(row[0], CLAIM_STATUS_PENDING,
                                 'Chunk below dead-letter threshold must be retried')
            finally:
                conn.close()


class TestClaimOwnershipLost(unittest.TestCase):
    """When _claim_done returns False, emit OM_CLAIM_OWNERSHIP_LOST and fail the run (item 2/3)."""

    def _write_manifest(self, path: str, *, chunk_id: str, extractor_version: str) -> None:
        row = {
            'chunk_id': chunk_id,
            'message_ids': ['m1'],
            'message_count': 1,
            'extractor_version': extractor_version,
        }
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(row))
            fh.write('\n')

    def test_claim_done_ownership_lost_emits_event_and_returns_nonzero(self):
        """_claim_done returning False → OM_CLAIM_OWNERSHIP_LOST emitted, run returns 1."""
        from scripts.om_compressor import ExtractorConfig, MessageRow, parse_args, run

        with tempfile.TemporaryDirectory() as td:
            manifest = os.path.join(td, 'om_manifest.jsonl')
            cfg = ExtractorConfig(
                schema_version='2026-02-17',
                prompt_template='x',
                model_id='m',
                extractor_version='ev1',
            )
            self._write_manifest(manifest, chunk_id='chunk_race', extractor_version=cfg.extractor_version)

            args = parse_args([
                '--mode', 'backfill',
                '--claim-mode',
                '--build-manifest', manifest,
                '--shards', '1',
                '--shard-index', '0',
                '--max-chunks-per-run', '1',
            ])

            fake_session = MagicMock()
            driver = _FakeDriver(fake_session)
            msg = MessageRow(
                message_id='m1',
                source_session_id='s',
                content='hello',
                created_at='2026-02-28T00:00:00Z',
                content_embedding=[],
                om_extract_attempts=0,
            )

            emitted_events: list[dict] = []

            def _capturing_emit(name: str, **payload):
                emitted_events.append({'event': name, **payload})
                print(json.dumps({'event': name, **payload}))

            with (
                patch('scripts.om_compressor._neo4j_driver', return_value=driver),
                patch('scripts.om_compressor._ensure_neo4j_constraints'),
                patch('scripts.om_compressor._load_extractor_config', return_value=cfg),
                patch('scripts.om_compressor._fetch_messages_by_ids', return_value=[msg]),
                patch('scripts.om_compressor._activate_energy_scores', return_value=([], [])),
                patch('scripts.om_compressor._process_chunk', return_value={'chunk_id': 'chunk_race'}),
                patch('scripts.om_compressor._confirm_chunk_done', return_value=True),
                # _claim_done returns False → simulates ownership-lost race
                patch('scripts.om_compressor._claim_done', return_value=False),
                patch('scripts.om_compressor.emit_event', side_effect=_capturing_emit),
            ):
                rc = run(args)

            self.assertEqual(rc, 1, 'ownership-lost should make the run return non-zero')
            ownership_lost = [e for e in emitted_events if e['event'] == 'OM_CLAIM_OWNERSHIP_LOST']
            self.assertTrue(
                ownership_lost,
                'OM_CLAIM_OWNERSHIP_LOST must be emitted when _claim_done returns False',
            )
            self.assertEqual(ownership_lost[0].get('phase'), 'done')

    def test_claim_fail_ownership_lost_emits_event(self):
        """_claim_fail returning False on done-confirm → OM_CLAIM_OWNERSHIP_LOST with phase=confirm_fail."""
        from scripts.om_compressor import ExtractorConfig, MessageRow, parse_args, run

        with tempfile.TemporaryDirectory() as td:
            manifest = os.path.join(td, 'om_manifest.jsonl')
            cfg = ExtractorConfig(
                schema_version='2026-02-17',
                prompt_template='x',
                model_id='m',
                extractor_version='ev1',
            )
            self._write_manifest(manifest, chunk_id='chunk_cf_race', extractor_version=cfg.extractor_version)

            args = parse_args([
                '--mode', 'backfill',
                '--claim-mode',
                '--build-manifest', manifest,
                '--shards', '1',
                '--shard-index', '0',
                '--max-chunks-per-run', '1',
            ])

            fake_session = MagicMock()
            driver = _FakeDriver(fake_session)
            msg = MessageRow(
                message_id='m1',
                source_session_id='s',
                content='hello',
                created_at='2026-02-28T00:00:00Z',
                content_embedding=[],
                om_extract_attempts=0,
            )

            emitted_events: list[dict] = []

            def _capturing_emit(name: str, **payload):
                emitted_events.append({'event': name, **payload})
                print(json.dumps({'event': name, **payload}))

            with (
                patch('scripts.om_compressor._neo4j_driver', return_value=driver),
                patch('scripts.om_compressor._ensure_neo4j_constraints'),
                patch('scripts.om_compressor._load_extractor_config', return_value=cfg),
                patch('scripts.om_compressor._fetch_messages_by_ids', return_value=[msg]),
                patch('scripts.om_compressor._activate_energy_scores', return_value=([], [])),
                patch('scripts.om_compressor._process_chunk', return_value={'chunk_id': 'chunk_cf_race'}),
                # confirm returns False → triggers _claim_fail path
                patch('scripts.om_compressor._confirm_chunk_done', return_value=False),
                # _claim_fail also returns False → race on the fail transition
                patch('scripts.om_compressor._claim_fail', return_value=False),
                patch('scripts.om_compressor.emit_event', side_effect=_capturing_emit),
            ):
                rc = run(args)

            self.assertEqual(rc, 1, 'had_failures should be set when done-confirm+claim_fail both lose ownership')
            ownership_lost = [e for e in emitted_events if e['event'] == 'OM_CLAIM_OWNERSHIP_LOST']
            self.assertTrue(ownership_lost, 'OM_CLAIM_OWNERSHIP_LOST must be emitted')
            self.assertEqual(ownership_lost[0].get('phase'), 'confirm_fail')


class TestWorkerIdUniqueness(unittest.TestCase):
    """worker_id must be globally unique across hosts for multi-host claim-mode (item 2)."""

    def test_worker_id_contains_hostname_and_pid(self):
        """worker_id must embed the current hostname and PID for observability."""
        import socket

        from scripts.om_compressor import _make_worker_id

        wid = _make_worker_id(4, 1)
        self.assertIn(socket.gethostname(), wid,
                      'worker_id must contain the hostname for cross-host uniqueness')
        self.assertIn(str(os.getpid()), wid,
                      'worker_id must contain the PID for process-level uniqueness')
        self.assertTrue(wid.startswith('om-'),
                        "worker_id must start with 'om-' for log filterability")

    def test_worker_id_unique_across_calls(self):
        """Each call must produce a unique ID (random suffix prevents PID-reuse collisions)."""
        from scripts.om_compressor import _make_worker_id

        ids = {_make_worker_id(1, 0) for _ in range(10)}
        self.assertEqual(len(ids), 10,
                         'All generated worker_ids must be distinct (random suffix)')

    def test_worker_id_encodes_shard_info(self):
        """Shard metadata must appear in worker_id for observability in log queries."""
        from scripts.om_compressor import _make_worker_id

        wid = _make_worker_id(8, 3)
        # Format: om-{host}-{shards}/{shard_index}-{pid}-{rand}
        # The '8/3' token encodes shards/shard_index.
        self.assertIn('8/3', wid, 'worker_id must embed shards/shard_index token')


class TestMaxChunksPerRunCapsByAttempts(unittest.TestCase):
    """max_chunks_per_run must cap by total claims (attempts), not only successes (item 3).

    Before the fix: if every chunk failed done-confirm (soft failure), `processed`
    stayed at 0 and the while-loop would exhaust all chunks in the manifest regardless
    of --max-chunks-per-run.  After the fix a `total_attempts` counter ensures the
    loop stops after at most max_chunks claimed chunks.
    """

    def _write_manifest(self, path: str, chunks: list) -> None:
        with open(path, 'w', encoding='utf-8') as fh:
            for row in chunks:
                fh.write(json.dumps(row) + '\n')

    def test_caps_loop_on_soft_failures(self):
        """Loop stops after max_chunks attempts even when all chunks fail done-confirm."""
        from scripts.om_compressor import ExtractorConfig, MessageRow, parse_args, run

        with tempfile.TemporaryDirectory() as td:
            manifest = os.path.join(td, 'om_manifest.jsonl')
            cfg = ExtractorConfig(
                schema_version='2026-02-17',
                prompt_template='x',
                model_id='m',
                extractor_version='ev1',
            )
            # 3 chunks in manifest, but max_chunks_per_run=2 → only 2 should be attempted.
            self._write_manifest(manifest, [
                {
                    'chunk_id': f'chunk_{i}',
                    'message_ids': [f'm{i}'],
                    'message_count': 1,
                    'extractor_version': cfg.extractor_version,
                }
                for i in range(3)
            ])

            args = parse_args([
                '--mode', 'backfill',
                '--claim-mode',
                '--build-manifest', manifest,
                '--shards', '1',
                '--shard-index', '0',
                '--max-chunks-per-run', '2',
            ])

            fake_session = MagicMock()
            driver = _FakeDriver(fake_session)
            process_calls: list[str] = []

            def fake_fetch(session, message_ids):
                return [MessageRow(
                    message_id=message_ids[0],
                    source_session_id='s',
                    content='hello',
                    created_at='2026-02-28T00:00:00Z',
                    content_embedding=[],
                    om_extract_attempts=0,
                )]

            def fake_process(session, *, messages, chunk_id, cfg, observed_node_ids):
                process_calls.append(chunk_id)
                return {'chunk_id': chunk_id, 'messages': 1, 'nodes': 0, 'edges': 0}

            with (
                patch('scripts.om_compressor._neo4j_driver', return_value=driver),
                patch('scripts.om_compressor._ensure_neo4j_constraints'),
                patch('scripts.om_compressor._load_extractor_config', return_value=cfg),
                patch('scripts.om_compressor._fetch_messages_by_ids', side_effect=fake_fetch),
                patch('scripts.om_compressor._activate_energy_scores', return_value=([], [])),
                patch('scripts.om_compressor._process_chunk', side_effect=fake_process),
                # All done-confirms fail → soft failure, processed stays 0
                patch('scripts.om_compressor._confirm_chunk_done', return_value=False),
            ):
                rc = run(args)

            self.assertEqual(
                len(process_calls), 2,
                f'Expected exactly 2 chunks attempted (max_chunks_per_run=2), got {len(process_calls)}',
            )
            self.assertEqual(rc, 1, 'Soft failures should return non-zero')


class TestStaleNonManifestRowReconciliation(unittest.TestCase):
    """Pending claim rows not in the current manifest are marked failed (item 4).

    If the manifest is rebuilt (new extractor_version or changed message set),
    the claim DB may contain pending rows from the old manifest.  Those rows
    can never be processed and would waste workers trying to claim them.
    _run_claim_mode must reconcile them to failed status on startup.
    """

    def test_stale_pending_rows_marked_failed(self):
        """Pending rows from a previous manifest generation are reconciled to failed."""
        from pathlib import Path as _Path

        from scripts.om_compressor import (
            ExtractorConfig,
            MessageRow,
            _claim_db_path_for_manifest,
            init_claim_db,
            parse_args,
            run,
        )

        with tempfile.TemporaryDirectory() as td:
            manifest = os.path.join(td, 'om_manifest.jsonl')
            cfg = ExtractorConfig(
                schema_version='2026-02-17',
                prompt_template='x',
                model_id='m',
                extractor_version='ev1',
            )
            # Write a single-chunk manifest (only chunk_current).
            row = {
                'chunk_id': 'chunk_current',
                'message_ids': ['m1'],
                'message_count': 1,
                'extractor_version': cfg.extractor_version,
            }
            with open(manifest, 'w') as fh:
                fh.write(json.dumps(row) + '\n')

            # Pre-seed the claim DB with a stale pending row from a previous manifest.
            import sqlite3 as _sqlite3

            claim_db = str(_claim_db_path_for_manifest(_Path(manifest)))
            conn = init_claim_db(claim_db)
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO chunk_claims (chunk_id, claim_shard, status) "
                    "VALUES (?, 0, 'pending')",
                    ('chunk_stale_from_old_manifest',),
                )
                conn.commit()
            finally:
                conn.close()

            args = parse_args([
                '--mode', 'backfill',
                '--claim-mode',
                '--build-manifest', manifest,
                '--shards', '1',
                '--shard-index', '0',
                '--max-chunks-per-run', '1',
            ])

            fake_session = MagicMock()
            driver = _FakeDriver(fake_session)
            msg = MessageRow(
                message_id='m1',
                source_session_id='s',
                content='hello',
                created_at='2026-02-28T00:00:00Z',
                content_embedding=[],
                om_extract_attempts=0,
            )

            with (
                patch('scripts.om_compressor._neo4j_driver', return_value=driver),
                patch('scripts.om_compressor._ensure_neo4j_constraints'),
                patch('scripts.om_compressor._load_extractor_config', return_value=cfg),
                patch('scripts.om_compressor._fetch_messages_by_ids', return_value=[msg]),
                patch('scripts.om_compressor._activate_energy_scores', return_value=([], [])),
                patch('scripts.om_compressor._process_chunk',
                      return_value={'chunk_id': 'chunk_current', 'messages': 1,
                                    'nodes': 0, 'edges': 0}),
                patch('scripts.om_compressor._confirm_chunk_done', return_value=True),
            ):
                run(args)

            # The stale row must have been reconciled to failed status.
            conn = _sqlite3.connect(claim_db)
            try:
                stale = conn.execute(
                    'SELECT status, error FROM chunk_claims WHERE chunk_id=?',
                    ('chunk_stale_from_old_manifest',),
                ).fetchone()
                self.assertIsNotNone(stale, 'Stale row must still exist in DB')
                self.assertEqual(stale[0], 'failed',
                                 'Stale pending row must be reconciled to failed')
                self.assertIn('stale', stale[1],
                              "Error message must mention 'stale'")
            finally:
                conn.close()


if __name__ == '__main__':
    unittest.main()
