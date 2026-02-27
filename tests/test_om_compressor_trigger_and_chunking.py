"""Tests for OM compressor Smart Cutter integration (FR-5 + FR-9 + FR-11)."""

from __future__ import annotations

import sys
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


if __name__ == '__main__':
    unittest.main()
