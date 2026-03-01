"""Tests for OM node timeline semantics (Phase B: first/last observed derivation).

Covers:
- _derive_node_timestamps in om_compressor: correct first/last from source messages
- om_backfill_timestamps: _compute_timestamps logic (no Neo4j required)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect

om_compressor = importlib.import_module("scripts.om_compressor")
om_backfill = importlib.import_module("scripts.om_backfill_timestamps")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_message_row(
    message_id: str,
    created_at: str,
    content: str = "test content",
    source_session_id: str = "session1",
) -> om_compressor.MessageRow:
    return om_compressor.MessageRow(
        message_id=message_id,
        source_session_id=source_session_id,
        content=content,
        created_at=created_at,
        content_embedding=[],
        om_extract_attempts=0,
    )


def _make_extraction_node(
    node_id: str = "node1",
    source_message_ids: list[str] | None = None,
) -> om_compressor.ExtractionNode:
    return om_compressor.ExtractionNode(
        node_id=node_id,
        node_type="Judgment",
        semantic_domain="sessions_main",
        content="test content",
        urgency_score=3,
        source_session_id="session1",
        source_message_ids=source_message_ids or [],
    )


# ---------------------------------------------------------------------------
# _derive_node_timestamps (om_compressor)
# ---------------------------------------------------------------------------

class TestDeriveNodeTimestamps:
    def test_single_message_returns_same_first_and_last(self):
        msg = _make_message_row("m1", "2026-01-10T08:00:00Z")
        node = _make_extraction_node(source_message_ids=["m1"])
        msg_by_id = {"m1": msg}

        first, last = om_compressor._derive_node_timestamps(node, msg_by_id)

        assert first == "2026-01-10T08:00:00Z"
        assert last == "2026-01-10T08:00:00Z"

    def test_multiple_messages_first_is_min_last_is_max(self):
        messages = {
            "m1": _make_message_row("m1", "2026-01-05T00:00:00Z"),
            "m2": _make_message_row("m2", "2026-01-15T00:00:00Z"),
            "m3": _make_message_row("m3", "2026-01-10T00:00:00Z"),
        }
        node = _make_extraction_node(source_message_ids=["m1", "m2", "m3"])

        first, last = om_compressor._derive_node_timestamps(node, messages)

        assert first == "2026-01-05T00:00:00Z"
        assert last == "2026-01-15T00:00:00Z"

    def test_returns_none_when_no_source_ids(self):
        node = _make_extraction_node(source_message_ids=[])
        first, last = om_compressor._derive_node_timestamps(node, {})
        assert first is None
        assert last is None

    def test_returns_none_when_messages_not_in_lookup(self):
        node = _make_extraction_node(source_message_ids=["missing_id"])
        first, last = om_compressor._derive_node_timestamps(node, {})
        assert first is None
        assert last is None

    def test_partial_lookup_uses_available_messages(self):
        """If only some source_message_ids resolve, use those for timestamps."""
        messages = {
            "m1": _make_message_row("m1", "2026-02-01T00:00:00Z"),
        }
        node = _make_extraction_node(source_message_ids=["m1", "missing"])

        first, last = om_compressor._derive_node_timestamps(node, messages)

        assert first == "2026-02-01T00:00:00Z"
        assert last == "2026-02-01T00:00:00Z"

    def test_microseconds_stripped_from_output(self):
        msg = _make_message_row("m1", "2026-01-10T08:30:45.123456Z")
        node = _make_extraction_node(source_message_ids=["m1"])

        first, last = om_compressor._derive_node_timestamps(node, {"m1": msg})

        # Output should be rounded to seconds (no microseconds)
        assert "." not in (first or "")
        assert "." not in (last or "")

    def test_returns_none_when_message_has_no_created_at(self):
        msg = _make_message_row("m1", "")  # empty created_at
        node = _make_extraction_node(source_message_ids=["m1"])

        first, last = om_compressor._derive_node_timestamps(node, {"m1": msg})

        assert first is None
        assert last is None

    def test_output_format_is_utc_iso_z(self):
        msg = _make_message_row("m1", "2026-03-15T12:00:00Z")
        node = _make_extraction_node(source_message_ids=["m1"])

        first, last = om_compressor._derive_node_timestamps(node, {"m1": msg})

        assert first is not None and first.endswith("Z")
        assert last is not None and last.endswith("Z")


# ---------------------------------------------------------------------------
# om_backfill_timestamps._compute_timestamps
# ---------------------------------------------------------------------------

class TestBackfillComputeTimestamps:
    def _make_row(
        self,
        node_id: str = "n1",
        node_created_at: str | None = "2026-01-01T10:00:00Z",
        message_timestamps: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "node_id": node_id,
            "node_created_at": node_created_at,
            "message_timestamps": message_timestamps or [],
        }

    def test_message_timestamps_produce_correct_first_last(self):
        row = self._make_row(message_timestamps=[
            "2026-01-05T00:00:00Z",
            "2026-01-20T00:00:00Z",
            "2026-01-10T00:00:00Z",
        ])
        first, last, source = om_backfill._compute_timestamps(row)

        assert first == "2026-01-05T00:00:00Z"
        assert last == "2026-01-20T00:00:00Z"
        assert source == "messages"

    def test_single_message_timestamp(self):
        row = self._make_row(message_timestamps=["2026-03-01T00:00:00Z"])
        first, last, source = om_backfill._compute_timestamps(row)

        assert first == "2026-03-01T00:00:00Z"
        assert last == "2026-03-01T00:00:00Z"
        assert source == "messages"

    def test_no_messages_falls_back_to_node_created_at(self):
        row = self._make_row(node_created_at="2026-01-15T08:00:00Z", message_timestamps=[])
        first, last, source = om_backfill._compute_timestamps(row)

        assert first == "2026-01-15T08:00:00Z"
        assert last == "2026-01-15T08:00:00Z"
        assert source == "fallback_created_at"

    def test_no_messages_and_no_created_at_returns_none(self):
        row = self._make_row(node_created_at=None, message_timestamps=[])
        first, last, source = om_backfill._compute_timestamps(row)

        assert first is None
        assert last is None
        assert source == "no_timestamp"

    def test_null_timestamps_in_message_list_are_skipped(self):
        row = self._make_row(
            node_created_at="2026-01-01T00:00:00Z",
            message_timestamps=[None, "2026-02-01T00:00:00Z", None],  # type: ignore[arg-type]
        )
        first, last, source = om_backfill._compute_timestamps(row)

        assert first == "2026-02-01T00:00:00Z"
        assert source == "messages"

    def test_microseconds_stripped(self):
        row = self._make_row(message_timestamps=["2026-01-10T09:30:45.999999Z"])
        first, last, source = om_backfill._compute_timestamps(row)

        assert "." not in (first or "")
        assert source == "messages"

    def test_output_ends_with_z(self):
        row = self._make_row(message_timestamps=["2026-06-01T12:00:00Z"])
        first, last, source = om_backfill._compute_timestamps(row)
        assert first is not None and first.endswith("Z")
        assert last is not None and last.endswith("Z")


# ---------------------------------------------------------------------------
# Backfill CLI args
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Compressor last_observed_at — no wall-clock overwrite
# ---------------------------------------------------------------------------

class TestCompressorObservedNodeUpdate:
    """Regression: _process_chunk_tx must NOT overwrite last_observed_at with
    wall-clock time when updating 'observed' (semantically similar) nodes.
    It should use max(message.created_at) semantics instead.

    Inspects the Cypher emitted in the function source to verify the contract.
    """

    def test_observed_node_update_does_not_use_now_iso(self):
        """The observed-node SET block must not unconditionally set last_observed_at = $now_iso."""
        src = inspect.getsource(om_compressor._process_chunk_tx)
        # Confirm the wall-clock approach is gone
        assert "SET n.last_observed_at = $now_iso" not in src, (
            "_process_chunk_tx still overwrites last_observed_at with wall-clock "
            "time ($now_iso) for observed nodes.  Use message-timestamp max instead."
        )

    def test_observed_node_update_uses_msg_max_ts(self):
        """The observed-node SET block must reference msg_max_ts."""
        src = inspect.getsource(om_compressor._process_chunk_tx)
        assert "msg_max_ts" in src, (
            "_process_chunk_tx does not use msg_max_ts for observed-node "
            "last_observed_at updates."
        )

    def test_observed_node_update_uses_max_semantics(self):
        """The Cypher must use CASE … WHEN $msg_max_ts > n.last_observed_at logic."""
        src = inspect.getsource(om_compressor._process_chunk_tx)
        assert "$msg_max_ts > n.last_observed_at" in src, (
            "Observed-node last_observed_at update is missing max-semantics guard "
            "('$msg_max_ts > n.last_observed_at').  Wall-clock time may silently "
            "overwrite a later event-time timestamp."
        )


class TestBackfillParseArgs:
    def test_default_is_dry_run(self):
        args = om_backfill.parse_args([])
        assert args.dry_run is True
        assert args.apply is False

    def test_apply_flag(self):
        args = om_backfill.parse_args(["--apply"])
        assert args.apply is True

    def test_dry_run_and_apply_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            om_backfill.parse_args(["--dry-run", "--apply"])


# ---------------------------------------------------------------------------
# Cursor-based pagination (_fetch_batch_cursor)
# ---------------------------------------------------------------------------

class TestFetchBatchCursor:
    """Unit tests for cursor-based pagination in _fetch_batch_cursor.

    These tests confirm the pagination contract without a live Neo4j session
    by exercising the public function signature and the run() loop logic
    through a mock session.
    """

    def _make_row(self, node_id: str, msg_ts: str | None = "2026-01-01T00:00:00Z") -> dict:
        return {
            "node_id": node_id,
            "node_created_at": "2026-01-01T00:00:00Z",
            "message_timestamps": [msg_ts] if msg_ts else [],
        }

    def test_run_terminates_when_all_nodes_skipped_apply_mode(self):
        """Regression: run() must not loop infinitely when every node in a
        batch is skipped (no timestamp).  With the old SKIP-based approach,
        the cursor never advanced and the loop ran forever.
        """
        from unittest.mock import MagicMock, patch

        skippable_rows = [
            {"node_id": "n1", "node_created_at": None, "message_timestamps": []},
            {"node_id": "n2", "node_created_at": None, "message_timestamps": []},
        ]

        mock_session = MagicMock()
        # _count_unbackfilled → 2 nodes
        mock_session.run.return_value.single.return_value = {"total": 2}

        call_count = {"n": 0}

        def fake_fetch_batch_cursor(session, after_id):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return skippable_rows  # first call: two unskippable rows
            return []  # second call: nothing left (cursor advanced past them)

        with (
            patch.object(om_backfill, "_fetch_batch_cursor", side_effect=fake_fetch_batch_cursor),
            patch.object(om_backfill, "_neo4j_driver") as mock_driver,
        ):
            mock_driver.return_value.__enter__ = lambda s: s
            mock_driver.return_value.__exit__ = MagicMock(return_value=False)
            mock_driver.return_value.session.return_value.__enter__ = lambda s: mock_session
            mock_driver.return_value.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.run.return_value.single.return_value = {"total": 2}

            args = om_backfill.parse_args(["--apply"])
            result = om_backfill.run(args)

        # Must exit cleanly (return 0) and must NOT have looped more than twice
        assert result == 0
        assert call_count["n"] == 2, (
            f"Expected exactly 2 fetch calls (one real batch + one empty sentinel), "
            f"got {call_count['n']} — possible infinite loop"
        )

    def test_cursor_advances_past_last_seen_node_id(self):
        """Cursor is set to the node_id of the last row in every batch."""
        from unittest.mock import MagicMock, patch

        batch_1 = [
            {"node_id": "aaa", "node_created_at": "2026-01-01T00:00:00Z", "message_timestamps": ["2026-01-01T00:00:00Z"]},
            {"node_id": "bbb", "node_created_at": "2026-01-02T00:00:00Z", "message_timestamps": ["2026-01-02T00:00:00Z"]},
        ]

        cursors_seen: list[str] = []

        def fake_fetch(session, after_id):
            cursors_seen.append(after_id)
            if after_id == "":
                return batch_1
            return []

        with (
            patch.object(om_backfill, "_fetch_batch_cursor", side_effect=fake_fetch),
            patch.object(om_backfill, "_neo4j_driver") as mock_driver,
        ):
            mock_session = MagicMock()
            mock_driver.return_value.__enter__ = lambda s: s
            mock_driver.return_value.__exit__ = MagicMock(return_value=False)
            mock_driver.return_value.session.return_value.__enter__ = lambda s: mock_session
            mock_driver.return_value.session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.run.return_value.single.return_value = {"total": 2}

            args = om_backfill.parse_args(["--dry-run"])
            om_backfill.run(args)

        # First call: cursor starts at ""
        assert cursors_seen[0] == ""
        # Second call: cursor must have advanced to the last node_id in batch_1
        assert cursors_seen[1] == "bbb"
