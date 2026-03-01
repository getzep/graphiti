"""Tests for OM node timeline semantics (Phase B: first/last observed derivation).

Covers:
- _derive_node_timestamps in om_compressor: correct first/last from source messages
- om_backfill_timestamps: _compute_timestamps logic (no Neo4j required)
"""
from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
) -> "om_compressor.MessageRow":
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
) -> "om_compressor.ExtractionNode":
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
