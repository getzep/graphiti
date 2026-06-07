"""Unit tests for add_memory's reference_time parsing (_parse_reference_time)."""

from datetime import datetime, timezone

import pytest

from graphiti_mcp_server import _parse_reference_time


def test_aware_value_passes_through():
    result = _parse_reference_time('2026-05-14T19:00:00+00:00')
    assert result == datetime(2026, 5, 14, 19, 0, 0, tzinfo=timezone.utc)
    assert result.tzinfo is not None


def test_trailing_z_is_treated_as_utc():
    result = _parse_reference_time('2026-05-14T19:00:00Z')
    assert result == datetime(2026, 5, 14, 19, 0, 0, tzinfo=timezone.utc)


def test_naive_value_is_coerced_to_utc():
    # No offset in the input -> must not produce a naive datetime.
    result = _parse_reference_time('2026-05-14T19:00:00')
    assert result.tzinfo == timezone.utc
    assert result == datetime(2026, 5, 14, 19, 0, 0, tzinfo=timezone.utc)


def test_date_only_is_coerced_to_utc_midnight():
    result = _parse_reference_time('2026-05-14')
    assert result.tzinfo == timezone.utc
    assert result == datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)


def test_explicit_offset_is_preserved():
    result = _parse_reference_time('2026-05-14T19:00:00+05:00')
    assert result.utcoffset() is not None
    assert result.utcoffset().total_seconds() == 5 * 3600


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        _parse_reference_time('not-a-timestamp')
