from datetime import datetime, timezone

import pytest

from graphiti_core.utils.maintenance.temporal_edge_utils import normalize_future_temporal_bounds


def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ('fact', 'valid_at', 'invalid_at'),
    [
        (
            'Emma will relocate to Tokyo on December 25, 2022',
            _dt(2022, 12, 25),
            None,
        ),
        (
            'David will launch the TechNova mobile app on September 1, 2026',
            _dt(2026, 9, 1),
            None,
        ),
        (
            'The team will ship version 2.0 by the end of Q3 2025',
            _dt(2025, 9, 30),
            None,
        ),
        (
            'Maria will pay back the loan by March 2026',
            None,
            None,
        ),
    ],
)
def test_future_assertions_are_valid_at_reference_time(
    fact: str,
    valid_at: datetime | None,
    invalid_at: datetime | None,
):
    reference_time = _dt(2020, 1, 1)

    normalized_valid_at, normalized_invalid_at = normalize_future_temporal_bounds(
        fact,
        reference_time,
        valid_at,
        invalid_at,
    )

    assert normalized_valid_at == reference_time
    assert normalized_invalid_at == invalid_at


def test_future_soft_deadline_is_not_invalid_at():
    reference_time = _dt(2020, 1, 1)

    normalized_valid_at, normalized_invalid_at = normalize_future_temporal_bounds(
        'Alice will complete the project by July 30, 2024',
        reference_time,
        _dt(2024, 7, 30),
        _dt(2024, 7, 30),
    )

    assert normalized_valid_at == reference_time
    assert normalized_invalid_at is None


def test_future_ongoing_state_end_keeps_invalid_at():
    reference_time = _dt(2020, 1, 1)
    invalid_at = _dt(2024, 12, 25)

    normalized_valid_at, normalized_invalid_at = normalize_future_temporal_bounds(
        'TinyBirds styles will be unavailable until December 25, 2024',
        reference_time,
        None,
        invalid_at,
    )

    assert normalized_valid_at == reference_time
    assert normalized_invalid_at == invalid_at


def test_completed_event_date_is_unchanged():
    valid_at = _dt(2024, 9, 1)

    normalized_valid_at, normalized_invalid_at = normalize_future_temporal_bounds(
        'David launched the TechNova mobile app on September 1, 2024',
        _dt(2020, 1, 1),
        valid_at,
        None,
    )

    assert normalized_valid_at == valid_at
    assert normalized_invalid_at is None
