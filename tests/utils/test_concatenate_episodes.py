"""Tests for graphiti_core.utils.text_utils."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from graphiti_core.utils.text_utils import concatenate_episodes


def _make_episode(content: str, valid_at: datetime | None = None) -> MagicMock:
    ep = MagicMock()
    ep.content = content
    ep.valid_at = valid_at or datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return ep


class TestConcatenateEpisodes:
    def test_single_episode_returns_content_as_is(self):
        ep = _make_episode('Hello world')
        assert concatenate_episodes([ep]) == 'Hello world'

    def test_multiple_episodes_adds_headers_with_timestamps(self):
        eps = [
            _make_episode('First', datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)),
            _make_episode('Second', datetime(2025, 1, 1, 10, 5, 0, tzinfo=timezone.utc)),
            _make_episode('Third', datetime(2025, 1, 1, 10, 10, 0, tzinfo=timezone.utc)),
        ]
        result = concatenate_episodes(eps)
        assert '[Episode 0] (timestamp: 2025-01-01T10:00:00+00:00)\nFirst' in result
        assert '[Episode 1] (timestamp: 2025-01-01T10:05:00+00:00)\nSecond' in result
        assert '[Episode 2] (timestamp: 2025-01-01T10:10:00+00:00)\nThird' in result

    def test_multiple_episodes_separated_by_blank_line(self):
        ts = datetime(2025, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
        eps = [_make_episode('A', ts), _make_episode('B', ts)]
        result = concatenate_episodes(eps)
        ts_str = ts.isoformat()
        assert result == (
            f'[Episode 0] (timestamp: {ts_str})\nA\n\n[Episode 1] (timestamp: {ts_str})\nB'
        )

    def test_single_episode_no_header(self):
        ep = _make_episode('No header please')
        result = concatenate_episodes([ep])
        assert '[Episode' not in result

    def test_multiple_episodes_with_none_valid_at(self):
        ts = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        eps = [_make_episode('A', ts), _make_episode('B')]
        eps[1].valid_at = None
        result = concatenate_episodes(eps)
        assert f'(timestamp: {ts.isoformat()})' in result
        assert '(timestamp: unknown)' in result
