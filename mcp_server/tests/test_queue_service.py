"""Unit tests for QueueService reference_time plumbing."""

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from services.queue_service import QueueService


class _FakeGraphitiClient:
    """Captures the kwargs passed to add_episode for assertion."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.done = asyncio.Event()

    async def add_episode(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        self.done.set()


async def _enqueue_and_wait(
    fake_client: _FakeGraphitiClient,
    **add_episode_kwargs: Any,
) -> None:
    service = QueueService()
    await service.initialize(fake_client)
    await service.add_episode(
        group_id='test-group',
        name='ep',
        content='body',
        source_description='src',
        episode_type='text',
        entity_types=None,
        uuid=None,
        **add_episode_kwargs,
    )
    # Wait for the background worker to drain the queue.
    await asyncio.wait_for(fake_client.done.wait(), timeout=5)


@pytest.mark.asyncio
async def test_explicit_reference_time_is_passed_through() -> None:
    fake = _FakeGraphitiClient()
    explicit = datetime(2026, 5, 14, 19, 0, 0, tzinfo=timezone.utc)

    await _enqueue_and_wait(fake, reference_time=explicit)

    assert len(fake.calls) == 1
    assert fake.calls[0]['reference_time'] == explicit


@pytest.mark.asyncio
async def test_missing_reference_time_defaults_to_now() -> None:
    fake = _FakeGraphitiClient()
    before = datetime.now(timezone.utc)

    await _enqueue_and_wait(fake)

    after = datetime.now(timezone.utc)
    assert len(fake.calls) == 1
    used = fake.calls[0]['reference_time']
    assert isinstance(used, datetime)
    assert used.tzinfo is not None
    assert before <= used <= after
