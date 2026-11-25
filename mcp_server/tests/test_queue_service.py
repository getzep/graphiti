"""The episode queue must process one group's episodes one at a time, in order.

These tests need no database and no model. A fake client records when each
add_episode call starts and ends.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services.queue_service import QueueService  # noqa: E402


class RecordingClient:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.events: list[tuple[str, str]] = []

    async def add_episode(self, name: str, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.events.append(('start', name))
        # Yield, as the LLM and database calls of a real add_episode do.
        await asyncio.sleep(0.01)
        self.events.append(('end', name))
        self.active -= 1


async def _add(service: QueueService, group_id: str, name: str) -> None:
    await service.add_episode(
        group_id=group_id,
        name=name,
        content='body',
        source_description='desc',
        episode_type='text',
        entity_types=None,
        uuid=None,
    )


def _worker_tasks() -> list[asyncio.Task]:
    return [
        task
        for task in asyncio.all_tasks()
        if '_process_episode_queue' in repr(task.get_coro()) and not task.done()
    ]


@pytest.mark.asyncio
async def test_concurrent_calls_for_one_group_start_one_worker():
    """Calls that arrive together, as parallel MCP tool calls do, share one worker."""
    client = RecordingClient()
    service = QueueService()
    await service.initialize(client)

    await asyncio.gather(*(_add(service, 'g', f'ep{i}') for i in range(3)))
    await service._episode_queues['g'].join()

    workers = _worker_tasks()
    try:
        assert len(workers) == 1
        assert client.max_active == 1
        assert client.events == [
            ('start', 'ep0'),
            ('end', 'ep0'),
            ('start', 'ep1'),
            ('end', 'ep1'),
            ('start', 'ep2'),
            ('end', 'ep2'),
        ]
        assert service.is_worker_running('g')
    finally:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


@pytest.mark.asyncio
async def test_each_group_gets_its_own_worker():
    client = RecordingClient()
    service = QueueService()
    await service.initialize(client)

    await asyncio.gather(_add(service, 'a', 'a0'), _add(service, 'b', 'b0'))
    await asyncio.gather(service._episode_queues['a'].join(), service._episode_queues['b'].join())

    workers = _worker_tasks()
    try:
        assert len(workers) == 2
        assert service.is_worker_running('a')
        assert service.is_worker_running('b')
    finally:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    assert not service.is_worker_running('a')
    assert not service.is_worker_running('b')
