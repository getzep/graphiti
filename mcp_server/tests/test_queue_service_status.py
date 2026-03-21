import asyncio

import pytest

from services.queue_service import QueueService


class _BlockingGraphitiClient:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def add_episode(self, **kwargs):
        self.started.set()
        await self.release.wait()


class _FailingGraphitiClient:
    def __init__(self):
        self.started = asyncio.Event()

    async def add_episode(self, **kwargs):
        self.started.set()
        raise RuntimeError('boom')


async def _wait_for(predicate, timeout: float = 1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.01)
    raise AssertionError('condition not met before timeout')


@pytest.mark.asyncio
async def test_queue_service_tracks_successful_episode_lifecycle():
    queue_service = QueueService()
    graphiti_client = _BlockingGraphitiClient()
    await queue_service.initialize(graphiti_client)

    queue_position = await queue_service.add_episode(
        group_id='group-1',
        name='Episode',
        content='content',
        source_description='test',
        episode_type='text',
        entity_types=None,
        uuid='episode-1',
    )

    assert queue_position == 1
    queued_status = queue_service.get_episode_status('episode-1')
    assert queued_status is not None
    assert queued_status.state == 'queued'

    await asyncio.wait_for(graphiti_client.started.wait(), timeout=1.0)

    processing_status = await _wait_for(
        lambda: queue_service.get_episode_status('episode-1')
        if queue_service.get_episode_status('episode-1').state == 'processing'
        else None
    )
    assert processing_status.started_at is not None
    assert processing_status.processed_at is None

    graphiti_client.release.set()
    completed_status = await _wait_for(
        lambda: queue_service.get_episode_status('episode-1')
        if queue_service.get_episode_status('episode-1').state == 'completed'
        else None
    )
    assert completed_status.processed_at is not None
    assert completed_status.last_error is None


@pytest.mark.asyncio
async def test_queue_service_tracks_failed_episode_lifecycle():
    queue_service = QueueService()
    graphiti_client = _FailingGraphitiClient()
    await queue_service.initialize(graphiti_client)

    await queue_service.add_episode(
        group_id='group-1',
        name='Episode',
        content='content',
        source_description='test',
        episode_type='text',
        entity_types=None,
        uuid='episode-2',
    )

    await asyncio.wait_for(graphiti_client.started.wait(), timeout=1.0)
    failed_status = await _wait_for(
        lambda: queue_service.get_episode_status('episode-2')
        if queue_service.get_episode_status('episode-2').state == 'failed'
        else None
    )
    assert failed_status.last_error == 'boom'
    assert failed_status.processed_at is not None


@pytest.mark.asyncio
async def test_queue_service_reports_pending_queue_position():
    queue_service = QueueService()
    graphiti_client = _BlockingGraphitiClient()
    await queue_service.initialize(graphiti_client)

    await queue_service.add_episode(
        group_id='group-1',
        name='Episode 1',
        content='content',
        source_description='test',
        episode_type='text',
        entity_types=None,
        uuid='episode-3',
    )
    await asyncio.wait_for(graphiti_client.started.wait(), timeout=1.0)

    queue_position = await queue_service.add_episode(
        group_id='group-1',
        name='Episode 2',
        content='content',
        source_description='test',
        episode_type='text',
        entity_types=None,
        uuid='episode-4',
    )

    assert queue_position == 1
    pending_status = queue_service.get_episode_status('episode-4')
    assert pending_status is not None
    assert pending_status.state == 'queued'
    assert queue_service.get_queue_position('episode-4') == 1
