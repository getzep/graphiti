import asyncio
from unittest.mock import AsyncMock

import pytest

from services.queue_service import QueueService


@pytest.mark.unit
def test_add_episode_forwards_custom_extraction_instructions():
    async def run_test():
        service = QueueService()
        mock_client = AsyncMock()
        await service.initialize(mock_client)

        async def run_immediately(group_id, process_func):
            await process_func()
            return 1

        service.add_episode_task = run_immediately

        await service.add_episode(
            group_id="group-a",
            name="Episode Name",
            content="Episode body",
            source_description="desc",
            episode_type="text",
            entity_types=[],
            uuid="ep-1",
            custom_extraction_instructions="Preserve original language.",
        )

        mock_client.add_episode.assert_awaited_once_with(
            name="Episode Name",
            episode_body="Episode body",
            source_description="desc",
            source="text",
            group_id="group-a",
            reference_time=mock_client.add_episode.await_args.kwargs["reference_time"],
            entity_types=[],
            uuid="ep-1",
            custom_extraction_instructions="Preserve original language.",
        )

    asyncio.run(run_test())
