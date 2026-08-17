from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graph_service.dto import AddMessagesRequest, Message
from graph_service.routers.ingest import add_messages


@pytest.mark.asyncio
async def test_add_messages_finishes_before_returning():
    add_episode = AsyncMock()
    graphiti = SimpleNamespace(add_episode=add_episode)
    request = AddMessagesRequest(
        group_id='neo4j',
        messages=[
            Message(
                name='first',
                content='Alice works at Acme.',
                role_type='user',
                role='Alice',
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            Message(
                name='second',
                content='Bob joined Acme.',
                role_type='assistant',
                role='Bob',
                timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
        ],
    )

    result = await add_messages(request, graphiti)  # type: ignore[arg-type]

    assert result.success is True
    assert result.message == 'Messages processed'
    assert add_episode.await_count == 2
    assert add_episode.await_args_list[0].kwargs['group_id'] == 'neo4j'
    assert (
        add_episode.await_args_list[0].kwargs['episode_body'] == 'Alice(user): Alice works at Acme.'
    )
