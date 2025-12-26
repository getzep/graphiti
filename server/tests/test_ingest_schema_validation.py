from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from graph_service.dto import AddMessagesRequest, Message
from graph_service.routers.ingest import add_messages


@pytest.mark.asyncio
async def test_add_messages_rejects_unknown_schema_id():
    request = AddMessagesRequest(
        group_id='test',
        schema_id='unknown_schema',
        messages=[
            Message(
                content='hello',
                role_type='user',
                role='user',
                timestamp=datetime.now(timezone.utc),
                source_description='test',
            )
        ],
    )

    with pytest.raises(HTTPException) as exc:
        await add_messages(request, graphiti=Mock())

    assert exc.value.status_code == 400
