"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from graphiti_core.graphiti import Graphiti

pytest_plugins = ('pytest_asyncio',)


@pytest.mark.asyncio
async def test_get_or_create_saga_refetch_preserves_episode_uuids():
    """Re-fetching an existing saga must round-trip first/last_episode_uuid.

    The re-fetch query used to select only uuid/name/group_id/created_at, so
    the returned node always carried first/last_episode_uuid as None and the
    caller's first-episode guard fired on every episode (#1755).
    """
    driver = AsyncMock()
    driver.execute_query.return_value = (
        [
            {
                'uuid': 'saga-1',
                'name': 'my-saga',
                'group_id': 'g1',
                'created_at': '2026-08-13T00:00:00Z',
                'first_episode_uuid': 'ep-1',
                'last_episode_uuid': 'ep-2',
            }
        ],
        None,
        None,
    )

    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver

    saga = await graphiti._get_or_create_saga('my-saga', 'g1', datetime(2026, 8, 13))

    assert saga.first_episode_uuid == 'ep-1'
    assert saga.last_episode_uuid == 'ep-2'
    # The existing-saga branch must not write anything back.
    driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_or_create_saga_creates_when_absent():
    """No matching saga row: a fresh node is minted and saved."""
    driver = AsyncMock()
    driver.execute_query.return_value = ([], None, None)

    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver

    saga = await graphiti._get_or_create_saga('new-saga', 'g1', datetime(2026, 8, 13))

    assert saga.name == 'new-saga'
    assert saga.first_episode_uuid is None
    assert saga.last_episode_uuid is None
    assert saga.save is not None
