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

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.neo4j_driver import Neo4jDriver


@pytest.mark.asyncio
async def test_close_cancels_pending_indices_task():
    started = asyncio.Event()
    mock_client = MagicMock()
    mock_client.close = AsyncMock()

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        started.set()
        await asyncio.Event().wait()

    with (
        patch(
            'graphiti_core.driver.neo4j_driver.AsyncGraphDatabase.driver', return_value=mock_client
        ),
        patch.object(
            Neo4jDriver,
            'build_indices_and_constraints',
            build_indices_and_constraints,
        ),
    ):
        driver = Neo4jDriver('bolt://localhost:7687', 'neo4j', 'password')
        assert driver._indices_task is not None
        await asyncio.wait_for(started.wait(), timeout=1)

        task = driver._indices_task
        await driver.close()

    assert task.cancelled()
    mock_client.close.assert_awaited_once()
