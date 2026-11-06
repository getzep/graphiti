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

# Routing tests for the two non-abstract ops hooks that let non-Cypher backends serve
# Graphiti._get_or_create_saga and Graphiti.remove_episode without inline Cypher.
#
# Existing drivers do not implement the hooks, so the default NotImplementedError must fall
# through to the inline query unchanged, and anything that is not a real ops instance (such
# as a loosely mocked driver) must be ignored.

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import SagaNode

pytestmark = pytest.mark.asyncio

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _driver(saga_ops=None, episode_ops=None) -> MagicMock:
    driver = MagicMock(spec=GraphDriver)
    driver.execute_query = AsyncMock(return_value=([], [], None))
    type(driver).saga_node_ops = property(lambda self: saga_ops)
    type(driver).episode_node_ops = property(lambda self: episode_ops)
    return driver


def _saga_ops(**kwargs) -> MagicMock:
    ops = MagicMock(spec=SagaNodeOperations)
    ops.get_by_name = AsyncMock(**kwargs)
    return ops


def _episode_ops(**kwargs) -> MagicMock:
    ops = MagicMock(spec=EpisodeNodeOperations)
    ops.count_mentions_of_entity = AsyncMock(**kwargs)
    return ops


def _client(driver) -> SimpleNamespace:
    return SimpleNamespace(driver=driver)


async def test_loosely_mocked_driver_is_ignored_and_falls_back_to_cypher():
    # Mirrors tests that build the driver as a bare AsyncMock: its fake ops attributes are
    # not real operations instances, so the hooks must be skipped entirely.
    driver = AsyncMock()
    driver.execute_query.return_value = ([], None, None)

    with patch.object(SagaNode, 'save', new_callable=AsyncMock):
        await Graphiti._get_or_create_saga(_client(driver), 'saga', 'group', NOW)
    driver.execute_query.assert_awaited_once()

    driver.execute_query.reset_mock()
    driver.execute_query.return_value = ([{'episode_count': 1}], None, None)
    assert await Graphiti._count_entity_mentions(_client(driver), 'entity-1') == 1
    driver.execute_query.assert_awaited_once()


async def test_default_hooks_raise_not_implemented():
    executor = MagicMock()
    with pytest.raises(NotImplementedError):
        await SagaNodeOperations.get_by_name(MagicMock(), executor, 'saga', 'group')
    with pytest.raises(NotImplementedError):
        await EpisodeNodeOperations.count_mentions_of_entity(MagicMock(), executor, 'uuid')


async def test_get_or_create_saga_falls_back_to_cypher_when_hook_not_implemented():
    saga_ops = _saga_ops(side_effect=NotImplementedError)
    driver = _driver(saga_ops=saga_ops)
    driver.execute_query = AsyncMock(
        return_value=(
            [
                {
                    'uuid': 'saga-1',
                    'name': 'saga',
                    'group_id': 'group',
                    'created_at': NOW,
                    'first_episode_uuid': None,
                    'last_episode_uuid': None,
                }
            ],
            [],
            None,
        )
    )

    saga = await Graphiti._get_or_create_saga(_client(driver), 'saga', 'group', NOW)

    saga_ops.get_by_name.assert_awaited_once_with(driver, 'saga', 'group')
    driver.execute_query.assert_awaited_once()
    assert 'MATCH (s:Saga' in driver.execute_query.await_args.args[0]
    assert saga.uuid == 'saga-1'


async def test_get_or_create_saga_falls_back_when_driver_has_no_saga_ops():
    driver = _driver(saga_ops=None)
    driver.execute_query = AsyncMock(return_value=([], [], None))

    with patch.object(SagaNode, 'save', new_callable=AsyncMock) as save:
        saga = await Graphiti._get_or_create_saga(_client(driver), 'saga', 'group', NOW)

    driver.execute_query.assert_awaited_once()
    save.assert_awaited_once_with(driver)
    assert saga.name == 'saga' and saga.group_id == 'group' and saga.created_at == NOW


async def test_get_or_create_saga_returns_existing_saga_from_hook_without_cypher():
    existing = SagaNode(uuid='saga-1', name='saga', group_id='group', created_at=NOW)
    saga_ops = _saga_ops(return_value=existing)
    driver = _driver(saga_ops=saga_ops)

    saga = await Graphiti._get_or_create_saga(_client(driver), 'saga', 'group', NOW)

    assert saga is existing
    driver.execute_query.assert_not_awaited()


async def test_get_or_create_saga_creates_saga_when_hook_returns_none():
    saga_ops = _saga_ops(return_value=None)
    driver = _driver(saga_ops=saga_ops)

    with patch.object(SagaNode, 'save', new_callable=AsyncMock) as save:
        saga = await Graphiti._get_or_create_saga(_client(driver), 'saga', 'group', NOW)

    save.assert_awaited_once_with(driver)
    driver.execute_query.assert_not_awaited()
    assert saga.name == 'saga' and saga.created_at == NOW


async def test_count_entity_mentions_falls_back_to_cypher_when_hook_not_implemented():
    episode_ops = _episode_ops(side_effect=NotImplementedError)
    driver = _driver(episode_ops=episode_ops)
    driver.execute_query = AsyncMock(return_value=([{'episode_count': 2}], [], None))

    count = await Graphiti._count_entity_mentions(_client(driver), 'entity-1')

    assert count == 2
    episode_ops.count_mentions_of_entity.assert_awaited_once_with(driver, 'entity-1')
    query = driver.execute_query.await_args.args[0]
    assert 'MENTIONS' in query and 'count(*)' in query
    assert driver.execute_query.await_args.kwargs['uuid'] == 'entity-1'


async def test_count_entity_mentions_uses_hook_without_cypher():
    episode_ops = _episode_ops(return_value=1)
    driver = _driver(episode_ops=episode_ops)

    count = await Graphiti._count_entity_mentions(_client(driver), 'entity-1')

    assert count == 1
    driver.execute_query.assert_not_awaited()


async def test_count_entity_mentions_without_episode_ops_returns_zero_for_no_rows():
    driver = _driver(episode_ops=None)
    driver.execute_query = AsyncMock(return_value=([], [], None))

    assert await Graphiti._count_entity_mentions(_client(driver), 'entity-1') == 0
