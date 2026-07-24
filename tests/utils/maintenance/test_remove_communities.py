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

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.utils.maintenance.community_operations import remove_communities


def _mock_driver():
    driver = MagicMock()
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock()
    return driver


@pytest.mark.asyncio
async def test_remove_communities_unscoped_deletes_all():
    driver = _mock_driver()

    await remove_communities(driver)

    driver.execute_query.assert_awaited_once()
    query = driver.execute_query.await_args.args[0]
    kwargs = driver.execute_query.await_args.kwargs
    assert 'MATCH (c:Community)' in query
    assert 'WHERE' not in query
    assert 'group_ids' not in kwargs


@pytest.mark.asyncio
async def test_remove_communities_scoped_deletes_only_selected_groups():
    driver = _mock_driver()

    await remove_communities(driver, group_ids=['group_a', 'group_b'])

    driver.execute_query.assert_awaited_once()
    query = driver.execute_query.await_args.args[0]
    kwargs = driver.execute_query.await_args.kwargs
    assert 'WHERE c.group_id IN $group_ids' in query
    assert kwargs['group_ids'] == ['group_a', 'group_b']


@pytest.mark.asyncio
async def test_remove_communities_empty_group_ids_deletes_all():
    # group_ids=[] means "no scoping requested" (matches build_communities'
    # None/blank semantics), not "delete nothing".
    driver = _mock_driver()

    await remove_communities(driver, group_ids=[])

    driver.execute_query.assert_awaited_once()
    query = driver.execute_query.await_args.args[0]
    assert 'WHERE' not in query


@pytest.mark.asyncio
async def test_remove_communities_scoped_via_graph_operations_interface():
    driver = MagicMock()
    driver.graph_operations_interface = MagicMock()
    driver.graph_operations_interface.remove_communities = AsyncMock()
    driver.execute_query = AsyncMock()

    await remove_communities(driver, group_ids=['group_a'])

    driver.graph_operations_interface.remove_communities.assert_awaited_once_with(
        driver, group_ids=['group_a']
    )
    driver.execute_query.assert_not_awaited()
