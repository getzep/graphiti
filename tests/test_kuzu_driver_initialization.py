"""
Copyright 2026, Zep Software, Inc.

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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip('kuzu')

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.graph_queries import get_fulltext_indices
from graphiti_core.graphiti import Graphiti


@pytest.fixture
def driver():
    with (
        patch('graphiti_core.driver.kuzu_driver.kuzu.Database'),
        patch('graphiti_core.driver.kuzu_driver.kuzu.Connection'),
        patch('graphiti_core.driver.kuzu_driver.kuzu.AsyncConnection'),
        pytest.warns(DeprecationWarning, match='Kuzu backend is deprecated'),
    ):
        return KuzuDriver(db='local.kuzu')


def test_database_name_is_initialized(driver):
    assert driver._database == 'local.kuzu'


def test_explicit_group_id_resolves_without_switching_embedded_database(driver):
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.clients = MagicMock(driver=driver)
    graphiti.clients.model_copy.side_effect = lambda *, update: SimpleNamespace(**update)

    group_id, scoped_driver, scoped_clients = graphiti._resolve_request_scope('tenant_a')

    assert group_id == 'tenant_a'
    assert scoped_driver is driver
    assert scoped_clients.driver is driver
    assert driver._database == 'local.kuzu'


async def test_public_setup_creates_all_four_fulltext_indexes(driver):
    driver.execute_query = AsyncMock(return_value=([], None, None))
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver

    await graphiti.build_indices_and_constraints()

    queries = [call.args[0] for call in driver.execute_query.await_args_list]
    assert len(queries) == 4
    assert set(queries) == set(get_fulltext_indices(GraphProvider.KUZU))


async def test_setup_does_not_silently_ignore_index_creation_errors(driver):
    driver.execute_query = AsyncMock(side_effect=RuntimeError('cannot create FTS index'))

    with pytest.raises(RuntimeError, match='cannot create FTS index'):
        await driver.build_indices_and_constraints()
