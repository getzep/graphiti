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

import unittest
from unittest.mock import MagicMock

import pytest

try:
    from graphiti_core.driver.neptune_driver import NeptuneDriver

    HAS_NEPTUNE = True
except ImportError:
    NeptuneDriver = None
    HAS_NEPTUNE = False


@unittest.skipIf(not HAS_NEPTUNE, 'Neptune dependencies are not installed')
class TestNeptuneDriver:
    @staticmethod
    def _driver() -> NeptuneDriver:
        driver = NeptuneDriver.__new__(NeptuneDriver)
        driver.client = MagicMock()
        driver.aoss_client = MagicMock()
        return driver

    @pytest.mark.asyncio
    async def test_execute_query_keeps_only_cypher_referenced_params(self):
        driver = self._driver()
        driver.client.query.return_value = [{'uuid': 'node-1'}]
        query = 'MATCH (n:Entity) WHERE n.group_id IN $group_ids AND n.name = $query RETURN n'

        result = await driver.execute_query(
            query,
            params={'group_ids': ['group-1'], 'unreferenced_filter': 'unused'},
            query='Ada',
            search_vector=[0.1, 0.2],
            min_score=0.6,
            limit=10,
            database_='tenant-a',
            routing_='r',
            optional=None,
        )

        driver.client.query.assert_called_once_with(
            query,
            params={'group_ids': ['group-1'], 'query': 'Ada'},
        )
        assert result == ([{'uuid': 'node-1'}], None, None)

    @pytest.mark.asyncio
    async def test_execute_query_preserves_referenced_limit(self):
        driver = self._driver()
        driver.client.query.return_value = [{'uuid': 'node-1'}]
        query = 'MATCH (n:Entity) WHERE n.group_id = $group_id RETURN n LIMIT $limit'

        await driver.execute_query(
            query,
            group_id='group-1',
            limit=5,
            database_='tenant-a',
            optional=None,
        )

        driver.client.query.assert_called_once_with(
            query,
            params={'group_id': 'group-1', 'limit': 5},
        )

    def test_save_to_aoss_aliases_communities_to_community_name(self, monkeypatch):
        driver = self._driver()
        captured = {}

        def fake_bulk(client, actions, stats_only):
            captured['client'] = client
            captured['actions'] = list(actions)
            captured['stats_only'] = stats_only
            return 1, []

        monkeypatch.setattr('graphiti_core.driver.neptune_driver.helpers.bulk', fake_bulk)

        success = driver.save_to_aoss(
            'communities',
            [{'uuid': 'community-1', 'name': 'Graphiti users', 'group_id': 'group-1'}],
        )

        assert success == 1
        assert captured['client'] == driver.aoss_client
        assert captured['stats_only'] is True
        assert captured['actions'][0]['_index'] == 'community_name'
        assert captured['actions'][0]['uuid'] == 'community-1'
