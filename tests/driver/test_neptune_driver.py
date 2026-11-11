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
        driver._database = ''
        return driver

    @pytest.mark.asyncio
    async def test_execute_query_flattens_params_and_drops_routing(self):
        driver = self._driver()
        driver.client.query.return_value = [{'uuid': 'edge-1'}]

        result = await driver.execute_query(
            'MATCH (e:Entity) WHERE e.uuid = $uuid RETURN e',
            params={'uuid': 'edge-1'},
            routing_='r',
        )

        driver.client.query.assert_called_once_with(
            'MATCH (e:Entity) WHERE e.uuid = $uuid RETURN e',
            params={'uuid': 'edge-1'},
        )
        assert result == ([{'uuid': 'edge-1'}], None, None)

    def test_save_to_aoss_indexes_uuid_without_custom_id(self, monkeypatch):
        driver = self._driver()
        captured = {}

        def fake_bulk(client, actions, stats_only):
            captured['client'] = client
            captured['actions'] = actions
            captured['stats_only'] = stats_only
            return 1, []

        monkeypatch.setattr('graphiti_core.driver.neptune_driver.helpers.bulk', fake_bulk)

        success = driver.save_to_aoss(
            'edge_name_and_fact',
            [
                {
                    'uuid': 'edge-1',
                    'name': 'knows',
                    'fact': 'Alice knows Bob',
                    'group_id': 'tenant-a',
                    'ignored': 'not indexed',
                }
            ],
        )

        assert success == 1
        assert captured['client'] is driver.aoss_client
        assert captured['stats_only'] is True
        assert captured['actions'] == [
            {
                '_index': 'edge_name_and_fact',
                'uuid': 'edge-1',
                'name': 'knows',
                'fact': 'Alice knows Bob',
                'group_id': 'tenant-a',
            }
        ]
        assert '_id' not in captured['actions'][0]

    def test_clone_reuses_clients_with_new_database(self):
        driver = self._driver()
        driver._database = 'tenant-a'

        clone = driver.clone(database='tenant-b')

        assert clone is not driver
        assert clone.client is driver.client
        assert clone.aoss_client is driver.aoss_client
        assert clone._database == 'tenant-b'
        assert driver._database == 'tenant-a'
