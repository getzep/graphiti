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

import logging
from unittest.mock import AsyncMock, MagicMock

from graphiti_core.driver.falkordb_driver import FalkorDriver


def make_driver(graph_map: dict[str, int], database: str = 'GRAPHITI') -> FalkorDriver:
    client = MagicMock()
    client.list_graphs = AsyncMock(return_value=list(graph_map))

    def select_graph(name: str) -> MagicMock:
        graph = MagicMock()
        result = MagicMock()
        result.result_set = [[graph_map[name]]]
        graph.query = AsyncMock(return_value=result)
        return graph

    client.select_graph.side_effect = select_graph

    driver = FalkorDriver.__new__(FalkorDriver)  # no connection needed
    driver.client = client
    driver._database = database
    return driver


async def test_warns_on_legacy_episodic_graphs(caplog):
    """Graphs other than the unified one that contain Episodic data are reported."""
    driver = make_driver({'GRAPHITI': 0, 'other-app': 0, 'old-group': 42})

    with caplog.at_level(logging.WARNING, logger='graphiti_core.driver.falkordb_driver'):
        await driver._warn_on_legacy_graph_layout()

    assert 'old-group' in caplog.text
    assert 'migrate_falkordb_graphs' in caplog.text


async def test_no_warning_without_legacy_data(caplog):
    """No graphiti data outside the unified graph means no warning."""
    driver = make_driver({'GRAPHITI': 0, 'other-app': 0})

    with caplog.at_level(logging.WARNING, logger='graphiti_core.driver.falkordb_driver'):
        await driver._warn_on_legacy_graph_layout()

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


async def test_detection_failure_is_silent(caplog):
    """A broken list_graphs call must not break startup."""
    driver = make_driver({})
    driver.client.list_graphs.side_effect = RuntimeError('boom')

    with caplog.at_level(logging.WARNING, logger='graphiti_core.driver.falkordb_driver'):
        await driver._warn_on_legacy_graph_layout()

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
