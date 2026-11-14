# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.driver.oracle.config import Oracle26aiConfig
from graphiti_core.driver.oracle.driver import Oracle26aiDriver
from graphiti_core.driver.oracle.legacy import (
    OracleLegacyGraphOperations,
    OracleLegacySearchInterface,
)
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from tests.driver.oracle.unit.fakes import FakePool


def public_async_methods(type_: type) -> set[str]:
    return {
        name
        for name, value in type_.__dict__.items()
        if not name.startswith('_') and inspect.iscoroutinefunction(value)
    }


def test_driver_exposes_every_non_search_operation_family() -> None:
    driver = Oracle26aiDriver(
        Oracle26aiConfig(dsn='myatp_low', user='app', password='secret'),
        pool=FakePool(),
    )
    names = (
        'entity_node_ops',
        'episode_node_ops',
        'community_node_ops',
        'saga_node_ops',
        'entity_edge_ops',
        'episodic_edge_ops',
        'community_edge_ops',
        'has_episode_edge_ops',
        'next_episode_edge_ops',
        'graph_ops',
        'search_ops',
    )

    assert all(getattr(driver, name) is not None for name in names)


def test_legacy_graph_interface_methods_are_overridden() -> None:
    base_methods = public_async_methods(GraphOperationsInterface)
    adapter_methods = public_async_methods(OracleLegacyGraphOperations)

    assert base_methods <= adapter_methods


def test_legacy_search_interface_methods_are_overridden() -> None:
    base_methods = public_async_methods(SearchInterface)
    adapter_methods = public_async_methods(OracleLegacySearchInterface)

    assert base_methods <= adapter_methods
