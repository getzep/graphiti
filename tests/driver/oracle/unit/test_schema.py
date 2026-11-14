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

import pytest

from graphiti_core.driver.oracle.schema import SchemaInspector


class FakeCatalogExecutor:
    def __init__(self, *, vector_indexes: int = 2) -> None:
        self.vector_indexes = vector_indexes

    async def execute_query(self, statement: str, **_binds: object):
        if 'graphiti_schema_migrations' in statement.lower():
            records = [{'applied_count': 4, 'applying_count': 0}]
        elif 'user_tables' in statement.lower():
            records = [{'object_count': 2}]
        elif 'user_property_graphs' in statement.lower():
            records = [{'object_count': 1}]
        elif 'GRAPHITI_NODES_TEXT_IDX' in statement:
            records = [{'object_count': 2}]
        elif 'GRAPHITI_NODES_VECTOR_IDX' in statement:
            records = [{'object_count': self.vector_indexes}]
        else:
            raise AssertionError(f'unexpected catalog query: {statement}')
        return records, list(records[0]), None


@pytest.mark.asyncio
async def test_schema_readiness_requires_every_component() -> None:
    readiness = await SchemaInspector(FakeCatalogExecutor()).readiness()

    assert readiness.ready is True
    assert readiness.model_dump() == {
        'migrations_applied': 4,
        'migration_in_progress': False,
        'tables_ready': True,
        'property_graph_ready': True,
        'text_indexes_ready': True,
        'vector_indexes_ready': True,
        'ready': True,
    }


@pytest.mark.asyncio
async def test_schema_readiness_fails_when_a_vector_index_is_missing() -> None:
    readiness = await SchemaInspector(FakeCatalogExecutor(vector_indexes=1)).readiness()

    assert readiness.vector_indexes_ready is False
    assert readiness.ready is False
