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

from unittest.mock import patch

import pytest

pytest.importorskip('kuzu')

from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.graph_queries import INDEX_TO_LABEL_KUZU_MAPPING
from graphiti_core.graphiti import Graphiti


@pytest.mark.integration
async def test_fulltext_indexes_are_searchable_in_embedded_kuzu():
    import kuzu

    database = kuzu.Database(':memory:', buffer_pool_size=64 * 1024**2, max_db_size=256 * 1024**2)
    try:
        with (
            patch('graphiti_core.driver.kuzu_driver.kuzu.Database', return_value=database),
            pytest.warns(DeprecationWarning, match='Kuzu backend is deprecated'),
        ):
            driver = KuzuDriver()

        for table, properties in {
            'Entity': "name: 'memory', summary: 'projects'",
            'Community': "name: 'memory'",
            'Episodic': "content: 'memory', source: 'text', source_description: 'test'",
            'RelatesToNode_': "name: 'remembers', fact: 'memory'",
        }.items():
            await driver.execute_query(f"CREATE (:{table} {{uuid: 'sample', {properties}}})")

        graphiti = Graphiti.__new__(Graphiti)
        graphiti.driver = driver
        await graphiti.build_indices_and_constraints()

        indexes, _, _ = await driver.execute_query('CALL SHOW_INDEXES() RETURN *')
        assert {record['index_name'] for record in indexes} == set(INDEX_TO_LABEL_KUZU_MAPPING)
        for index, table in INDEX_TO_LABEL_KUZU_MAPPING.items():
            records, _, _ = await driver.execute_query(
                f"CALL QUERY_FTS_INDEX('{table}', '{index}', 'memory') RETURN node.uuid AS uuid"
            )
            assert [record['uuid'] for record in records] == ['sample']
    finally:
        database.close()
