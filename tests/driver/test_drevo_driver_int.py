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

import os

import pytest

try:
    from graphiti_core.driver.drevo_driver import DrevoDriver
    from graphiti_core.search.search_filters import SearchFilters

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

DREVO_URI = os.getenv('DREVO_URI', 'bolt://localhost:7687')
DREVO_USER = os.getenv('DREVO_USER', 'neo4j')
DREVO_PASSWORD = os.getenv('DREVO_PASSWORD', 'password')
GROUP = 'drevo_int_test'


async def _connect_or_skip() -> 'DrevoDriver':
    driver = DrevoDriver(uri=DREVO_URI, user=DREVO_USER, password=DREVO_PASSWORD)
    try:
        await driver.execute_query('RETURN 1 AS ok')
    except Exception as e:  # pragma: no cover - depends on a live drevo
        await driver.close()
        pytest.skip(f'drevo not reachable at {DREVO_URI}: {e}')
    return driver


async def _seed(driver: 'DrevoDriver') -> None:
    await driver.execute_query(f'MATCH (n:Entity {{group_id: "{GROUP}"}}) DETACH DELETE n')
    # Two entities with orthogonal embeddings so cosine ranking is unambiguous.
    await driver.execute_query(
        'CREATE (:Entity {uuid: "int-a", name: "Alpha cats", summary: "about felines", '
        f'group_id: "{GROUP}", created_at: "2026-01-01T00:00:00Z", '
        'name_embedding: [1.0, 0.0, 0.0]}) '
    )
    await driver.execute_query(
        'CREATE (:Entity {uuid: "int-b", name: "Beta dogs", summary: "about canines", '
        f'group_id: "{GROUP}", created_at: "2026-01-01T00:00:00Z", '
        'name_embedding: [0.0, 1.0, 0.0]}) '
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_NEO4J, reason='neo4j driver package is not installed')
async def test_drevo_node_similarity_int():
    """Vector similarity ranks by cosine (library-side) against a live drevo."""
    driver = await _connect_or_skip()
    try:
        await _seed(driver)
        # Query vector aligned with entity "int-a".
        results = await driver.search_interface.node_similarity_search(
            driver,
            search_vector=[1.0, 0.0, 0.0],
            search_filter=SearchFilters(),
            group_ids=[GROUP],
            limit=10,
            min_score=0.5,
        )
        uuids = [n.uuid for n in results]
        assert uuids == ['int-a'], uuids
    finally:
        await driver.execute_query(f'MATCH (n:Entity {{group_id: "{GROUP}"}}) DETACH DELETE n')
        await driver.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_NEO4J, reason='neo4j driver package is not installed')
async def test_drevo_node_fulltext_int():
    """Node full-text against a live drevo, preferring native BM25 fts.search."""
    driver = await _connect_or_skip()
    try:
        await _seed(driver)
        interface = driver.search_interface
        results = await interface.node_fulltext_search(
            driver,
            query='alpha cats',
            search_filter=SearchFilters(),
            group_ids=[GROUP],
            limit=10,
        )
        uuids = [n.uuid for n in results]
        assert uuids == ['int-a'], uuids
        # On a current drevo (fts.search available) the native BM25 path is taken,
        # not the lexical fallback.
        assert interface._native_fts is True, 'expected native fts.search, got fallback'
    finally:
        await driver.execute_query(f'MATCH (n:Entity {{group_id: "{GROUP}"}}) DETACH DELETE n')
        await driver.close()
