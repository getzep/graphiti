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

---

Regression tests for #1258:

  - KuzuDriver was missing the ``_database`` attribute that
    ``Graphiti.add_episode`` and ``add_episode_bulk`` read.
  - KuzuDriver.build_indices_and_constraints was a no-op, so the FTS
    indices required by search queries never got created through the
    ``Graphiti.build_indices_and_constraints`` entry point.
"""

import unittest

import pytest

from graphiti_core.driver.driver import GraphProvider

try:
    from graphiti_core.driver.kuzu_driver import KuzuDriver

    HAS_KUZU = True
except ImportError:
    KuzuDriver = None
    HAS_KUZU = False


class TestKuzuDriverDatabaseAndFTS:
    """Regression tests for the two bugs reported in #1258."""

    @unittest.skipIf(not HAS_KUZU, 'kuzu is not installed')
    def test_provider_is_kuzu(self):
        driver = KuzuDriver(':memory:')
        assert driver.provider == GraphProvider.KUZU

    @unittest.skipIf(not HAS_KUZU, 'kuzu is not installed')
    def test_database_attribute_is_set(self):
        """Bug 1 from #1258: KuzuDriver.__init__ must set ``_database``.

        Without this, ``Graphiti.add_episode`` (graphiti.py:1032) and
        ``Graphiti.add_episode_bulk`` (graphiti.py:1113) raise
        ``AttributeError: 'KuzuDriver' object has no attribute '_database'``.
        Empty string matches ``GraphDriver.default_group_id: str = ''``.
        """
        driver = KuzuDriver(':memory:')
        assert hasattr(driver, '_database')
        assert driver._database == ''

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_KUZU, 'kuzu is not installed')
    async def test_build_indices_and_constraints_creates_fts_indices(self):
        """Bug 2 from #1258: ``build_indices_and_constraints`` was a no-op.

        ``Graphiti.build_indices_and_constraints`` calls into the driver's
        method (not graph_ops directly), so this is the entry point that
        must produce the FTS indices. Without them, subsequent
        ``CALL QUERY_FTS_INDEX`` raises:
            ``Binder exception: Table Entity doesn't have an index with name node_name_and_summary``
        """
        driver = KuzuDriver(':memory:')
        await driver.build_indices_and_constraints(delete_existing=False)

        # If FTS indices were built, this query runs without raising.
        # (Returns an empty result because no Entities were inserted.)
        rows, _, _ = await driver.execute_query(
            "CALL QUERY_FTS_INDEX('Entity', 'node_name_and_summary', 'test', TOP := 10) "
            'RETURN node.uuid AS uuid'
        )
        assert rows == []

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_KUZU, 'kuzu is not installed')
    async def test_build_indices_and_constraints_is_idempotent(self):
        """Calling ``build_indices_and_constraints`` twice with
        ``delete_existing=False`` must not raise. ``Graphiti`` may call
        it multiple times across a process lifetime."""
        driver = KuzuDriver(':memory:')
        await driver.build_indices_and_constraints(delete_existing=False)
        # Second call must not raise.
        await driver.build_indices_and_constraints(delete_existing=False)

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_KUZU, 'kuzu is not installed')
    async def test_search_pipeline_after_build_indices_works(self):
        """End-to-end smoke: after build_indices_and_constraints, the
        complete FTS pipeline (insert + search) succeeds. This is the
        scenario the original #1258 bug report blocks on."""
        from datetime import datetime, timezone

        driver = KuzuDriver(':memory:')
        await driver.build_indices_and_constraints(delete_existing=False)

        # Insert one Entity and search via the FTS index.
        await driver.execute_query(
            'CREATE (n:Entity {uuid: $uuid, name: $name, group_id: $gid, '
            'labels: $labels, created_at: $created_at, '
            'name_embedding: $embedding, summary: $summary, attributes: $attrs})',
            uuid='u1',
            name='Avaloq Core',
            gid='g',
            labels=['App'],
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            embedding=[0.1] * 1024,
            summary='core banking',
            attrs='{}',
        )

        rows, _, _ = await driver.execute_query(
            "CALL QUERY_FTS_INDEX('Entity', 'node_name_and_summary', 'Avaloq', TOP := 10) "
            'RETURN node.uuid AS uuid'
        )
        uuids = {r['uuid'] for r in rows}
        assert 'u1' in uuids
