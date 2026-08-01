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
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager

from graphiti_core.driver.capabilities import GraphCapabilities
from graphiti_core.driver.drevo_search_interface import DrevoSearchInterface
from graphiti_core.driver.driver import GraphProvider, _SessionTransaction
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.driver.query_executor import Transaction

logger = logging.getLogger(__name__)


class DrevoDriver(Neo4jDriver):
    """Connector for `drevo <https://github.com/ice1x/drevo>`_.

    drevo speaks the Bolt wire protocol on port 7687 and accepts a Cypher subset,
    so the connector reuses the Neo4j Bolt machinery (connection, sessions, query
    execution, CRUD operations) and only diverges where drevo differs:

    * **No index DDL.** drevo's Cypher subset has no ``CREATE INDEX/FULLTEXT/VECTOR``
      statements — indexing is managed out-of-band — so index build/teardown are
      no-ops rather than emitting Neo4j index procedures.
    * **No explicit transactions.** The Bolt handshake advertises only
      ``RUN/PULL/DISCARD/RESET`` (no ``BEGIN/COMMIT``), so ``transaction()`` runs
      in immediate mode instead of Neo4j's begin/commit path.

    Search is served by :class:`DrevoSearchInterface` (set on the instance below
    and honored by ``graphiti_core.search.search_utils``): vector similarity uses
    drevo's native ``cosine_similarity`` scalar (drevo#202), with a library-side
    fallback for older builds; fulltext is an interim lexical match (no BM25 over
    Cypher yet — drevo#208); and BFS / rerankers / community embeddings fall back to
    the Neo4j-compatible inline Cypher that drevo's Cypher subset supports.
    """

    provider = GraphProvider.DREVO

    capabilities = GraphCapabilities(
        supports_transactions=False,
        # Native BM25 node full-text via fts.search (drevo#208/#227) and native
        # vector similarity via cosine_similarity (drevo#202); DrevoSearchInterface
        # falls back to lexical / library-side ranking on older builds.
        supports_native_fulltext_search=True,
        supports_native_vector_search=True,
        supports_vector_index=False,
    )

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
        database: str = 'neo4j',
    ) -> None:
        super().__init__(uri, user, password, database)
        self.search_interface = DrevoSearchInterface()

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        """No-op: drevo manages indexes out-of-band and has no index DDL."""
        return None

    def delete_all_indexes(self) -> Coroutine:
        """No-op: nothing to drop, and drevo exposes no index-management procedures."""

        async def _noop() -> None:
            return None

        return _noop()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        """Immediate-mode transaction — drevo's Bolt has no BEGIN/COMMIT, so
        queries execute as they run against a plain session."""
        session = self.session()
        try:
            yield _SessionTransaction(session)
        finally:
            await session.close()
