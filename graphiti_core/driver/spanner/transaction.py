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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphiti_core.driver.query_executor import Transaction
from graphiti_core.driver.spanner.mutations import Mutation

if TYPE_CHECKING:
    from graphiti_core.driver.spanner.driver import SpannerDriver


class SpannerTransaction(Transaction):
    """A buffered mutation batch, committed after the caller's work returns.

    Ops write through ``driver.apply_mutations(mutations, tx=tx)``, which buffers here.
    ``commit()`` applies the buffer with ``SpannerDriver.apply_mutations`` (in order, split
    into commits of at most ``mutation_chunk_rows`` rows, so a very large buffer is not
    atomic). This deliberately avoids the client's retrying ``run_in_transaction``: Graphiti's
    bulk callback calls the embedder inside the transaction, and a retry would repeat it.

    ``run()`` executes immediately through ``SpannerDriver.execute_query``, outside the
    buffer: reads never see buffered writes, and DML passed to ``run()`` commits on its own.
    """

    def __init__(self, driver: SpannerDriver):
        self._driver = driver
        self._mutations: list[Mutation] = []

    @property
    def driver(self) -> SpannerDriver:
        return self._driver

    @property
    def mutations(self) -> tuple[Mutation, ...]:
        return tuple(self._mutations)

    def buffer(self, mutation: Mutation) -> None:
        self._mutations.append(mutation)

    async def run(self, query: str, **kwargs: Any) -> Any:
        return await self._driver.execute_query(query, **kwargs)

    async def commit(self) -> None:
        mutations = self._mutations
        self._mutations = []
        await self._driver.apply_mutations(mutations)

    def discard(self) -> None:
        self._mutations.clear()
