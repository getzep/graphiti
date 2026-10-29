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

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from graphiti_core.driver.driver import GraphDriverSession, GraphProvider
from graphiti_core.driver.spanner.transaction import SpannerTransaction

if TYPE_CHECKING:
    from graphiti_core.driver.spanner.driver import SpannerDriver


class SpannerSession(GraphDriverSession):
    """Session facade over a :class:`SpannerDriver`; holds no server-side resources."""

    provider = GraphProvider.SPANNER

    def __init__(self, driver: SpannerDriver):
        self._driver = driver

    @property
    def driver(self) -> SpannerDriver:
        return self._driver

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def run(self, query: str, **kwargs: Any) -> Any:
        return await self._driver.execute_query(query, **kwargs)

    async def close(self):
        pass

    async def execute_write(
        self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """Call ``func(tx, *args, **kwargs)`` with a fresh :class:`SpannerTransaction` and
        commit its buffered mutations after it returns; discard them if it raises.

        Keyword arguments such as ``driver=`` pass through unchanged, as
        ``bulk_utils.add_nodes_and_edges_bulk`` requires.
        """
        tx = SpannerTransaction(self._driver)
        try:
            result = await func(tx, *args, **kwargs)
        except BaseException:
            tx.discard()
            raise
        await tx.commit()
        return result
