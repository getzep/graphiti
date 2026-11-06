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

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Any, Protocol


class ColumnProtocol(Protocol):
    name: str


class AsyncCursorProtocol(Protocol):
    description: Sequence[ColumnProtocol] | None

    async def __aenter__(self) -> AsyncCursorProtocol: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    def setinputsizes(self, **kwargs: object) -> object: ...

    async def execute(self, statement: str, parameters: dict[str, object]) -> None: ...

    async def fetchall(self) -> list[tuple[Any, ...]]: ...


class AsyncConnectionProtocol(Protocol):
    def cursor(self) -> AsyncCursorProtocol: ...

    async def createlob(
        self,
        lob_type: object,
        data: str | bytes | None = None,
    ) -> object: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


class AsyncPoolProtocol(Protocol):
    def acquire(self) -> Awaitable[AsyncConnectionProtocol]: ...

    async def release(self, connection: AsyncConnectionProtocol) -> None: ...

    async def close(self, *, force: bool = False) -> None: ...
