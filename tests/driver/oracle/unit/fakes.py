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

from types import SimpleNamespace
from typing import Any


class FakeLOB:
    def __init__(self, lob_type: object, data: str | bytes | None) -> None:
        self.lob_type = lob_type
        self.data = data


class FakeCursor:
    def __init__(
        self,
        *,
        rows: list[tuple[Any, ...]] | None = None,
        columns: tuple[str, ...] = (),
    ) -> None:
        self.rows = rows or []
        self.description = [SimpleNamespace(name=name) for name in columns] or None
        self.executions: list[tuple[str, dict[str, object]]] = []
        self.input_size_calls: list[dict[str, object]] = []
        self.events: list[tuple[str, dict[str, object]]] = []

    async def __aenter__(self) -> FakeCursor:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def execute(self, statement: str, parameters: dict[str, object]) -> None:
        self.executions.append((statement, parameters))
        self.events.append(('execute', dict(parameters)))

    def setinputsizes(self, **kwargs: object) -> dict[str, object]:
        self.input_size_calls.append(dict(kwargs))
        self.events.append(('setinputsizes', dict(kwargs)))
        return dict(kwargs)

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor | None = None) -> None:
        self.fake_cursor = cursor or FakeCursor()
        self.commits = 0
        self.rollbacks = 0
        self.lob_creations: list[FakeLOB] = []

    def cursor(self) -> FakeCursor:
        return self.fake_cursor

    async def createlob(self, lob_type: object, data: str | bytes | None = None) -> FakeLOB:
        lob = FakeLOB(lob_type, data)
        self.lob_creations.append(lob)
        return lob

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


class FakePool:
    def __init__(self, connection: FakeConnection | None = None) -> None:
        self.connection = connection or FakeConnection()
        self.acquired = 0
        self.released = 0
        self.closed = 0

    async def acquire(self) -> FakeConnection:
        self.acquired += 1
        return self.connection

    async def release(self, connection: FakeConnection) -> None:
        assert connection is self.connection
        self.released += 1

    async def close(self, *, force: bool = False) -> None:
        assert force is False
        self.closed += 1
