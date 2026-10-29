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

from contextlib import asynccontextmanager
from typing import Any


class RecordingExecutor:
    def __init__(
        self,
        records: list[dict[str, Any]] | None = None,
        *,
        responses: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.records = records or []
        self.responses = list(responses or [])
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute_query(self, statement: str, **binds: object):
        self.calls.append((statement, binds))
        records = self.responses.pop(0) if self.responses else self.records
        return records, list(records[0]) if records else [], None

    @asynccontextmanager
    async def transaction(self):
        transaction = RecordingTransaction(records=self.records, responses=self.responses)
        try:
            yield transaction
        finally:
            self.calls.extend(transaction.calls)
            self.responses = transaction.responses


class RecordingTransaction:
    def __init__(
        self,
        *,
        records: list[dict[str, Any]] | None = None,
        responses: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.records = records or []
        self.responses = list(responses or [])
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def run(self, statement: str, **binds: object):
        self.calls.append((statement, binds))
        records = self.responses.pop(0) if self.responses else self.records
        return records, list(records[0]) if records else [], None
