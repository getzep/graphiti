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

from abc import ABC, abstractmethod
from typing import Any


class Transaction(ABC):
    """Minimal transaction interface yielded by GraphDriver.transaction().

    For drivers with real transaction support (e.g., Neo4j), this wraps a native
    transaction with commit/rollback semantics. For drivers without transaction
    support, this is a thin wrapper where queries execute immediately.
    """

    @abstractmethod
    async def run(self, query: str, **kwargs: Any) -> Any: ...


class QueryExecutor(ABC):
    """Slim interface for executing queries against a graph database.

    GraphDriver extends this. Operations ABCs depend only on QueryExecutor
    (not GraphDriver), which avoids circular imports.
    """

    @abstractmethod
    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Any: ...
