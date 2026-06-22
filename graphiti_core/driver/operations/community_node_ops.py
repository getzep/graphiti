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

from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.nodes import CommunityNode


class CommunityNodeOperations(ABC):
    @abstractmethod
    async def save(
        self,
        executor: QueryExecutor,
        node: CommunityNode,
        tx: Transaction | None = None,
    ) -> None: ...

    @abstractmethod
    async def save_bulk(
        self,
        executor: QueryExecutor,
        nodes: list[CommunityNode],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None: ...

    @abstractmethod
    async def delete(
        self,
        executor: QueryExecutor,
        node: CommunityNode,
        tx: Transaction | None = None,
    ) -> None: ...

    @abstractmethod
    async def delete_by_group_id(
        self,
        executor: QueryExecutor,
        group_id: str,
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None: ...

    @abstractmethod
    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None: ...

    @abstractmethod
    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> CommunityNode: ...

    @abstractmethod
    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[CommunityNode]: ...

    @abstractmethod
    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityNode]: ...

    @abstractmethod
    async def load_name_embedding(
        self,
        executor: QueryExecutor,
        node: CommunityNode,
    ) -> None: ...
