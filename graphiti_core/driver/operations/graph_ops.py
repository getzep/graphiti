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

from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode


class GraphMaintenanceOperations(ABC):
    @abstractmethod
    async def clear_data(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None: ...

    @abstractmethod
    async def build_indices_and_constraints(
        self,
        executor: QueryExecutor,
        delete_existing: bool = False,
    ) -> None: ...

    @abstractmethod
    async def delete_all_indexes(
        self,
        executor: QueryExecutor,
    ) -> None: ...

    @abstractmethod
    async def get_community_clusters(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> list[Any]: ...

    @abstractmethod
    async def remove_communities(
        self,
        executor: QueryExecutor,
    ) -> None: ...

    @abstractmethod
    async def determine_entity_community(
        self,
        executor: QueryExecutor,
        entity: EntityNode,
    ) -> None: ...

    @abstractmethod
    async def get_mentioned_nodes(
        self,
        executor: QueryExecutor,
        episodes: list[EpisodicNode],
    ) -> list[EntityNode]: ...

    @abstractmethod
    async def get_communities_by_nodes(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
    ) -> list[CommunityNode]: ...
