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

from typing import Any

from pydantic import BaseModel


class SearchInterface(BaseModel):
    """
    This is an interface for implementing custom search logic
    """

    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        raise NotImplementedError

    async def edge_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        raise NotImplementedError

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        raise NotImplementedError

    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        raise NotImplementedError

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,  # kept for parity even if unused in your impl
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        raise NotImplementedError

    # ---------- SEARCH FILTERS (sync) ----------
    def build_node_search_filters(self, search_filters: Any) -> Any:
        raise NotImplementedError

    def build_edge_search_filters(self, search_filters: Any) -> Any:
        raise NotImplementedError

    class Config:
        arbitrary_types_allowed = True
