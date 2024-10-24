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

from enum import Enum

from pydantic import BaseModel, Field

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode
from graphiti_core.search.search_utils import (
    DEFAULT_MIN_SCORE,
    DEFAULT_MMR_LAMBDA,
    MAX_SEARCH_DEPTH,
)

DEFAULT_SEARCH_LIMIT = 10


class EdgeSearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'
    bfs = 'breadth_first_search'


class NodeSearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'
    bfs = 'breadth_first_search'


class CommunitySearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'


class EdgeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    node_distance = 'node_distance'
    episode_mentions = 'episode_mentions'
    mmr = 'mmr'
    cross_encoder = 'cross_encoder'


class NodeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    node_distance = 'node_distance'
    episode_mentions = 'episode_mentions'
    mmr = 'mmr'
    cross_encoder = 'cross_encoder'


class CommunityReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    mmr = 'mmr'
    cross_encoder = 'cross_encoder'


class EdgeSearchConfig(BaseModel):
    search_methods: list[EdgeSearchMethod]
    reranker: EdgeReranker = Field(default=EdgeReranker.rrf)
    sim_min_score: float = Field(default=DEFAULT_MIN_SCORE)
    mmr_lambda: float = Field(default=DEFAULT_MMR_LAMBDA)
    bfs_max_depth: int = Field(default=MAX_SEARCH_DEPTH)


class NodeSearchConfig(BaseModel):
    search_methods: list[NodeSearchMethod]
    reranker: NodeReranker = Field(default=NodeReranker.rrf)
    sim_min_score: float = Field(default=DEFAULT_MIN_SCORE)
    mmr_lambda: float = Field(default=DEFAULT_MMR_LAMBDA)
    bfs_max_depth: int = Field(default=MAX_SEARCH_DEPTH)


class CommunitySearchConfig(BaseModel):
    search_methods: list[CommunitySearchMethod]
    reranker: CommunityReranker = Field(default=CommunityReranker.rrf)
    sim_min_score: float = Field(default=DEFAULT_MIN_SCORE)
    mmr_lambda: float = Field(default=DEFAULT_MMR_LAMBDA)
    bfs_max_depth: int = Field(default=MAX_SEARCH_DEPTH)


class SearchConfig(BaseModel):
    edge_config: EdgeSearchConfig | None = Field(default=None)
    node_config: NodeSearchConfig | None = Field(default=None)
    community_config: CommunitySearchConfig | None = Field(default=None)
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT)


class SearchResults(BaseModel):
    edges: list[EntityEdge]
    nodes: list[EntityNode]
    communities: list[CommunityNode]
