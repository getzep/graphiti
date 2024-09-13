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
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.utils.maintenance.graph_data_operations import EPISODE_WINDOW_LEN


class EdgeSearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'


class NodeSearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'


class EpisodeSearchMethod(Enum):
    node_connections = 'node_connections'
    edge_connections = 'edge_connections'


class CommunitySearchMethod(Enum):
    cosine_similarity = 'cosine_similarity'
    bm25 = 'bm25'


class EdgeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    node_distance = 'node_distance'


class NodeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    node_distance = 'node_distance'


class EpisodeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'


class CommunityReranker(Enum):
    rrf = 'reciprocal_rank_fusion'


class EdgeSearchConfig(BaseModel):
    num_edges: int = Field(default=10)
    search_methods: list[EdgeSearchMethod]
    reranker: EdgeReranker | None
    center_node_uuid: str | None = None


class NodeSearchConfig(BaseModel):
    num_nodes: int = Field(default=10)
    search_methods: list[NodeSearchMethod]
    reranker: NodeReranker | None
    center_node_uuid: str | None = None


class EpisodeSearchConfig(BaseModel):
    num_edges: int = Field(default=EPISODE_WINDOW_LEN)
    search_methods: list[EpisodeSearchMethod]
    reranker: EpisodeReranker | None


class CommunitySearchConfig(BaseModel):
    num_edges: int = Field(default=10)
    search_methods: list[CommunitySearchMethod]
    reranker: CommunityReranker | None


class SearchConfig(BaseModel):
    edge_config: EdgeSearchConfig
    node_config: NodeSearchConfig
    episode_config: EpisodeSearchConfig
    community_config: CommunitySearchConfig


class SearchResults(BaseModel):
    edges: list[EntityEdge]
    nodes: list[EntityNode]
    episodes: list[EpisodicNode]
    communities: list[CommunityNode]
