from enum import Enum

from pydantic import Field, BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode, EntityNode, CommunityNode
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


class Reranker(Enum):
    rrf = 'reciprocal_rank_fusion'
    node_distance = 'node_distance'


class SearchEdges(BaseModel):
    num_edges: int = Field(default=10)
    search_methods: list[EdgeSearchMethod]
    reranker: Reranker | None


class SearchNodes(BaseModel):
    num_nodes: int = Field(default=10)
    search_methods: list[NodeSearchMethod]
    reranker: Reranker | None


class SearchEpisodes(BaseModel):
    num_edges: int = Field(default=EPISODE_WINDOW_LEN)
    search_methods: list[EpisodeSearchMethod]
    reranker: Reranker | None


class SearchCommunities(BaseModel):
    num_edges: int = Field(default=10)
    search_methods: list[CommunitySearchMethod]
    reranker: Reranker | None


class SearchConfig(BaseModel):
    edge_config: SearchEdges
    node_config: SearchNodes
    episode_config: SearchEpisodes
    community_config: SearchCommunities
    group_ids: list[str | None] | None


class SearchResults(BaseModel):
    edges: list[EntityEdge]
    nodes: list[EntityNode]
    episodes: list[EpisodicNode]
    communities: list[CommunityNode]
