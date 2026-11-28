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
from typing import TypeVar

from pydantic import BaseModel, Field

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
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


class EpisodeSearchMethod(Enum):
    bm25 = 'bm25'


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


class EpisodeReranker(Enum):
    rrf = 'reciprocal_rank_fusion'
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


class EpisodeSearchConfig(BaseModel):
    search_methods: list[EpisodeSearchMethod]
    reranker: EpisodeReranker = Field(default=EpisodeReranker.rrf)
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
    episode_config: EpisodeSearchConfig | None = Field(default=None)
    community_config: CommunitySearchConfig | None = Field(default=None)
    limit: int = Field(default=DEFAULT_SEARCH_LIMIT)
    reranker_min_score: float = Field(default=0)


T = TypeVar('T')


def _merge_scored_items(
    results: list[tuple[list[T], list[float]]],
    limit: int | None = None,
) -> tuple[list[T], list[float]]:
    """Merge items and their corresponding scores across multiple result sets.

    Deduplicates items by UUID (keeping the highest score) and sorts descending
    by reranker score when scores are present. If no scores are present across all
    result sets, original encounter order is preserved with deduplication.
    If limit is specified, the merged results are truncated to that limit.
    """
    has_any_scores = any(bool(scores) for _, scores in results)

    # Map item uuid -> (item, score, encounter_order)
    seen: dict[str, tuple[T, float | None, int]] = {}
    order_counter = 0

    for items, scores in results:
        has_scores = bool(scores) and len(scores) == len(items)
        for i, item in enumerate(items):
            score = scores[i] if has_scores else None
            uuid = getattr(item, 'uuid', str(order_counter))
            if uuid not in seen:
                seen[uuid] = (item, score, order_counter)
                order_counter += 1
            else:
                existing_item, existing_score, existing_order = seen[uuid]
                if score is not None and (existing_score is None or score > existing_score):
                    seen[uuid] = (item, score, existing_order)

    entries = list(seen.values())

    if has_any_scores:
        # Sort descending by score; missing score defaults to float('-inf') during sorting
        entries.sort(key=lambda x: (x[1] if x[1] is not None else float('-inf')), reverse=True)
    else:
        # Preserve first-encounter order
        entries.sort(key=lambda x: x[2])

    if limit is not None and limit > 0:
        entries = entries[:limit]

    merged_items = [e[0] for e in entries]
    merged_scores = [e[1] if e[1] is not None else 0.0 for e in entries] if has_any_scores else []

    return merged_items, merged_scores


class SearchResults(BaseModel):
    edges: list[EntityEdge] = Field(default_factory=list)
    edge_reranker_scores: list[float] = Field(default_factory=list)
    nodes: list[EntityNode] = Field(default_factory=list)
    node_reranker_scores: list[float] = Field(default_factory=list)
    episodes: list[EpisodicNode] = Field(default_factory=list)
    episode_reranker_scores: list[float] = Field(default_factory=list)
    communities: list[CommunityNode] = Field(default_factory=list)
    community_reranker_scores: list[float] = Field(default_factory=list)

    @classmethod
    def merge(
        cls, results_list: list['SearchResults'], limit: int | None = None
    ) -> 'SearchResults':
        """
        Merge multiple SearchResults objects into a single SearchResults object.

        Parameters
        ----------
        results_list : list[SearchResults]
            List of SearchResults objects to merge
        limit : int | None, optional
            Maximum number of results to keep for each entity type. Defaults to None.

        Returns
        -------
        SearchResults
            A single SearchResults object containing deduplicated, score-ranked results
        """
        if not results_list:
            return cls()

        merged_edges, merged_edge_scores = _merge_scored_items(
            [(r.edges, r.edge_reranker_scores) for r in results_list], limit=limit
        )
        merged_nodes, merged_node_scores = _merge_scored_items(
            [(r.nodes, r.node_reranker_scores) for r in results_list], limit=limit
        )
        merged_episodes, merged_episode_scores = _merge_scored_items(
            [(r.episodes, r.episode_reranker_scores) for r in results_list], limit=limit
        )
        merged_communities, merged_community_scores = _merge_scored_items(
            [(r.communities, r.community_reranker_scores) for r in results_list], limit=limit
        )

        return cls(
            edges=merged_edges,
            edge_reranker_scores=merged_edge_scores,
            nodes=merged_nodes,
            node_reranker_scores=merged_node_scores,
            episodes=merged_episodes,
            episode_reranker_scores=merged_episode_scores,
            communities=merged_communities,
            community_reranker_scores=merged_community_scores,
        )
