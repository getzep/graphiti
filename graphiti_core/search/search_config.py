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

        When reranker scores are present, items are sorted by score descending.
        Duplicate UUIDs across partitions keep the highest score. An optional
        ``limit`` caps each result list after merge (multi-group search).

        Parameters
        ----------
        results_list : list[SearchResults]
            List of SearchResults objects to merge
        limit : int | None
            Optional cap applied to each merged item list after ranking

        Returns
        -------
        SearchResults
            A single SearchResults object containing all results
        """
        if not results_list:
            return cls()

        def _merge_scored(items_lists: list, scores_lists: list) -> tuple[list, list[float]]:
            has_any_scores = any(bool(s) for s in scores_lists)
            best: dict[str, tuple[object, float | None]] = {}
            order: list[str] = []
            for items, scores in zip(items_lists, scores_lists, strict=False):
                for i, item in enumerate(items):
                    score: float | None = scores[i] if i < len(scores) else None
                    uuid = getattr(item, 'uuid', None)
                    key = uuid if uuid is not None else f'__idx_{id(item)}'
                    if key not in best:
                        order.append(key)
                        best[key] = (item, score)
                    else:
                        prev = best[key][1]
                        if score is not None and (prev is None or score > prev):
                            best[key] = (item, score)

            if has_any_scores:
                paired = sorted(
                    best.values(),
                    key=lambda x: (x[1] is not None, x[1] if x[1] is not None else float('-inf')),
                    reverse=True,
                )
            else:
                paired = [best[k] for k in order]

            if limit is not None:
                paired = paired[:limit]

            out_items = [p[0] for p in paired]
            if not has_any_scores:
                return out_items, []
            return out_items, [float('-inf') if p[1] is None else p[1] for p in paired]

        nodes, node_scores = _merge_scored(
            [r.nodes for r in results_list],
            [r.node_reranker_scores for r in results_list],
        )
        edges, edge_scores = _merge_scored(
            [r.edges for r in results_list],
            [r.edge_reranker_scores for r in results_list],
        )
        episodes, episode_scores = _merge_scored(
            [r.episodes for r in results_list],
            [r.episode_reranker_scores for r in results_list],
        )
        communities, community_scores = _merge_scored(
            [r.communities for r in results_list],
            [r.community_reranker_scores for r in results_list],
        )

        return cls(
            nodes=nodes,
            node_reranker_scores=node_scores,
            edges=edges,
            edge_reranker_scores=edge_scores,
            episodes=episodes,
            episode_reranker_scores=episode_scores,
            communities=communities,
            community_reranker_scores=community_scores,
        )
