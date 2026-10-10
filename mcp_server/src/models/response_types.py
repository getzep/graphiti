"""Response type definitions for Graphiti MCP Server."""

from typing import Any

from typing_extensions import NotRequired, TypedDict


class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


class NodeResult(TypedDict):
    uuid: str
    name: str
    labels: list[str]
    created_at: str | None
    summary: str | None
    group_id: str
    attributes: dict[str, Any]
    # Reranker score for this node; only present on search results. Score
    # semantics depend on the ranker reported in the enclosing response.
    score: NotRequired[float]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]
    # Which reranker ordered these results (e.g. 'rrf', 'node_distance').
    # The reranker is deployment config, so per-result scores are only
    # comparable against the same ranker value.
    ranker: str


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]
    # Which reranker ordered these results (e.g. 'rrf', 'node_distance').
    ranker: str
    # How many of the returned facts have been invalidated or superseded
    # (invalid_at/expired_at set). Lets a caller distinguish "no fact was
    # ever recorded" from "facts existed but expired" without any filtering.
    invalidated_count: int
    invalidated_uuids: list[str]


class EpisodeSearchResponse(TypedDict):
    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    status: str
    message: str


class SagaSummaryResponse(TypedDict):
    message: str
    uuid: str
    name: str
    summary: str


class CommunityResult(TypedDict):
    uuid: str
    name: str
    group_id: str
    summary: str | None


class BuildCommunitiesResponse(TypedDict):
    message: str
    community_count: int
    edge_count: int
    communities: list[CommunityResult]


class EdgeResult(TypedDict):
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    group_id: str
    created_at: str | None
    valid_at: str | None
    invalid_at: str | None


class TripletResponse(TypedDict):
    message: str
    nodes: list[NodeResult]
    edges: list[EdgeResult]


class EpisodeEntitiesResponse(TypedDict):
    message: str
    nodes: list[NodeResult]
    edges: list[EdgeResult]
