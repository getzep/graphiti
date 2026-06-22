"""Response type definitions for Graphiti MCP Server."""

from typing import Any

from typing_extensions import TypedDict


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


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


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
