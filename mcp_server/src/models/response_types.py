"""Response type definitions for Graphiti MCP Server."""

from typing import Any

from typing_extensions import TypedDict


class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


class AddMemoryResponse(TypedDict):
    message: str
    episode_uuid: str
    group_id: str
    queue_position: int


class IngestStatusResponse(TypedDict):
    message: str
    episode_uuid: str
    group_id: str
    state: str
    queue_depth: int
    queue_position: int | None
    queued_at: str
    started_at: str | None
    processed_at: str | None
    last_error: str | None


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
    details: dict[str, Any] | None
