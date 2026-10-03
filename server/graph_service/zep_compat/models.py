"""Wire-exact request/response models for the Zep Cloud v2 subset MiroFish uses.

Field names here are the WIRE names on purpose. The zep-cloud SDK declares
`uuid_: Annotated[str, FieldMetadata(alias="uuid")]`, so the JSON key is `uuid`;
emitting `uuid_` makes the client raise ValidationError. Declaring plain `uuid`
here removes any chance of that mistake. See WIRE_SPEC.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def iso(value: datetime | None) -> str | None:
    """Render a datetime the way the Zep API does: RFC3339 with a Z suffix."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def now_iso() -> str:
    return iso(datetime.now(timezone.utc))  # type: ignore[return-value]


EpisodeSource = Literal['text', 'json', 'message', 'fact_triple']
BatchStatus = Literal[
    'draft', 'invalid', 'queued', 'processing', 'succeeded', 'partial', 'failed', 'canceled'
]
BatchItemStatus = Literal[
    'pending', 'queued', 'processing', 'succeeded', 'failed', 'skipped', 'canceled'
]
PropertyType = Literal['Text', 'Int', 'Float', 'Boolean']


# ---------------------------------------------------------------------------
# responses
# ---------------------------------------------------------------------------


class SuccessResponse(BaseModel):
    message: str | None = 'ok'


class Graph(BaseModel):
    uuid: str | None = None
    graph_id: str | None = None
    name: str | None = None
    description: str | None = None
    created_at: str | None = None
    project_uuid: str | None = None
    time_zone: str | None = None
    id: int | None = None


class Episode(BaseModel):
    uuid: str
    content: str
    created_at: str
    source: EpisodeSource | None = 'text'
    source_description: str | None = None
    metadata: dict[str, Any] | None = None
    processed: bool | None = None
    role: str | None = None
    role_type: str | None = None
    thread_id: str | None = None
    task_id: str | None = None
    score: float | None = None
    relevance: float | None = None
    selection_rank: int | None = None


class EntityNode(BaseModel):
    uuid: str
    name: str
    summary: str
    created_at: str
    labels: list[str] | None = None
    attributes: dict[str, Any] | None = None
    score: float | None = None
    relevance: float | None = None
    selection_rank: int | None = None


class EntityEdge(BaseModel):
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    created_at: str
    episodes: list[str] | None = None
    valid_at: str | None = None
    invalid_at: str | None = None
    expired_at: str | None = None
    attributes: dict[str, Any] | None = None
    scope: str | None = None
    score: float | None = None
    relevance: float | None = None
    selection_rank: int | None = None


class EpisodeResponse(BaseModel):
    episodes: list[Episode] | None = None


class EpisodeMentions(BaseModel):
    nodes: list[EntityNode] | None = None
    edges: list[EntityEdge] | None = None


class GraphSearchResults(BaseModel):
    nodes: list[EntityNode] | None = None
    edges: list[EntityEdge] | None = None
    episodes: list[Episode] | None = None
    context: str | None = None


class BatchProgress(BaseModel):
    total_items: int | None = None
    queued_items: int | None = None
    processing_items: int | None = None
    succeeded_items: int | None = None
    failed_items: int | None = None
    skipped_items: int | None = None
    canceled_items: int | None = None
    percent_complete: float | None = None


class BatchSummary(BaseModel):
    batch_id: str | None = None
    status: BatchStatus | None = None
    item_count: int | None = None
    metadata: dict[str, Any] | None = None
    progress: BatchProgress | None = None
    created_at: str | None = None
    updated_at: str | None = None
    processed_at: str | None = None
    completed_at: str | None = None
    ignore_roles: list[str] | None = None


class BatchItemDetail(BaseModel):
    item_id: str | None = None
    batch_id: str | None = None
    kind: Literal['graph_episode', 'thread_message'] | None = 'graph_episode'
    status: BatchItemStatus | None = None
    sequence_index: int | None = None
    graph_id: str | None = None
    graph_uuid: str | None = None
    episode_uuid: str | None = None
    source_uuid: str | None = None
    thread_id: str | None = None
    user_id: str | None = None
    user_uuid: str | None = None
    error: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None


class BatchItemListResponse(BaseModel):
    items: list[BatchItemDetail] | None = None
    next_cursor: int | None = None


class BatchListResponse(BaseModel):
    batches: list[BatchSummary] | None = None
    next_cursor: int | None = None


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------


class CreateGraphRequest(BaseModel):
    graph_id: str
    name: str | None = None
    description: str | None = None
    time_zone: str | None = None


class AddDataRequest(BaseModel):
    graph_id: str | None = None
    user_id: str | None = None
    data: str
    type: EpisodeSource = 'text'
    source_description: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] | None = None
    strict_ontology: bool | None = None


class SearchRequest(BaseModel):
    query: str
    graph_id: str | None = None
    user_id: str | None = None
    limit: int | None = 10
    scope: str | None = 'edges'
    reranker: str | None = None
    center_node_uuid: str | None = None
    bfs_origin_node_uuids: list[str] | None = None
    mmr_lambda: float | None = None
    max_characters: int | None = None
    return_raw_results: bool | None = None
    search_filters: dict[str, Any] | None = None


class ListByGraphRequest(BaseModel):
    """Body of POST graph/node/graph/{graph_id} and graph/edge/graph/{graph_id}."""

    limit: int | None = 100
    cursor: str | None = None
    uuid_cursor: str | None = None
    order_by: str | None = None
    direction: str | None = None
    filters: dict[str, Any] | None = None


class EntityProperty(BaseModel):
    name: str
    description: str
    type: PropertyType = 'Text'


class EntityEdgeSourceTarget(BaseModel):
    source: str | None = None
    target: str | None = None


class EntityType(BaseModel):
    name: str
    description: str
    properties: list[EntityProperty] | None = None
    identity_properties: list[str] | None = None


class EdgeType(BaseModel):
    name: str
    description: str
    properties: list[EntityProperty] | None = None
    source_targets: list[EntityEdgeSourceTarget] | None = None


class SetEntityTypesRequest(BaseModel):
    entity_types: list[EntityType] = Field(default_factory=list)
    edge_types: list[EdgeType] = Field(default_factory=list)
    graph_ids: list[str] | None = None
    user_ids: list[str] | None = None


class EntityTypeResponse(BaseModel):
    entity_types: list[EntityType] | None = None
    edge_types: list[EdgeType] | None = None


class CreateBatchRequest(BaseModel):
    metadata: dict[str, Any] | None = None
    ignore_roles: list[str] | None = None


class BatchAddItem(BaseModel):
    type: Literal['graph_episode', 'thread_message'] = 'graph_episode'
    graph_id: str | None = None
    user_id: str | None = None
    thread_id: str | None = None
    data: str | None = None
    content: str | None = None
    data_type: EpisodeSource | None = 'text'
    name: str | None = None
    source_description: str | None = None
    created_at: str | None = None
    role: str | None = None
    metadata: dict[str, Any] | None = None

    def payload(self) -> str:
        return self.data if self.data is not None else (self.content or '')


class AddBatchItemsRequest(BaseModel):
    items: list[BatchAddItem] = Field(default_factory=list)
