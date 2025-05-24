from datetime import datetime, timezone

from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class SearchQuery(BaseModel):
    group_ids: list[str] | None = Field(
        None, description='The group ids for the memories to search'
    )
    query: str
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')


class FactResult(BaseModel):
    uuid: str
    name: str
    fact: str
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime
    expired_at: datetime | None

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class SearchResults(BaseModel):
    facts: list[FactResult]


class GetEntityRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the entity to get')
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    center_node_uuid: str | None = Field(
        ..., description='The uuid of the node to center the retrieval on'
    )
    messages: list[Message] = Field(
        ..., description='The messages to build the retrieval query from '
    )


class GetEntityResponse(BaseModel):
    facts: list[FactResult] = Field(..., description='The facts that were retrieved from the graph')


class RelationItem(BaseModel):
    episodic_id: str = Field(..., description='UUID of the episodic node')
    text: str = Field(..., description='Text of the relation/fact/emotion/entity')


class GetRelationsRequest(BaseModel):
    group_id: str = Field(..., description='The group id to fetch relations for')
    relation_types: list[str] = Field(default_factory=lambda: ['emotions', 'relations', 'facts', 'entities'], description='The types of relations to retrieve')


class RelationsResponse(BaseModel):
    relations: dict[str, list[RelationItem]]
