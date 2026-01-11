from datetime import datetime, timezone

from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class SearchQuery(BaseModel):
    group_ids: list[str] | None = Field(
        None, description='The group ids for the memories to search'
    )
    query: str
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    meeting_ids: list[str] | None = Field(
        None, description='Optional: Filter results to specific meeting IDs'
    )
    meeting_type_ids: list[str] | None = Field(
        None, description='Optional: Filter results to specific meeting type IDs'
    )
    user_ids: list[str] | None = Field(
        None, description='Optional: Filter results to meetings where user is owner or has direct access'
    )


class SourceEpisode(BaseModel):
    """Source episode information extracted from metadata"""
    uuid: str
    name: str
    meeting_id: str | None = None
    meeting_type_id: str | None = None
    owner_id: str | None = None
    valid_at: datetime | None = None

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class FactResult(BaseModel):
    uuid: str
    name: str
    fact: str
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime
    expired_at: datetime | None
    source_episodes: list[SourceEpisode] = Field(
        default_factory=list,
        description='List of source episodes where this fact was extracted from, including meeting metadata'
    )

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class SearchResults(BaseModel):
    facts: list[FactResult]


class GetMemoryRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the memory to get')
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    center_node_uuid: str | None = Field(
        ..., description='The uuid of the node to center the retrieval on'
    )
    messages: list[Message] = Field(
        ..., description='The messages to build the retrieval query from '
    )


class GetMemoryResponse(BaseModel):
    facts: list[FactResult] = Field(..., description='The facts that were retrieved from the graph')
