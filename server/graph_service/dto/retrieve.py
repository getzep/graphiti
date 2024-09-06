from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class SearchQuery(BaseModel):
    group_id: str = Field(..., description='The group id of the memory to get')
    query: str
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    search_type: Literal['facts', 'user_centered_facts'] = Field(
        default='facts', description='The type of search to perform'
    )


class FactResult(BaseModel):
    uuid: str
    name: str
    fact: str
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime
    expired_at: datetime | None


class SearchResults(BaseModel):
    facts: list[FactResult]


class GetMemoryRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the memory to get')
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    messages: list[Message] = Field(
        ..., description='The messages to build the retrieval query from '
    )


class GetMemoryResponse(BaseModel):
    facts: list[FactResult] = Field(..., description='The facts that were retrieved from the graph')
