from typing import Literal

from pydantic import BaseModel, Field


class ResolveGroupIdRequest(BaseModel):
    scope: Literal['user', 'workspace', 'session'] = Field(
        ..., description='The scope to resolve a group id for'
    )
    key: str = Field(..., min_length=1, description='Stable key used to derive the group id')


class ResolveGroupIdResponse(BaseModel):
    group_id: str
