from graphiti_core.helpers import validate_group_id  # type: ignore
from pydantic import BaseModel, Field, field_validator

from graph_service.dto.common import Message


class AddMessagesRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the messages to add')
    messages: list[Message] = Field(..., description='The messages to add')

    @field_validator('group_id')
    @classmethod
    def validate_group_id_field(cls, value: str) -> str:
        validate_group_id(value)
        return value


class AddEntityNodeRequest(BaseModel):
    uuid: str = Field(..., description='The uuid of the node to add')
    group_id: str = Field(..., description='The group id of the node to add')
    name: str = Field(..., description='The name of the node to add')
    summary: str = Field(default='', description='The summary of the node to add')

    @field_validator('group_id')
    @classmethod
    def validate_group_id_field(cls, value: str) -> str:
        validate_group_id(value)
        return value
