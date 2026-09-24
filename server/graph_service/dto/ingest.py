from graphiti_core.errors import GroupIdValidationError  # type: ignore
from graphiti_core.helpers import validate_group_id  # type: ignore
from pydantic import BaseModel, Field, field_validator

from graph_service.dto.common import Message


def _validate_request_group_id(value: str) -> str:
    # Re-raise as ValueError so Pydantic wraps it into a ValidationError (HTTP 422)
    # regardless of whether the installed graphiti-core makes GroupIdValidationError a
    # ValueError subclass. Rejecting bad group_ids on write keeps records reachable and
    # deletable via the read/delete API paths, which validate the same pattern.
    try:
        validate_group_id(value)
    except GroupIdValidationError as error:
        raise ValueError(str(error)) from error
    return value


class AddMessagesRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the messages to add')
    messages: list[Message] = Field(..., description='The messages to add')

    @field_validator('group_id')
    @classmethod
    def validate_group_id_field(cls, value: str) -> str:
        return _validate_request_group_id(value)


class AddEntityNodeRequest(BaseModel):
    uuid: str = Field(..., description='The uuid of the node to add')
    group_id: str = Field(..., description='The group id of the node to add')
    name: str = Field(..., description='The name of the node to add')
    summary: str = Field(default='', description='The summary of the node to add')

    @field_validator('group_id')
    @classmethod
    def validate_group_id_field(cls, value: str) -> str:
        return _validate_request_group_id(value)
