from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class AddMessagesRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the messages to add')
    messages: list[Message] = Field(..., description='The messages to add')
    edge_types: dict[str, str] | None = Field(
        default=None, description='Optional: Edge type names to descriptions for Graphiti extraction'
    )
    custom_extraction_instructions: str | None = Field(
        default=None, description='Optional: Custom instructions to guide the LLM during edge extraction'
    )


class AddEntityNodeRequest(BaseModel):
    uuid: str = Field(..., description='The uuid of the node to add')
    group_id: str = Field(..., description='The group id of the node to add')
    name: str = Field(..., description='The name of the node to add')
    summary: str = Field(default='', description='The summary of the node to add')
