from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any

from graph_service.dto.common import Message


class ChatHistoryMessage(BaseModel):
    role: str = Field(..., description='The role of the message (user or assistant)')
    content: str = Field(..., description='The content of the message')


class AddMessagesRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the messages to add')
    messages: list[Message] = Field(..., description='The messages to add')
    chat_history: Union[List[ChatHistoryMessage], str, None] = Field(
        default=None, 
        description='The chat history - can be a list of messages with role/content or a string for backward compatibility (optional)'
    )
    shirt_slug: str | None = Field(default=None, description='The shirt slug (optional)')


class AddEntityNodeRequest(BaseModel):
    uuid: str = Field(..., description='The uuid of the node to add')
    group_id: str = Field(..., description='The group id of the node to add')
    name: str = Field(..., description='The name of the node to add')
    summary: str = Field(default='', description='The summary of the node to add')
