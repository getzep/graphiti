from datetime import datetime

from graphiti_core.nodes import EpisodeType  # type: ignore
from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class Episode(BaseModel):
    name: str = Field(..., description='The name of the episode')
    content: str = Field(..., description='The content of the episode')
    timestamp: datetime = Field(
        default_factory=datetime.now, description='The timestamp of the episode'
    )
    type: EpisodeType = Field(
        default=EpisodeType.message, description='The type of the episode (json or message)'
    )
    source_description: str = Field(
        default='', description='The description of the source of the episode'
    )


class AddMessagesRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the messages to add')
    messages: list[Message] = Field(..., description='The messages to add')
