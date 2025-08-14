from pydantic import BaseModel, Field

from .common import Episode, Message


class AddMessagesRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the messages to add')
    messages: list[Message] = Field(..., description='The messages to add')


class AddEntityNodeRequest(BaseModel):
    uuid: str = Field(..., description='The uuid of the node to add')
    group_id: str = Field(..., description='The group id of the node to add')
    name: str = Field(..., description='The name of the node to add')
    summary: str = Field(default='', description='The summary of the node to add')


class AddEpisodesRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the episodes to add')
    episodes: list[Episode] = Field(..., description='The episodes to add')
