import json
from datetime import datetime
from typing import Annotated, Literal, Union, Any

from graphiti_core.utils.datetime_utils import utc_now
from pydantic import BaseModel, Field, BeforeValidator


def normalize_episode_body(v: Any) -> str:
    """
    Normalize episode_body to always be a string.
    Accepts: str, dict, list, or any JSON-serializable object
    Returns: JSON string representation
    """
    if isinstance(v, str):
        # If it's already a string, check if it's valid JSON
        try:
            # Try to parse it as JSON to validate it
            json.loads(v)
            # If parsing succeeds, return the original string
            return v
        except (json.JSONDecodeError, TypeError):
            # If it's not valid JSON, treat it as a plain string and wrap it
            return json.dumps(v, ensure_ascii=False)
    else:
        # If it's not a string (dict, list, etc.), convert to JSON string
        return json.dumps(v, ensure_ascii=False)


class Result(BaseModel):
    message: str
    success: bool


class Message(BaseModel):
    content: str = Field(..., description='The content of the message')
    uuid: str | None = Field(default=None, description='The uuid of the message (optional)')
    name: str = Field(
        default='', description='The name of the episodic node for the message (optional)'
    )
    role_type: Literal['user', 'assistant', 'system'] = Field(
        ..., description='The role type of the message (user, assistant or system)'
    )
    role: str | None = Field(
        description='The custom role of the message to be used alongside role_type (user name, bot name, etc.)',
    )
    timestamp: datetime = Field(default_factory=utc_now, description='The timestamp of the message')
    source_description: str = Field(
        default='', description='The description of the source of the message'
    )


class Episode(BaseModel):
    uuid: str | None = Field(default=None, description='The uuid of the episode (optional)')
    name: str = Field(..., description='The name of the episode')
    episode_body: Annotated[str, BeforeValidator(normalize_episode_body), Field(description='The content of the episode - accepts str, dict, list, or any JSON-serializable object and converts to JSON string')]
    source_description: str = Field(..., description='The description of the source of the episode')
    reference_time: datetime = Field(default_factory=utc_now, description='The reference time of the episode')
    source: Literal['message', 'json', 'text'] = Field(default='text', description='The source type of the episode')
    update_communities: bool = Field(default=False, description='Whether to update communities with new node information')


class MetricsResponse(BaseModel):
    queue_size: int = Field(..., description='Current number of jobs in the processing queue')
    worker_status: str = Field(..., description='Status of the async worker (running/stopped)')
    timestamp: datetime = Field(default_factory=utc_now, description='Timestamp when metrics were collected')