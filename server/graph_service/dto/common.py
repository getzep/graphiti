from datetime import datetime
from typing import Any, Literal, Optional

from graphiti_core.utils.datetime_utils import utc_now
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage information including model and temperature"""
    input_tokens: int = Field(default=0, description='Number of input tokens used')
    output_tokens: int = Field(default=0, description='Number of output tokens used')
    total_tokens: int = Field(default=0, description='Total number of tokens used')
    model: Optional[str] = Field(default=None, description='Model used for the request')
    temperature: Optional[float] = Field(default=None, description='Temperature setting used')


class Result(BaseModel):
    message: str
    success: bool
    tokens: Optional[TokenUsage] = None  # Contains token usage information
    data: Optional[Any] = None  # Contains extracted data (facts, emotions, entities, etc.)


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
