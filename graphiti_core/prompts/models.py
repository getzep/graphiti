"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections.abc import Callable
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from .prompt_helpers import DO_NOT_ESCAPE_UNICODE


class Message(BaseModel):
    """LLM transport message (role + content). Used by LLMClient.generate_response."""

    role: str
    content: str


class PromptVersion(Protocol):
    def __call__(self, context: dict[str, Any]) -> list[Message]: ...


PromptFunction = Callable[[dict[str, Any]], list[Message]]


class SystemMessage(BaseModel):
    role: Literal['system'] = 'system'
    content: str


class UserMessage(BaseModel):
    role: Literal['user'] = 'user'
    content: str


class ChatPrompt(BaseModel):
    """Typed system+user prompt returned by all prompt builders."""

    system: SystemMessage
    user: UserMessage

    @classmethod
    def from_messages(cls, messages: list[Message]) -> 'ChatPrompt':
        if len(messages) != 2:
            raise ValueError('ChatPrompt requires exactly two messages')
        if messages[0].role != 'system' or messages[1].role != 'user':
            raise ValueError('ChatPrompt requires a system message followed by a user message')
        return cls(
            system=SystemMessage(content=messages[0].content),
            user=UserMessage(content=messages[1].content),
        )

    def as_messages(self, *, append_unicode_note: bool = True) -> list[Message]:
        """Render to transport messages.

        When ``append_unicode_note`` is True (default), appends the do-not-escape-unicode
        note to the system message content so LLM providers preserve non-ASCII characters.
        """
        system_content = self.system.content
        if append_unicode_note:
            system_content = system_content + DO_NOT_ESCAPE_UNICODE
        return [
            Message(role='system', content=system_content),
            Message(role='user', content=self.user.content),
        ]


class PromptSpec(BaseModel):
    """Fixed schema registry entry for a production prompt.

    Schemas are not overridable — only text builders may be customized.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    response_model: type[BaseModel] | None = None
    dynamic_schema: bool = False


ChatPromptFunction = Callable[[dict[str, Any]], ChatPrompt]
