from typing import Any, Callable, Protocol

from pydantic import BaseModel


class Message(BaseModel):
	role: str
	content: str


class PromptVersion(Protocol):
	def __call__(self, context: dict[str, Any]) -> list[Message]: ...


PromptFunction = Callable[[dict[str, Any]], list[Message]]
