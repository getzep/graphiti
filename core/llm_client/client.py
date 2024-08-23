import typing
from abc import ABC, abstractmethod

from ..prompts.models import Message
from .config import LLMConfig


class LLMClient(ABC):
	@abstractmethod
	def __init__(self, config: LLMConfig):
		pass

	@abstractmethod
	def get_embedder(self) -> typing.Any:
		pass

	@abstractmethod
	async def generate_response(self, messages: list[Message]) -> dict[str, typing.Any]:
		pass
