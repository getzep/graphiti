import typing
from abc import ABC, abstractmethod

from .config import LLMConfig
from .messages import Message


class LLMClient(ABC):
	@abstractmethod
	def __init__(self, config: LLMConfig):
		pass

	@abstractmethod
	async def generate_response(self, messages: list[Message]) -> dict[str, typing.Any]:
		pass
