from abc import ABC, abstractmethod

from .config import LLMConfig


class LLMClient(ABC):
	@abstractmethod
	def __init__(self, config: LLMConfig):
		pass

	@abstractmethod
	async def generate_response(self, messages: list[dict[str, str]]) -> dict[str, any]:
		pass
