import json
import logging

from openai import AsyncOpenAI

from .client import LLMClient
from .config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    def __init__(self, config: LLMConfig):
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model

    async def generate_response(self, messages: list[dict[str, str]]) -> dict[str, any]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in generating LLM response: {e}")
            raise
