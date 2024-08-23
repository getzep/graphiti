import json
import logging
import typing

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    def __init__(self, config: LLMConfig):
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model

    def get_embedder(self) -> typing.Any:
        return self.client.embeddings

    async def generate_response(self, messages: list[Message]) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=0.1,
                max_tokens=3000,
                response_format={'type': 'json_object'},
            )
            result = response.choices[0].message.content or ''
            return json.loads(result)
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise
