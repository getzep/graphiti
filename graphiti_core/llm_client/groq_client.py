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

import json
import logging
import typing

from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam
from openai import AsyncOpenAI

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'llama-3.1-70b-versatile'


class GroqClient(LLMClient):
    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        if config is None:
            config = LLMConfig()
        super().__init__(config, cache)
        self.client = AsyncGroq(api_key=config.api_key)

    def get_embedder(self) -> typing.Any:
        openai_client = AsyncOpenAI()
        return openai_client.embeddings

    async def _generate_response(self, messages: list[Message]) -> dict[str, typing.Any]:
        msgs: list[ChatCompletionMessageParam] = []
        for m in messages:
            if m.role == 'user':
                msgs.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                msgs.append({'role': 'system', 'content': m.content})
        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=msgs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={'type': 'json_object'},
            )
            result = response.choices[0].message.content or ''
            return json.loads(result)
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise
