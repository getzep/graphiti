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
                temperature=0,
                max_tokens=3000,
                response_format={'type': 'json_object'},
            )
            result = response.choices[0].message.content or ''
            return json.loads(result)
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise
