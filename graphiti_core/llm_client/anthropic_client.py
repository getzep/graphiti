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

import anthropic
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig
from .errors import RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'claude-3-7-sonnet-latest'
DEFAULT_MAX_TOKENS = 8192


class AnthropicClient(LLMClient):
    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        if config is None:
            config = LLMConfig(max_tokens=DEFAULT_MAX_TOKENS)
        elif config.max_tokens is None:
            config.max_tokens = DEFAULT_MAX_TOKENS
        super().__init__(config, cache)

        self.client = AsyncAnthropic(
            api_key=config.api_key,
            # we'll use tenacity to retry
            max_retries=1,
        )

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, typing.Any]:
        system_message = messages[0]
        user_messages = [{'role': m.role, 'content': m.content} for m in messages[1:]] + [
            {'role': 'assistant', 'content': '{'}
        ]

        # Ensure max_tokens is not greater than config.max_tokens or DEFAULT_MAX_TOKENS
        max_tokens = min(max_tokens, self.config.max_tokens, DEFAULT_MAX_TOKENS)

        try:
            result = await self.client.messages.create(
                system='Only include JSON in the response. Do not include any additional text or explanation of the content.\n'
                + system_message.content,
                max_tokens=max_tokens,
                temperature=self.temperature,
                messages=user_messages,  # type: ignore
                model=self.model or DEFAULT_MODEL,
            )

            return json.loads('{' + result.content[0].text)  # type: ignore
        except anthropic.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise
