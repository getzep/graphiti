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

import hashlib
import json
import logging
import typing
from abc import ABC, abstractmethod

import httpx
from diskcache import Cache
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from ..prompts.models import Message
from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .errors import RateLimitError

DEFAULT_TEMPERATURE = 0
DEFAULT_CACHE_DIR = './llm_cache'

logger = logging.getLogger(__name__)


def is_server_or_retry_error(exception):
    if isinstance(exception, (RateLimitError, json.decoder.JSONDecodeError)):
        return True

    return (
        isinstance(exception, httpx.HTTPStatusError) and 500 <= exception.response.status_code < 600
    )


class LLMClient(ABC):
    def __init__(self, config: LLMConfig | None, cache: bool = False):
        if config is None:
            config = LLMConfig()

        self.config = config
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.cache_enabled = cache
        self.cache_dir = None

        # Only create the cache directory if caching is enabled
        if self.cache_enabled:
            self.cache_dir = Cache(DEFAULT_CACHE_DIR)

    def _clean_input(self, input: str) -> str:
        """Clean input string of invalid unicode and control characters.

        Args:
            input: Raw input string to be cleaned

        Returns:
            Cleaned string safe for LLM processing
        """
        # Clean any invalid Unicode
        cleaned = input.encode('utf-8', errors='ignore').decode('utf-8')

        # Remove zero-width characters and other invisible unicode
        zero_width = '\u200b\u200c\u200d\ufeff\u2060'
        for char in zero_width:
            cleaned = cleaned.replace(char, '')

        # Remove control characters except newlines, returns, and tabs
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')

        return cleaned

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=10, min=5, max=120),
        retry=retry_if_exception(is_server_or_retry_error),
        after=lambda retry_state: logger.warning(
            f'Retrying {retry_state.fn.__name__ if retry_state.fn else "function"} after {retry_state.attempt_number} attempts...'
        )
        if retry_state.attempt_number > 1
        else None,
        reraise=True,
    )
    async def _generate_response_with_retry(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, typing.Any]:
        try:
            return await self._generate_response(messages, response_model, max_tokens)
        except (httpx.HTTPStatusError, RateLimitError) as e:
            raise e

    @abstractmethod
    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, typing.Any]:
        pass

    def _get_cache_key(self, messages: list[Message]) -> str:
        # Create a unique cache key based on the messages and model
        message_str = json.dumps([m.model_dump() for m in messages], sort_keys=True)
        key_str = f'{self.model}:{message_str}'
        return hashlib.md5(key_str.encode()).hexdigest()

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, typing.Any]:
        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f'\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
            )

        if self.cache_enabled and self.cache_dir is not None:
            cache_key = self._get_cache_key(messages)

            cached_response = self.cache_dir.get(cache_key)
            if cached_response is not None:
                logger.debug(f'Cache hit for {cache_key}')
                return cached_response

        for message in messages:
            message.content = self._clean_input(message.content)

        response = await self._generate_response_with_retry(messages, response_model, max_tokens)

        if self.cache_enabled and self.cache_dir is not None:
            cache_key = self._get_cache_key(messages)
            self.cache_dir.set(cache_key, response)

        return response
