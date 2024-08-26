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
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..prompts.models import Message
from .config import LLMConfig

DEFAULT_TEMPERATURE = 0
DEFAULT_CACHE_DIR = './llm_cache'

logger = logging.getLogger(__name__)


def is_server_error(exception):
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
        self.cache_dir = Cache(DEFAULT_CACHE_DIR)  # Create a cache directory

    @abstractmethod
    def get_embedder(self) -> typing.Any:
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(is_server_error),
    )
    async def _generate_response_with_retry(self, messages: list[Message]) -> dict[str, typing.Any]:
        try:
            return await self._generate_response(messages)
        except httpx.HTTPStatusError as e:
            if not is_server_error(e):
                raise Exception(f'LLM request error: {e}') from e
            else:
                raise

    @abstractmethod
    async def _generate_response(self, messages: list[Message]) -> dict[str, typing.Any]:
        pass

    def _get_cache_key(self, messages: list[Message]) -> str:
        # Create a unique cache key based on the messages and model
        message_str = json.dumps([m.model_dump() for m in messages], sort_keys=True)
        key_str = f'{self.model}:{message_str}'
        return hashlib.md5(key_str.encode()).hexdigest()

    async def generate_response(self, messages: list[Message]) -> dict[str, typing.Any]:
        if self.cache_enabled:
            cache_key = self._get_cache_key(messages)

            cached_response = self.cache_dir.get(cache_key)
            if cached_response is not None:
                logger.debug(f'Cache hit for {cache_key}')
                return cached_response

        response = await self._generate_response_with_retry(messages)

        if self.cache_enabled:
            self.cache_dir.set(cache_key, response)

        return response
