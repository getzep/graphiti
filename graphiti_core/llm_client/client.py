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
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

DEFAULT_TEMPERATURE = 0
DEFAULT_CACHE_DIR = './llm_cache'

MULTILINGUAL_EXTRACTION_RESPONSES = (
    '\n\nAny extracted information should be returned in the same language as it was written in.'
)

logger = logging.getLogger(__name__)


def is_server_or_retry_error(exception):
    if isinstance(exception, RateLimitError | json.decoder.JSONDecodeError):
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
        self.small_model = config.small_model
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
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        try:
            return await self._generate_response(messages, response_model, max_tokens, model_size)
        except (httpx.HTTPStatusError, RateLimitError) as e:
            raise e

    @abstractmethod
    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        pass

    def _get_cache_key(self, messages: list[Message]) -> str:
        # Create a unique cache key based on the messages and model
        message_str = json.dumps([m.model_dump() for m in messages], sort_keys=True)
        key_str = f'{self.model}:{message_str}'
        return hashlib.md5(key_str.encode()).hexdigest()

    def _clean_json_response(self, response: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Clean potential markdown code blocks from LLM JSON responses.
        
        This method provides a safety net for LLMs that might wrap JSON in ```json``` blocks
        despite being instructed not to do so.
        
        Args:
            response: The response dictionary from the LLM
            
        Returns:
            Cleaned response dictionary
        """
        # If response has a 'content' key with string value, check for code block wrapping
        if isinstance(response, dict) and 'content' in response:
            content = response.get('content', '')
            if isinstance(content, str) and content.strip().startswith('```'):
                # Extract JSON from code block
                lines = content.strip().split('\n')
                # Remove first line if it's ```json or just ```
                if lines[0].strip().lower() in ('```json', '```'):
                    lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                # Rejoin and update content
                cleaned_content = '\n'.join(lines).strip()
                if cleaned_content:
                    try:
                        # Validate it's still valid JSON
                        import json
                        parsed = json.loads(cleaned_content)
                        # If it's a dict, return it directly; if string content, keep in content wrapper
                        if isinstance(parsed, dict):
                            return parsed
                        else:
                            response['content'] = cleaned_content
                    except json.JSONDecodeError:
                        # If cleaning broke JSON, return original
                        pass
        
        return response

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f'\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
            )

        # Add multilingual extraction instructions
        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        if self.cache_enabled and self.cache_dir is not None:
            cache_key = self._get_cache_key(messages)

            cached_response = self.cache_dir.get(cache_key)
            if cached_response is not None:
                logger.debug(f'Cache hit for {cache_key}')
                return cached_response

        for message in messages:
            message.content = self._clean_input(message.content)

        response = await self._generate_response_with_retry(
            messages, response_model, max_tokens, model_size
        )

        # Clean potential code block wrapping from response
        response = self._clean_json_response(response)

        if self.cache_enabled and self.cache_dir is not None:
            cache_key = self._get_cache_key(messages)
            self.cache_dir.set(cache_key, response)

        return response

    def _get_failed_generation_log(self, messages: list[Message], output: str | None) -> str:
        """
        Log the full input messages, the raw output (if any), and the exception for debugging failed generations.
        """
        log = ''
        log += f'Input messages: {json.dumps([m.model_dump() for m in messages], indent=2)}\n'
        if output is not None:
            if len(output) > 4000:
                log += f'Raw output: {output[:2000]}... (truncated) ...{output[-2000:]}\n'
            else:
                log += f'Raw output: {output}\n'
        else:
            log += 'No raw output available'
        return log
