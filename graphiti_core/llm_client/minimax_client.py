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
from typing import Any, ClassVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, get_extraction_language_instruction
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'MiniMax-M2.5'
DEFAULT_SMALL_MODEL = 'MiniMax-M2.5-highspeed'
DEFAULT_BASE_URL = 'https://api.minimax.io/v1'


class MiniMaxClient(LLMClient):
    """
    MiniMaxClient is a client class for interacting with MiniMax's language models.

    MiniMax provides an OpenAI-compatible API, so this client uses the OpenAI SDK
    with MiniMax's base URL and model names.

    Available models:
        - MiniMax-M2.5: Full capability model with 204K context window
        - MiniMax-M2.5-highspeed: Fast model optimized for speed

    Note:
        MiniMax requires temperature to be in the range (0.0, 1.0].
        A temperature of 0 is not allowed and will be automatically adjusted to 0.01.

    Attributes:
        client (AsyncOpenAI): The OpenAI-compatible client configured for MiniMax API.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        if cache:
            raise NotImplementedError('Caching is not implemented for MiniMax client')

        if config is None:
            config = LLMConfig()

        # Set MiniMax-specific defaults
        if config.base_url is None:
            config.base_url = DEFAULT_BASE_URL
        if config.model is None:
            config.model = DEFAULT_MODEL
        if config.small_model is None:
            config.small_model = DEFAULT_SMALL_MODEL

        # MiniMax requires temperature > 0
        if config.temperature <= 0:
            config.temperature = 0.01

        super().__init__(config, cache)
        self.max_tokens = max_tokens

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})

        model = self.model or DEFAULT_MODEL
        if model_size == ModelSize.small:
            model = self.small_model or DEFAULT_SMALL_MODEL

        try:
            response_format: dict[str, Any] = {'type': 'json_object'}
            if response_model is not None:
                schema_name = getattr(response_model, '__name__', 'structured_response')
                json_schema = response_model.model_json_schema()
                response_format = {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': schema_name,
                        'schema': json_schema,
                    },
                }

            response = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format=response_format,  # type: ignore[arg-type]
            )
            result = response.choices[0].message.content or ''
            return json.loads(result)
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating MiniMax LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'minimax',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            retry_count = 0
            last_error = None

            while retry_count <= self.MAX_RETRIES:
                try:
                    response = await self._generate_response(
                        messages, response_model, max_tokens=max_tokens, model_size=model_size
                    )
                    return response
                except (RateLimitError, RefusalError):
                    span.set_status('error', str(last_error))
                    raise
                except (
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ):
                    span.set_status('error', str(last_error))
                    raise
                except Exception as e:
                    last_error = e

                    if retry_count >= self.MAX_RETRIES:
                        logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                        span.set_status('error', str(e))
                        span.record_exception(e)
                        raise

                    retry_count += 1

                    error_context = (
                        f'The previous response attempt was invalid. '
                        f'Error type: {e.__class__.__name__}. '
                        f'Error details: {str(e)}. '
                        f'Please try again with a valid response, ensuring the output matches '
                        f'the expected format and constraints.'
                    )

                    error_message = Message(role='user', content=error_context)
                    messages.append(error_message)
                    logger.warning(
                        f'Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                    )

            span.set_status('error', str(last_error))
            raise last_error or Exception('Max retries exceeded with no specific error')
