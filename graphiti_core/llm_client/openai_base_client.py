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
from abc import abstractmethod
from typing import Any, ClassVar

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import MULTILINGUAL_EXTRACTION_RESPONSES, LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-mini'
DEFAULT_SMALL_MODEL = 'gpt-4.1-nano'


class BaseOpenAIClient(LLMClient):
    """
    Base client class for OpenAI-compatible APIs (OpenAI and Azure OpenAI).

    This class contains shared logic for both OpenAI and Azure OpenAI clients,
    reducing code duplication while allowing for implementation-specific differences.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI-based clients')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)
        self.max_tokens = max_tokens

    @abstractmethod
    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
    ) -> Any:
        """Create a completion using the specific client implementation."""
        pass

    @abstractmethod
    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
    ) -> Any:
        """Create a structured completion using the specific client implementation."""
        pass

    def _convert_messages_to_openai_format(
        self, messages: list[Message]
    ) -> list[ChatCompletionMessageParam]:
        """Convert internal Message format to OpenAI ChatCompletionMessageParam format."""
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
        return openai_messages

    def _get_model_for_size(self, model_size: ModelSize) -> str:
        """Get the appropriate model name based on the requested size."""
        if model_size == ModelSize.small:
            return self.small_model or DEFAULT_SMALL_MODEL
        else:
            return self.model or DEFAULT_MODEL

    def _handle_structured_response(self, response: Any) -> dict[str, Any]:
        """Handle structured response parsing and validation."""
        response_object = response.choices[0].message

        if response_object.parsed:
            return response_object.parsed.model_dump()
        elif response_object.refusal:
            raise RefusalError(response_object.refusal)
        else:
            raise Exception(f'Invalid response from LLM: {response_object.model_dump()}')

    def _handle_json_response(self, response: Any) -> dict[str, Any]:
        """Handle JSON response parsing."""
        result = response.choices[0].message.content or '{}'
        return json.loads(result)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """Generate a response using the appropriate client implementation."""
        openai_messages = self._convert_messages_to_openai_format(messages)
        model = self._get_model_for_size(model_size)

        try:
            if response_model:
                response = await self._create_structured_completion(
                    model=model,
                    messages=openai_messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    response_model=response_model,
                )
                return self._handle_structured_response(response)
            else:
                response = await self._create_completion(
                    model=model,
                    messages=openai_messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                return self._handle_json_response(response)

        except openai.LengthFinishReasonError as e:
            raise Exception(f'Output length exceeded max tokens {self.max_tokens}: {e}') from e
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """Generate a response with retry logic and error handling."""
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        last_error = None

        # Add multilingual extraction instructions
        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens, model_size
                )
                return response
            except (RateLimitError, RefusalError):
                # These errors should not trigger retries
                raise
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                # Let OpenAI's client handle these retries
                raise
            except Exception as e:
                last_error = e

                # Don't retry if we've hit the max retries
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                    raise

                retry_count += 1

                # Construct a detailed error message for the LLM
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

        # If we somehow get here, raise the last error
        raise last_error or Exception('Max retries exceeded with no specific error')
