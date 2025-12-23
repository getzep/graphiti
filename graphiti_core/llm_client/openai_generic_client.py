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

DEFAULT_MODEL = 'gpt-4.1-mini'


class OpenAIGenericClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.

    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False, client: typing.Any = None):
            Initializes the OpenAIClient with the provided configuration, cache setting, and client.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = 16384,
    ):
        """
        Initialize the OpenAIGenericClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 16384 (16K) for better compatibility with local models.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Override max_tokens to support higher limits for local models
        self.max_tokens = max_tokens

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

        # Instance-level fallback state for providers that don't support json_schema
        # (e.g., LiteLLM with Gemini). Once set to True, remains True for client lifetime.
        self._use_json_object_mode: bool = False

    def _is_schema_returned_as_data(self, response: dict[str, Any]) -> bool:
        """Detect if the model returned the schema definition instead of data.

        When some providers (e.g., LiteLLM with Gemini) receive json_schema format,
        they return the schema definition itself instead of data conforming to the schema.

        Args:
            response: The parsed JSON response from the LLM

        Returns:
            True if the response appears to be a JSON Schema definition
        """
        # JSON Schema structural markers
        schema_indicators = {'properties', '$defs', '$schema', 'definitions'}

        # Quick check: if none of the indicators are present, it's not a schema
        if not any(key in response for key in schema_indicators):
            return False

        # Strong indicator: top-level "type": "object" with "properties"
        if response.get('type') == 'object' and 'properties' in response:
            return True

        # Another strong indicator: "required" as list of strings alongside "properties"
        if 'required' in response and 'properties' in response:
            required = response.get('required')
            if isinstance(required, list) and all(isinstance(item, str) for item in required):
                return True

        return False

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
        try:
            # Prepare response format based on mode
            response_format: dict[str, Any]
            if response_model is not None and not self._use_json_object_mode:
                # Preferred mode: use json_schema format (works with OpenAI, vLLM, etc.)
                schema_name = getattr(response_model, '__name__', 'structured_response')
                json_schema = response_model.model_json_schema()
                response_format = {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': schema_name,
                        'schema': json_schema,
                    },
                }
            else:
                # Fallback mode: use json_object format with schema embedded in prompt
                # (for providers that don't support json_schema, e.g., LiteLLM with Gemini)
                response_format = {'type': 'json_object'}
                if response_model is not None:
                    # Append schema to last user message (like base class does)
                    serialized_model = json.dumps(response_model.model_json_schema())
                    for i in range(len(openai_messages) - 1, -1, -1):
                        if openai_messages[i]['role'] == 'user':
                            content = openai_messages[i].get('content', '')
                            openai_messages[i]['content'] = (
                                f'{content}\n\nRespond with a JSON object in the following '
                                f'format:\n\n{serialized_model}'
                            )
                            break

            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,  # type: ignore[arg-type]
            )
            result = response.choices[0].message.content or ''
            return json.loads(result)
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
                'llm.provider': 'openai_generic',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
                'structured_output.mode': 'json_object'
                if self._use_json_object_mode
                else 'json_schema',
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            retry_count = 0
            last_error = None
            # Track if we've already attempted fallback in this call
            fallback_attempted_this_call = False

            while retry_count <= self.MAX_RETRIES:
                try:
                    response = await self._generate_response(
                        messages, response_model, max_tokens=max_tokens, model_size=model_size
                    )

                    # Check for schema-as-data pattern (only if using json_schema mode)
                    if (
                        response_model is not None
                        and not self._use_json_object_mode
                        and self._is_schema_returned_as_data(response)
                    ):
                        if not fallback_attempted_this_call:
                            logger.warning(
                                'Provider returned schema definition instead of data. '
                                'Switching to json_object mode with embedded schema.'
                            )
                            self._use_json_object_mode = True
                            fallback_attempted_this_call = True
                            span.add_attributes({'structured_output.fallback_triggered': True})
                            # Retry immediately with fallback mode (does NOT count against MAX_RETRIES)
                            continue
                        else:
                            # Fallback already attempted but still got schema - treat as error
                            raise ValueError(
                                'Provider returned schema definition even in fallback mode'
                            )

                    return response
                except (RateLimitError, RefusalError):
                    # These errors should not trigger retries
                    span.set_status('error', str(last_error))
                    raise
                except (
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ):
                    # Let OpenAI's client handle these retries
                    span.set_status('error', str(last_error))
                    raise
                except Exception as e:
                    last_error = e

                    # Don't retry if we've hit the max retries
                    if retry_count >= self.MAX_RETRIES:
                        logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                        span.set_status('error', str(e))
                        span.record_exception(e)
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
            span.set_status('error', str(last_error))
            raise last_error or Exception('Max retries exceeded with no specific error')
