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

import typing
import json
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .openai_base_client import DEFAULT_REASONING, DEFAULT_VERBOSITY, BaseOpenAIClient


class OpenAIClient(BaseOpenAIClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the BaseOpenAIClient and provides OpenAI-specific implementation
    for creating completions.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str = DEFAULT_REASONING,
        verbosity: str = DEFAULT_VERBOSITY,
    ):
        """
        Initialize the OpenAIClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
        """
        super().__init__(config, cache, max_tokens, reasoning, verbosity)

        if config is None:
            config = LLMConfig()

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a structured completion.

        Prefer the Responses API with beta parse when available; otherwise fall back to
        Chat Completions with JSON schema (json mode) compatible with providers like SiliconFlow.
        """
        # Reasoning models (gpt-5/o1/o3 family) often don't support temperature
        is_reasoning_model = model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')

        try:
            # Primary path: use OpenAI Responses API with structured parsing
            response = await self.client.responses.parse(
                model=model,
                input=messages,  # type: ignore
                temperature=temperature if not is_reasoning_model else None,
                max_output_tokens=max_tokens,
                text_format=response_model,  # type: ignore
                reasoning={'effort': reasoning} if reasoning is not None else None,  # type: ignore
                text={'verbosity': verbosity} if verbosity is not None else None,  # type: ignore
            )
            return response
        except Exception as e:
            # Fallback path: use chat.completions with JSON schema when /v1/responses isn't supported
            # Only fall back for clear non-support cases (e.g., 404 NotFound) or attribute issues
            should_fallback = isinstance(e, (openai.NotFoundError, AttributeError))
            if not should_fallback:
                # Some SDKs may wrap errors differently; be permissive if message hints 404/unknown endpoint
                msg = str(e).lower()
                if '404' in msg or 'not found' in msg or 'responses' in msg:
                    should_fallback = True

            if not should_fallback:
                raise

            # Build JSON schema from the Pydantic model (Pydantic v2 preferred)
            try:
                json_schema = response_model.model_json_schema()
            except Exception:
                # Pydantic v1 compatibility
                json_schema = response_model.schema()  # type: ignore[attr-defined]

            # Some providers require a schema name; use model class name by default
            schema_name = getattr(response_model, '__name__', 'structured_response')

            print(f'Falling back to chat.completions with JSON schema for model {model}...')
            completion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature if not is_reasoning_model else None,
                max_tokens=max_tokens,
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        'name': schema_name,
                        'schema': json_schema,
                    },
                },
            )

            content = completion.choices[0].message.content if completion.choices else None
            output_text = content if content is not None else '{}'

            # Ensure return a JSON string; serialize dict-like outputs defensively
            if not isinstance(output_text, str):
                try:
                    output_text = json.dumps(output_text)
                except Exception:
                    output_text = '{}'

            class _SimpleResponse:
                def __init__(self, text: str):
                    self.output_text = text

            return _SimpleResponse(output_text)

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a regular completion with JSON format."""
        # Reasoning models (gpt-5 family) don't support temperature
        is_reasoning_model = model.startswith('gpt-5') or model.startswith('o1') or model.startswith('o3')

        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature if not is_reasoning_model else None,
            max_tokens=max_tokens,
            response_format={'type': 'json_object'},
        )
