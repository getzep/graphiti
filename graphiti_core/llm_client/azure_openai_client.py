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
from typing import Any, ClassVar

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .openai_base_client import BaseOpenAIClient

logger = logging.getLogger(__name__)


class AzureOpenAILLMClient(BaseOpenAIClient):
    """Wrapper class for Azure OpenAI that implements the LLMClient interface.

    Supports both AsyncAzureOpenAI and AsyncOpenAI (with Azure v1 API endpoint).
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        azure_client: AsyncAzureOpenAI | AsyncOpenAI,
        config: LLMConfig | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        super().__init__(
            config,
            cache=False,
            max_tokens=max_tokens,
            reasoning=reasoning,
            verbosity=verbosity,
        )
        self.client = azure_client

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None,
        verbosity: str | None,
    ):
        """Create a structured completion using Azure OpenAI.

        For reasoning models (GPT-5, o1, o3): uses responses.parse API
        For regular models (GPT-4o, etc): uses chat.completions with response_format
        """
        supports_reasoning = self._supports_reasoning_features(model)

        if supports_reasoning:
            # Use responses.parse for reasoning models (o1, o3, gpt-5)
            request_kwargs = {
                'model': model,
                'input': messages,
                'max_output_tokens': max_tokens,
                'text_format': response_model,  # type: ignore
            }

            if reasoning:
                request_kwargs['reasoning'] = {'effort': reasoning}  # type: ignore

            if verbosity:
                request_kwargs['text'] = {'verbosity': verbosity}  # type: ignore

            return await self.client.responses.parse(**request_kwargs)
        else:
            # Use beta.chat.completions.parse for non-reasoning models (gpt-4o, etc.)
            # Azure's v1 compatibility endpoint doesn't fully support responses.parse
            # for non-reasoning models, so we use the structured output API instead
            request_kwargs = {
                'model': model,
                'messages': messages,
                'max_tokens': max_tokens,
                'response_format': response_model,  # Structured output
            }

            if temperature is not None:
                request_kwargs['temperature'] = temperature

            return await self.client.beta.chat.completions.parse(**request_kwargs)

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,  # noqa: ARG002 - inherited from abstract method
    ):
        """Create a regular completion with JSON format using Azure OpenAI."""
        supports_reasoning = self._supports_reasoning_features(model)

        request_kwargs = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'response_format': {'type': 'json_object'},
        }

        temperature_value = temperature if not supports_reasoning else None
        if temperature_value is not None:
            request_kwargs['temperature'] = temperature_value

        return await self.client.chat.completions.create(**request_kwargs)

    def _handle_structured_response(self, response: Any) -> dict[str, Any]:
        """Handle structured response parsing for both reasoning and non-reasoning models.

        For reasoning models (responses.parse): uses response.output_text
        For regular models (beta.chat.completions.parse): uses response.choices[0].message.parsed
        """
        # Check if this is a ParsedChatCompletion (from beta.chat.completions.parse)
        if hasattr(response, 'choices') and response.choices:
            # Standard ParsedChatCompletion format
            message = response.choices[0].message
            if hasattr(message, 'parsed') and message.parsed:
                # The parsed object is already a Pydantic model, convert to dict
                return message.parsed.model_dump()
            elif hasattr(message, 'refusal') and message.refusal:
                from graphiti_core.llm_client.errors import RefusalError

                raise RefusalError(message.refusal)
            else:
                raise Exception(f'Invalid response from LLM: {response.model_dump()}')
        elif hasattr(response, 'output_text'):
            # Reasoning model response format (responses.parse)
            response_object = response.output_text
            if response_object:
                return json.loads(response_object)
            elif hasattr(response, 'refusal') and response.refusal:
                from graphiti_core.llm_client.errors import RefusalError

                raise RefusalError(response.refusal)
            else:
                raise Exception(f'Invalid response from LLM: {response.model_dump()}')
        else:
            raise Exception(f'Unknown response format: {type(response)}')

    @staticmethod
    def _supports_reasoning_features(model: str) -> bool:
        """Return True when the Azure model supports reasoning/verbosity options."""
        reasoning_prefixes = ('o1', 'o3', 'gpt-5')
        return model.startswith(reasoning_prefixes)
