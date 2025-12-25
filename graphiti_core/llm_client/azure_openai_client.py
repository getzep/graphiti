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

import logging
from typing import ClassVar

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
        """Create a structured completion using Azure OpenAI's responses.parse API."""
        supports_reasoning = self._supports_reasoning_features(model)
        request_kwargs = {
            'model': model,
            'input': messages,
            'max_output_tokens': max_tokens,
            'text_format': response_model,  # type: ignore
        }

        temperature_value = temperature if not supports_reasoning else None
        if temperature_value is not None:
            request_kwargs['temperature'] = temperature_value

        if supports_reasoning and reasoning:
            request_kwargs['reasoning'] = {'effort': reasoning}  # type: ignore

        if supports_reasoning and verbosity:
            request_kwargs['text'] = {'verbosity': verbosity}  # type: ignore

        return await self.client.responses.parse(**request_kwargs)

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
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

    @staticmethod
    def _supports_reasoning_features(model: str) -> bool:
        """Return True when the Azure model supports reasoning/verbosity options."""
        reasoning_prefixes = ('o1', 'o3', 'gpt-5')
        return model.startswith(reasoning_prefixes)
