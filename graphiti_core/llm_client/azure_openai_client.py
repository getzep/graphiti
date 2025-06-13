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
from typing import Any

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig, ModelSize

logger = logging.getLogger(__name__)


class AzureOpenAILLMClient(LLMClient):
    """Wrapper class for AsyncAzureOpenAI that implements the LLMClient interface."""

    def __init__(self, azure_client: AsyncAzureOpenAI, config: LLMConfig | None = None):
        super().__init__(config, cache=False)
        self.azure_client = azure_client

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 1024,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """Generate response using Azure OpenAI client."""
        # Convert messages to OpenAI format
        openai_messages: list[ChatCompletionMessageParam] = []
        for message in messages:
            message.content = self._clean_input(message.content)
            if message.role == 'user':
                openai_messages.append({'role': 'user', 'content': message.content})
            elif message.role == 'system':
                openai_messages.append({'role': 'system', 'content': message.content})

        # Ensure model is a string
        model_name = self.model if self.model else 'gpt-4o-mini'

        try:
            response = await self.azure_client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
                temperature=float(self.temperature) if self.temperature is not None else 0.7,
                max_tokens=max_tokens,
                response_format={'type': 'json_object'},
            )
            result = response.choices[0].message.content or '{}'

            # Parse JSON response
            return json.loads(result)
        except Exception as e:
            logger.error(f'Error in Azure OpenAI LLM response: {e}')
            raise
