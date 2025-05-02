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

from google import genai  # type: ignore
from google.genai import types  # type: ignore
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gemini-2.0-flash'


class GeminiClient(LLMClient):
    """
    GeminiClient is a client class for interacting with Google's Gemini language models.

    This class extends the LLMClient and provides methods to initialize the client
    and generate responses from the Gemini language model.

    Attributes:
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.

    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False):
            Initializes the GeminiClient with the provided configuration and cache setting.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the GeminiClient with the provided configuration and cache setting.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
        """
        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        self.model = config.model
        # Configure the Gemini API
        self.client = genai.Client(
            api_key=config.api_key,
        )
        self.max_tokens = max_tokens

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Gemini language model.

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): An optional Pydantic model to parse the response into.
            max_tokens (int): The maximum number of tokens to generate in the response.

        Returns:
            dict[str, typing.Any]: The response from the language model.

        Raises:
            RateLimitError: If the API rate limit is exceeded.
            RefusalError: If the content is blocked by the model.
            Exception: If there is an error generating the response.
        """
        try:
            gemini_messages: list[types.Content] = []
            # If a response model is provided, add schema for structured output
            system_prompt = ''
            if response_model is not None:
                # Get the schema from the Pydantic model
                pydantic_schema = response_model.model_json_schema()

                # Create instruction to output in the desired JSON format
                system_prompt += (
                    f'Output ONLY valid JSON matching this schema: {json.dumps(pydantic_schema)}.\n'
                    'Do not include any explanatory text before or after the JSON.\n\n'
                )

            # Add messages content
            # First check for a system message
            if messages and messages[0].role == 'system':
                system_prompt = f'{messages[0].content}\n\n {system_prompt}'
                messages = messages[1:]

            # Add the rest of the messages
            for m in messages:
                m.content = self._clean_input(m.content)
                gemini_messages.append(
                    types.Content(role=m.role, parts=[types.Part.from_text(text=m.content)])
                )

            # Create generation config
            generation_config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=max_tokens or self.max_tokens,
                response_mime_type='application/json' if response_model else None,
                response_schema=response_model if response_model else None,
                system_instruction=system_prompt,
            )

            # Generate content using the simple string approach
            response = await self.client.aio.models.generate_content(
                model=self.model or DEFAULT_MODEL,
                contents=gemini_messages,
                config=generation_config,
            )

            # If this was a structured output request, parse the response into the Pydantic model
            if response_model is not None:
                try:
                    validated_model = response_model.model_validate(json.loads(response.text))

                    # Return as a dictionary for API consistency
                    return validated_model.model_dump()
                except Exception as e:
                    raise Exception(f'Failed to parse structured response: {e}') from e

            # Otherwise, return the response text as a dictionary
            return {'content': response.text}

        except Exception as e:
            # Check if it's a rate limit error
            if 'rate limit' in str(e).lower() or 'quota' in str(e).lower():
                raise RateLimitError from e
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Gemini language model.
        This method overrides the parent class method to provide a direct implementation.

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): An optional Pydantic model to parse the response into.
            max_tokens (int): The maximum number of tokens to generate in the response.

        Returns:
            dict[str, typing.Any]: The response from the language model.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Call the internal _generate_response method
        return await self._generate_response(
            messages=messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
        )
