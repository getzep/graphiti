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
import re
import typing
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from ..prompts.models import Message
from .client import MULTILINGUAL_EXTRACTION_RESPONSES, LLMClient
from .config import LLMConfig, ModelSize
from .errors import RateLimitError

if TYPE_CHECKING:
    from google import genai
    from google.genai import types
else:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        # If gemini client is not installed, raise an ImportError
        raise ImportError(
            'google-genai is required for GeminiClient. '
            'Install it with: pip install graphiti-core[google-genai]'
        ) from None


logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gemini-2.5-flash'
DEFAULT_SMALL_MODEL = 'gemini-2.5-flash-lite-preview-06-17'

# Maximum output tokens for different Gemini models
GEMINI_MODEL_MAX_TOKENS = {
    # Gemini 2.5 models
    'gemini-2.5-pro': 65536,
    'gemini-2.5-flash': 65536,
    'gemini-2.5-flash-lite': 64000,
    'models/gemini-2.5-flash-lite-preview-06-17': 64000,
    # Gemini 2.0 models
    'gemini-2.0-flash': 8192,
    'gemini-2.0-flash-lite': 8192,
    # Gemini 1.5 models
    'gemini-1.5-pro': 8192,
    'gemini-1.5-flash': 8192,
    'gemini-1.5-flash-8b': 8192,
}

# Default max tokens for models not in the mapping
DEFAULT_GEMINI_MAX_TOKENS = 8192


class GeminiClient(LLMClient):
    """
    GeminiClient is a client class for interacting with Google's Gemini language models.

    This class extends the LLMClient and provides methods to initialize the client
    and generate responses from the Gemini language model.

    Attributes:
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.
        thinking_config (types.ThinkingConfig | None): Optional thinking configuration for models that support it.
    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False, thinking_config: types.ThinkingConfig | None = None):
            Initializes the GeminiClient with the provided configuration, cache setting, and optional thinking config.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int | None = None,
        thinking_config: types.ThinkingConfig | None = None,
        client: 'genai.Client | None' = None,
    ):
        """
        Initialize the GeminiClient with the provided configuration, cache setting, and optional thinking config.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            thinking_config (types.ThinkingConfig | None): Optional thinking configuration for models that support it.
                Only use with models that support thinking (gemini-2.5+). Defaults to None.
            client (genai.Client | None): An optional async client instance to use. If not provided, a new genai.Client is created.
        """
        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        self.model = config.model

        if client is None:
            self.client = genai.Client(api_key=config.api_key)
        else:
            self.client = client

        self.max_tokens = max_tokens
        self.thinking_config = thinking_config

    def _check_safety_blocks(self, response) -> None:
        """Check if response was blocked for safety reasons and raise appropriate exceptions."""
        # Check if the response was blocked for safety reasons
        if not (hasattr(response, 'candidates') and response.candidates):
            return

        candidate = response.candidates[0]
        if not (hasattr(candidate, 'finish_reason') and candidate.finish_reason == 'SAFETY'):
            return

        # Content was blocked for safety reasons - collect safety details
        safety_info = []
        safety_ratings = getattr(candidate, 'safety_ratings', None)

        if safety_ratings:
            for rating in safety_ratings:
                if getattr(rating, 'blocked', False):
                    category = getattr(rating, 'category', 'Unknown')
                    probability = getattr(rating, 'probability', 'Unknown')
                    safety_info.append(f'{category}: {probability}')

        safety_details = (
            ', '.join(safety_info) if safety_info else 'Content blocked for safety reasons'
        )
        raise Exception(f'Response blocked by Gemini safety filters: {safety_details}')

    def _check_prompt_blocks(self, response) -> None:
        """Check if prompt was blocked and raise appropriate exceptions."""
        prompt_feedback = getattr(response, 'prompt_feedback', None)
        if not prompt_feedback:
            return

        block_reason = getattr(prompt_feedback, 'block_reason', None)
        if block_reason:
            raise Exception(f'Prompt blocked by Gemini: {block_reason}')

    def _get_model_for_size(self, model_size: ModelSize) -> str:
        """Get the appropriate model name based on the requested size."""
        if model_size == ModelSize.small:
            return self.small_model or DEFAULT_SMALL_MODEL
        else:
            return self.model or DEFAULT_MODEL

    def _get_max_tokens_for_model(self, model: str) -> int:
        """Get the maximum output tokens for a specific Gemini model."""
        return GEMINI_MODEL_MAX_TOKENS.get(model, DEFAULT_GEMINI_MAX_TOKENS)

    def _resolve_max_tokens(self, requested_max_tokens: int | None, model: str) -> int:
        """
        Resolve the maximum output tokens to use based on precedence rules.

        Precedence order (highest to lowest):
        1. Explicit max_tokens parameter passed to generate_response()
        2. Instance max_tokens set during client initialization
        3. Model-specific maximum tokens from GEMINI_MODEL_MAX_TOKENS mapping
        4. DEFAULT_MAX_TOKENS as final fallback

        Args:
            requested_max_tokens: The max_tokens parameter passed to generate_response()
            model: The model name to look up model-specific limits

        Returns:
            int: The resolved maximum tokens to use
        """
        # 1. Use explicit parameter if provided
        if requested_max_tokens is not None:
            return requested_max_tokens

        # 2. Use instance max_tokens if set during initialization
        if self.max_tokens is not None:
            return self.max_tokens

        # 3. Use model-specific maximum or return DEFAULT_GEMINI_MAX_TOKENS
        return self._get_max_tokens_for_model(model)

    def salvage_json(self, raw_output: str) -> dict[str, typing.Any] | None:
        """
        Attempt to salvage a JSON object if the raw output is truncated.

        This is accomplished by looking for the last closing bracket for an array or object.
        If found, it will try to load the JSON object from the raw output.
        If the JSON object is not valid, it will return None.

        Args:
            raw_output (str): The raw output from the LLM.

        Returns:
            dict[str, typing.Any]: The salvaged JSON object.
            None: If no salvage is possible.
        """
        if not raw_output:
            return None
        # Try to salvage a JSON array
        array_match = re.search(r'\]\s*$', raw_output)
        if array_match:
            try:
                return json.loads(raw_output[: array_match.end()])
            except Exception:
                pass
        # Try to salvage a JSON object
        obj_match = re.search(r'\}\s*$', raw_output)
        if obj_match:
            try:
                return json.loads(raw_output[: obj_match.end()])
            except Exception:
                pass
        return None

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Gemini language model.

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): An optional Pydantic model to parse the response into.
            max_tokens (int | None): The maximum number of tokens to generate in the response. If None, uses precedence rules.
            model_size (ModelSize): The size of the model to use (small or medium).

        Returns:
            dict[str, typing.Any]: The response from the language model.

        Raises:
            RateLimitError: If the API rate limit is exceeded.
            Exception: If there is an error generating the response or content is blocked.
        """
        try:
            gemini_messages: typing.Any = []
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

            # Get the appropriate model for the requested size
            model = self._get_model_for_size(model_size)

            # Resolve max_tokens using precedence rules (see _resolve_max_tokens for details)
            resolved_max_tokens = self._resolve_max_tokens(max_tokens, model)

            # Create generation config
            generation_config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=resolved_max_tokens,
                response_mime_type='application/json' if response_model else None,
                response_schema=response_model if response_model else None,
                system_instruction=system_prompt,
                thinking_config=self.thinking_config,
            )

            # Generate content using the simple string approach
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=gemini_messages,
                config=generation_config,
            )

            # Always capture the raw output for debugging
            raw_output = getattr(response, 'text', None)

            # Check for safety and prompt blocks
            self._check_safety_blocks(response)
            self._check_prompt_blocks(response)

            # If this was a structured output request, parse the response into the Pydantic model
            if response_model is not None:
                try:
                    if not raw_output:
                        raise ValueError('No response text')

                    validated_model = response_model.model_validate(json.loads(raw_output))

                    # Return as a dictionary for API consistency
                    return validated_model.model_dump()
                except Exception as e:
                    if raw_output:
                        logger.error(
                            '🦀 LLM generation failed parsing as JSON, will try to salvage.'
                        )
                        logger.error(self._get_failed_generation_log(gemini_messages, raw_output))
                        # Try to salvage
                        salvaged = self.salvage_json(raw_output)
                        if salvaged is not None:
                            logger.warning('Salvaged partial JSON from truncated/malformed output.')
                            return salvaged
                    raise Exception(f'Failed to parse structured response: {e}') from e

            # Otherwise, return the response text as a dictionary
            return {'content': raw_output}

        except Exception as e:
            # Check if it's a rate limit error based on Gemini API error codes
            error_message = str(e).lower()
            if (
                'rate limit' in error_message
                or 'quota' in error_message
                or 'resource_exhausted' in error_message
                or '429' in str(e)
            ):
                raise RateLimitError from e

            logger.error(f'Error in generating LLM response: {e}')
            raise Exception from e

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Gemini language model with retry logic and error handling.
        This method overrides the parent class method to provide a direct implementation with advanced retry logic.

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): An optional Pydantic model to parse the response into.
            max_tokens (int | None): The maximum number of tokens to generate in the response.
            model_size (ModelSize): The size of the model to use (small or medium).

        Returns:
            dict[str, typing.Any]: The response from the language model.
        """
        retry_count = 0
        last_error = None
        last_output = None

        # Add multilingual extraction instructions
        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        while retry_count < self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages=messages,
                    response_model=response_model,
                    max_tokens=max_tokens,
                    model_size=model_size,
                )
                last_output = (
                    response.get('content')
                    if isinstance(response, dict) and 'content' in response
                    else None
                )
                return response
            except RateLimitError as e:
                # Rate limit errors should not trigger retries (fail fast)
                raise e
            except Exception as e:
                last_error = e

                # Check if this is a safety block - these typically shouldn't be retried
                error_text = str(e) or (str(e.__cause__) if e.__cause__ else '')
                if 'safety' in error_text.lower() or 'blocked' in error_text.lower():
                    logger.warning(f'Content blocked by safety filters: {e}')
                    raise Exception(f'Content blocked by safety filters: {e}') from e

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

        # If we exit the loop without returning, all retries are exhausted
        logger.error('🦀 LLM generation failed and retries are exhausted.')
        logger.error(self._get_failed_generation_log(messages, last_output))
        logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {last_error}')
        raise last_error or Exception('Max retries exceeded')
