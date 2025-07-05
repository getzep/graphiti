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
import os
import typing
from json import JSONDecodeError
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ValidationError

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError

if TYPE_CHECKING:
    import anthropic
    from anthropic import AsyncAnthropic
    from anthropic.types import MessageParam, ToolChoiceParam, ToolUnionParam
else:
    try:
        import anthropic
        from anthropic import AsyncAnthropic
        from anthropic.types import MessageParam, ToolChoiceParam, ToolUnionParam
    except ImportError:
        raise ImportError(
            'anthropic is required for AnthropicClient. '
            'Install it with: pip install graphiti-core[anthropic]'
        ) from None


logger = logging.getLogger(__name__)

AnthropicModel = Literal[
    'claude-3-7-sonnet-latest',
    'claude-3-7-sonnet-20250219',
    'claude-3-5-haiku-latest',
    'claude-3-5-haiku-20241022',
    'claude-3-5-sonnet-latest',
    'claude-3-5-sonnet-20241022',
    'claude-3-5-sonnet-20240620',
    'claude-3-opus-latest',
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307',
    'claude-2.1',
    'claude-2.0',
]

DEFAULT_MODEL: AnthropicModel = 'claude-3-7-sonnet-latest'


class AnthropicClient(LLMClient):
    """
    A client for the Anthropic LLM.

    Args:
        config: A configuration object for the LLM.
        cache: Whether to cache the LLM responses.
        client: An optional client instance to use.
        max_tokens: The maximum number of tokens to generate.

    Methods:
        generate_response: Generate a response from the LLM.

    Notes:
        - If a LLMConfig is not provided, api_key will be pulled from the ANTHROPIC_API_KEY environment
            variable, and all default values will be used for the LLMConfig.

    """

    model: AnthropicModel

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: AsyncAnthropic | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if config is None:
            config = LLMConfig()
            config.api_key = os.getenv('ANTHROPIC_API_KEY')
            config.max_tokens = max_tokens

        if config.model is None:
            config.model = DEFAULT_MODEL

        super().__init__(config, cache)
        # Explicitly set the instance model to the config model to prevent type checking errors
        self.model = typing.cast(AnthropicModel, config.model)

        if not client:
            self.client = AsyncAnthropic(
                api_key=config.api_key,
                max_retries=1,
            )
        else:
            self.client = client

    def _extract_json_from_text(self, text: str) -> dict[str, typing.Any]:
        """Extract JSON from text content.

        A helper method to extract JSON from text content, used when tool use fails or
        no response_model is provided.

        Args:
            text: The text to extract JSON from

        Returns:
            Extracted JSON as a dictionary

        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError(f'Could not extract JSON from model response: {text}')
        except (JSONDecodeError, ValueError) as e:
            raise ValueError(f'Could not extract JSON from model response: {text}') from e

    def _create_tool(
        self, response_model: type[BaseModel] | None = None
    ) -> tuple[list[ToolUnionParam], ToolChoiceParam]:
        """
        Create a tool definition based on the response_model if provided, or a generic JSON tool if not.

        Args:
            response_model: Optional Pydantic model to use for structured output.

        Returns:
            A list containing a single tool definition for use with the Anthropic API.
        """
        if response_model is not None:
            # Use the response_model to define the tool
            model_schema = response_model.model_json_schema()
            tool_name = response_model.__name__
            description = model_schema.get('description', f'Extract {tool_name} information')
        else:
            # Create a generic JSON output tool
            tool_name = 'generic_json_output'
            description = 'Output data in JSON format'
            model_schema = {
                'type': 'object',
                'additionalProperties': True,
                'description': 'Any JSON object containing the requested information',
            }

        tool = {
            'name': tool_name,
            'description': description,
            'input_schema': model_schema,
        }
        tool_list = [tool]
        tool_list_cast = typing.cast(list[ToolUnionParam], tool_list)
        tool_choice = {'type': 'tool', 'name': tool_name}
        tool_choice_cast = typing.cast(ToolChoiceParam, tool_choice)
        return tool_list_cast, tool_choice_cast

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Anthropic LLM using tool-based approach for all requests.

        Args:
            messages: List of message objects to send to the LLM.
            response_model: Optional Pydantic model to use for structured output.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            Dictionary containing the structured response from the LLM.

        Raises:
            RateLimitError: If the rate limit is exceeded.
            RefusalError: If the LLM refuses to respond.
            Exception: If an error occurs during the generation process.
        """
        system_message = messages[0]
        user_messages = [{'role': m.role, 'content': m.content} for m in messages[1:]]
        user_messages_cast = typing.cast(list[MessageParam], user_messages)

        # TODO: Replace hacky min finding solution after fixing hardcoded EXTRACT_EDGES_MAX_TOKENS = 16384 in
        # edge_operations.py. Throws errors with cheaper models that lower max_tokens.
        max_creation_tokens: int = min(
            max_tokens if max_tokens is not None else self.config.max_tokens,
            DEFAULT_MAX_TOKENS,
        )

        try:
            # Create the appropriate tool based on whether response_model is provided
            tools, tool_choice = self._create_tool(response_model)
            result = await self.client.messages.create(
                system=system_message.content,
                max_tokens=max_creation_tokens,
                temperature=self.temperature,
                messages=user_messages_cast,
                model=self.model,
                tools=tools,
                tool_choice=tool_choice,
            )

            # Extract the tool output from the response
            for content_item in result.content:
                if content_item.type == 'tool_use':
                    if isinstance(content_item.input, dict):
                        tool_args: dict[str, typing.Any] = content_item.input
                    else:
                        tool_args = json.loads(str(content_item.input))
                    return tool_args

            # If we didn't get a proper tool_use response, try to extract from text
            for content_item in result.content:
                if content_item.type == 'text':
                    return self._extract_json_from_text(content_item.text)
                else:
                    raise ValueError(
                        f'Could not extract structured data from model response: {result.content}'
                    )

            # If we get here, we couldn't parse a structured response
            raise ValueError(
                f'Could not extract structured data from model response: {result.content}'
            )

        except anthropic.RateLimitError as e:
            raise RateLimitError(f'Rate limit exceeded. Please try again later. Error: {e}') from e
        except anthropic.APIError as e:
            # Special case for content policy violations. We convert these to RefusalError
            # to bypass the retry mechanism, as retrying policy-violating content will always fail.
            # This avoids wasting API calls and provides more specific error messaging to the user.
            if 'refused to respond' in str(e).lower():
                raise RefusalError(str(e)) from e
            raise e
        except Exception as e:
            raise e

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message objects to send to the LLM.
            response_model: Optional Pydantic model to use for structured output.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            Dictionary containing the structured response from the LLM.

        Raises:
            RateLimitError: If the rate limit is exceeded.
            RefusalError: If the LLM refuses to respond.
            Exception: If an error occurs during the generation process.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        max_retries = 2
        last_error: Exception | None = None

        while retry_count <= max_retries:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens, model_size
                )

                # If we have a response_model, attempt to validate the response
                if response_model is not None:
                    # Validate the response against the response_model
                    model_instance = response_model(**response)
                    return model_instance.model_dump()

                # If no validation needed, return the response
                return response

            except (RateLimitError, RefusalError):
                # These errors should not trigger retries
                raise
            except Exception as e:
                last_error = e

                if retry_count >= max_retries:
                    if isinstance(e, ValidationError):
                        logger.error(
                            f'Validation error after {retry_count}/{max_retries} attempts: {e}'
                        )
                    else:
                        logger.error(f'Max retries ({max_retries}) exceeded. Last error: {e}')
                    raise e

                if isinstance(e, ValidationError):
                    response_model_cast = typing.cast(type[BaseModel], response_model)
                    error_context = f'The previous response was invalid. Please provide a valid {response_model_cast.__name__} object. Error: {e}'
                else:
                    error_context = (
                        f'The previous response attempt was invalid. '
                        f'Error type: {e.__class__.__name__}. '
                        f'Error details: {str(e)}. '
                        f'Please try again with a valid response.'
                    )

                # Common retry logic
                retry_count += 1
                messages.append(Message(role='user', content=error_context))
                logger.warning(f'Retrying after error (attempt {retry_count}/{max_retries}): {e}')

        # If we somehow get here, raise the last error
        raise last_error or Exception('Max retries exceeded with no specific error')
