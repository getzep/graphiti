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
import typing

from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

logger = logging.getLogger(__name__)

try:
    import litellm  # type: ignore
    from litellm import acompletion  # type: ignore

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning('LiteLLM not available. Install with: pip install graphiti-core[litellm]')


class LiteLLMClient(LLMClient):
    """LLM client using LiteLLM for unified multi-provider support.

    LiteLLM provides a unified interface to 100+ LLM providers including:
    - OpenAI, Azure OpenAI
    - Anthropic
    - Google (Gemini, Vertex AI)
    - AWS Bedrock
    - Cohere, Replicate, HuggingFace
    - Local models (Ollama, vLLM, LocalAI)
    - And many more

    Examples:
        >>> # OpenAI via LiteLLM
        >>> client = LiteLLMClient(
        ...     LLMConfig(
        ...         model='gpt-4.1-mini',
        ...         api_key='sk-...',
        ...     )
        ... )

        >>> # Azure OpenAI
        >>> client = LiteLLMClient(
        ...     LLMConfig(
        ...         model='azure/gpt-4-deployment-name',
        ...         base_url='https://your-resource.openai.azure.com',
        ...         api_key='...',
        ...     )
        ... )

        >>> # AWS Bedrock
        >>> client = LiteLLMClient(
        ...     LLMConfig(
        ...         model='bedrock/anthropic.claude-3-sonnet-20240229-v1:0',
        ...     )
        ... )

        >>> # Ollama (local)
        >>> client = LiteLLMClient(
        ...     LLMConfig(
        ...         model='ollama/llama2',
        ...         base_url='http://localhost:11434',
        ...     )
        ... )
    """

    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        """Initialize LiteLLM client.

        Args:
            config: LLM configuration. Model name should follow LiteLLM conventions.
            cache: Whether to enable response caching.

        Raises:
            ImportError: If LiteLLM is not installed.
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                'LiteLLM is required for LiteLLMClient. '
                'Install with: pip install graphiti-core[litellm]'
            )

        super().__init__(config, cache)

        # Configure LiteLLM
        if self.config.base_url:
            litellm.api_base = self.config.base_url

        if self.config.api_key:
            litellm.api_key = self.config.api_key

        # Disable verbose logging by default
        litellm.suppress_debug_info = True

        logger.info(f'Initialized LiteLLM client with model: {self.model}')

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """Generate a response using LiteLLM.

        Args:
            messages: List of conversation messages
            response_model: Optional Pydantic model for structured output
            max_tokens: Maximum tokens in response
            model_size: Size of model to use (medium or small)

        Returns:
            Dictionary containing the response data

        Raises:
            RateLimitError: If rate limit is exceeded
            Exception: For other errors from the LLM provider
        """
        # Select model based on size
        model = self.model if model_size == ModelSize.medium else self.small_model

        if not model:
            raise ValueError('Model must be specified for LiteLLM client')

        # Convert messages to LiteLLM format
        litellm_messages = [
            {'role': msg.role, 'content': self._clean_input(msg.content)} for msg in messages
        ]

        try:
            # Check if provider supports structured output
            supports_structured = self._supports_structured_output(model)

            if response_model and supports_structured:
                # Use LiteLLM's structured output support
                with self.tracer.start_span('litellm_completion') as span:
                    span.add_attributes(
                        {
                            'model': model,
                            'structured_output': True,
                            'max_tokens': max_tokens,
                        }
                    )

                    response = await acompletion(
                        model=model,
                        messages=litellm_messages,
                        temperature=self.temperature,
                        max_tokens=max_tokens,
                        response_format={'type': 'json_object'},
                    )

                    # Parse JSON response into Pydantic model
                    content = response.choices[0].message.content
                    import json

                    result = json.loads(content)

                    # Validate with response model
                    if response_model:
                        validated = response_model(**result)
                        return validated.model_dump()

                    return result

            elif response_model:
                # Fallback: Use OpenAI-style function calling or prompt engineering
                with self.tracer.start_span('litellm_completion_json') as span:
                    span.add_attributes(
                        {
                            'model': model,
                            'structured_output': False,
                            'max_tokens': max_tokens,
                        }
                    )

                    # Add JSON schema to the last message
                    schema_str = response_model.model_json_schema()
                    litellm_messages[-1]['content'] += (
                        f'\n\nRespond with valid JSON matching this schema: {schema_str}'
                    )

                    response = await acompletion(
                        model=model,
                        messages=litellm_messages,
                        temperature=self.temperature,
                        max_tokens=max_tokens,
                    )

                    content = response.choices[0].message.content
                    import json

                    # Try to parse JSON from response
                    result = json.loads(content)
                    validated = response_model(**result)
                    return validated.model_dump()

            else:
                # Regular completion without structured output
                with self.tracer.start_span('litellm_completion_text') as span:
                    span.add_attributes(
                        {
                            'model': model,
                            'max_tokens': max_tokens,
                        }
                    )

                    response = await acompletion(
                        model=model,
                        messages=litellm_messages,
                        temperature=self.temperature,
                        max_tokens=max_tokens,
                    )

                    return {'content': response.choices[0].message.content}

        except Exception as e:
            error_str = str(e).lower()

            # Check for rate limiting
            if 'rate limit' in error_str or 'quota' in error_str or '429' in error_str:
                raise RateLimitError(f'Rate limit exceeded for model {model}: {e}') from e

            # Re-raise other exceptions
            logger.error(f'Error generating response with LiteLLM: {e}')
            raise

    def _supports_structured_output(self, model: str) -> bool:
        """Check if a model supports structured JSON output.

        Args:
            model: Model identifier (e.g., "gpt-4", "azure/gpt-4", "bedrock/claude-3")

        Returns:
            True if the model supports structured output, False otherwise
        """
        # Extract base model name from LiteLLM format
        model_lower = model.lower()

        # OpenAI models with structured output support
        if any(x in model_lower for x in ['gpt-4', 'gpt-3.5', 'gpt-4.1', 'gpt-5', 'o1', 'o3']):
            return True

        # Gemini models support JSON mode
        if 'gemini' in model_lower:
            return True

        # Claude 3+ models support JSON mode
        return 'claude-3' in model_lower or 'claude-4' in model_lower
