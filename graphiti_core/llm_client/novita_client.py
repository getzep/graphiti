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

import os

from .config import LLMConfig
from .openai_generic_client import OpenAIGenericClient

DEFAULT_MODEL = 'moonshotai/kimi-k2.5'
DEFAULT_MAX_TOKENS = 16384
DEFAULT_BASE_URL = 'https://api.novita.ai/openai'


class NovitaClient(OpenAIGenericClient):
    """
    Novita AI LLM Client.

    This client provides access to Novita AI's LLM models through their
    OpenAI-compatible API endpoint. It extends OpenAIGenericClient for
    full compatibility with structured outputs and function calling.

    Novita AI offers cost-effective access to various LLM models including:
    - moonshotai/kimi-k2.5 (default) - MoE model with function calling,
      structured output, reasoning, and vision support
    - zai-org/glm-5 - MoE model with function calling and structured output
    - minimax/minimax-m2.5 - MoE model with function calling and structured output

    Attributes:
        client: The AsyncOpenAI client configured for Novita AI.
        model: The Novita AI model name to use.
        temperature: The temperature for response generation.
        max_tokens: The maximum tokens for responses.

    Example:
        ```python
        from graphiti_core.llm_client.novita_client import NovitaClient
        from graphiti_core.llm_client.config import LLMConfig

        # Using environment variable NOVITA_API_KEY
        client = NovitaClient()

        # Or with explicit API key
        client = NovitaClient(
            config=LLMConfig(
                api_key='your-novita-api-key',
                model='moonshotai/kimi-k2.5',
            )
        )
        ```
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the Novita AI LLM client.

        Args:
            config: LLM configuration including api_key, model, etc.
                    If not provided, defaults are used.
            cache: Whether to enable response caching. Defaults to False.
            max_tokens: Maximum tokens for responses. Defaults to 16384.
        """
        if config is None:
            config = LLMConfig()

        # Set Novita-specific defaults if not provided
        if config.api_key is None:
            config.api_key = os.environ.get('NOVITA_API_KEY')

        if config.model is None:
            config.model = DEFAULT_MODEL

        if config.base_url is None:
            config.base_url = DEFAULT_BASE_URL

        if config.max_tokens is None:
            config.max_tokens = max_tokens

        super().__init__(config=config, cache=cache, max_tokens=max_tokens)
