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

from enum import Enum
from typing import Any

DEFAULT_MAX_TOKENS = 16384
DEFAULT_TEMPERATURE = 1


class ModelSize(Enum):
    small = 'small'
    medium = 'medium'


class LLMConfig:
    """
    Configuration class for the Language Learning Model (LLM).

    This class encapsulates the necessary parameters to interact with an LLM API,
    such as OpenAI's GPT models. It stores the API key, model name, and base URL
    for making requests to the LLM service.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        small_model: str | None = None,
        timeout: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        """
        Initialize the LLMConfig with the provided parameters.

        Args:
                api_key (str): The authentication key for accessing the LLM API.
                                                This is required for making authorized requests.

                model (str, optional): The specific LLM model to use for generating responses.
                                                                Defaults to "gpt-5.5".

                base_url (str, optional): The base URL of the LLM API service.
                                                                        Defaults to "https://api.openai.com", which is OpenAI's standard API endpoint.
                                                                        This can be changed if using a different provider or a custom endpoint.

                small_model (str, optional): The specific LLM model to use for generating responses of simpler prompts.
                                                                Defaults to "gpt-4.1-nano".

                timeout (float, optional): Per-request timeout in seconds for providers that support it.
                    Defaults to ``None``, which leaves the provider SDK default unchanged.

                extra_body (dict, optional): Provider-specific request-body fields for OpenAI-compatible
                    endpoints. Support varies by server; unsupported fields may be ignored. Defaults to ``None``.
        """
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError('timeout must be a float or None')
            if timeout < 0:
                raise ValueError('timeout must be greater than or equal to zero')

        if extra_body is not None:
            if not isinstance(extra_body, dict):
                raise TypeError('extra_body must be a dict or None')
            if not all(isinstance(key, str) for key in extra_body):
                raise TypeError('extra_body keys must be strings')

        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.small_model = small_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = float(timeout) if timeout is not None else None
        self.extra_body = extra_body
