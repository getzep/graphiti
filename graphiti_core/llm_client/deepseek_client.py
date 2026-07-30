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

from .config import LLMConfig
from .openai_generic_client import OpenAIGenericClient, StructuredOutputMode

# DeepSeek exposes an OpenAI-compatible API. https://api-docs.deepseek.com
DEFAULT_DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
DEFAULT_DEEPSEEK_MODEL = 'deepseek-chat'


class DeepSeekClient(OpenAIGenericClient):
    """LLM client for `DeepSeek <https://api-docs.deepseek.com>`_.

    DeepSeek speaks the OpenAI ``/chat/completions`` protocol, so this is a thin
    preset over :class:`OpenAIGenericClient` that fills in DeepSeek's endpoint and
    default model; set ``config.api_key`` to your DeepSeek key. DeepSeek supports
    ``json_object`` but not ``json_schema`` response formats, so this client
    defaults to the ``json_object`` fallback (schema injected into the prompt +
    inherited retry / code-fence recovery).
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = 16384,
        structured_output_mode: StructuredOutputMode = 'json_object',
    ):
        if config is None:
            config = LLMConfig()
        if not config.base_url:
            config.base_url = DEFAULT_DEEPSEEK_BASE_URL
        if not config.model:
            config.model = DEFAULT_DEEPSEEK_MODEL

        super().__init__(
            config=config,
            cache=cache,
            client=client,
            max_tokens=max_tokens,
            structured_output_mode=structured_output_mode,
        )
