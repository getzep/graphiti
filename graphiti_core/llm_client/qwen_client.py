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

# Qwen in the cloud via Alibaba DashScope's OpenAI-compatible endpoint.
# International: dashscope-intl; mainland China: https://dashscope.aliyuncs.com/compatible-mode/v1
DEFAULT_QWEN_BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
DEFAULT_QWEN_MODEL = 'qwen-plus'


class QwenClient(OpenAIGenericClient):
    """LLM client for `Qwen <https://www.alibabacloud.com/help/en/model-studio>`_.

    Qwen is OpenAI-compatible both in the cloud (Alibaba DashScope) and locally
    (served via Ollama or vLLM), so this is a thin preset over
    :class:`OpenAIGenericClient`.

    * **Cloud (default):** DashScope's ``compatible-mode/v1`` endpoint; set
      ``config.api_key`` to your DashScope key.
    * **Local:** point ``config.base_url`` at your local server, e.g.
      ``http://localhost:11434/v1`` (Ollama) with ``config.model='qwen2.5'``; the
      placeholder api_key requirement is the same as any local OpenAI-compatible
      endpoint.

    Defaults to the ``json_object`` structured-output fallback, which is the safe
    choice for both the cloud API and less-reliable local models.
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
            config.base_url = DEFAULT_QWEN_BASE_URL
        if not config.model:
            config.model = DEFAULT_QWEN_MODEL

        super().__init__(
            config=config,
            cache=cache,
            client=client,
            max_tokens=max_tokens,
            structured_output_mode=structured_output_mode,
        )
