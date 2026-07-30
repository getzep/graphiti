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

# Moonshot AI (Kimi) exposes an OpenAI-compatible API. https://platform.moonshot.ai
# Use the mainland endpoint https://api.moonshot.cn/v1 by setting config.base_url.
DEFAULT_KIMI_BASE_URL = 'https://api.moonshot.ai/v1'
DEFAULT_KIMI_MODEL = 'moonshot-v1-8k'


class KimiClient(OpenAIGenericClient):
    """LLM client for `Kimi / Moonshot AI <https://platform.moonshot.ai>`_.

    Moonshot speaks the OpenAI ``/chat/completions`` protocol, so this is a thin
    preset over :class:`OpenAIGenericClient` filling in Moonshot's endpoint and
    default model; set ``config.api_key`` to your Moonshot key. Defaults to the
    ``json_object`` structured-output fallback for broad compatibility.
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
            config.base_url = DEFAULT_KIMI_BASE_URL
        if not config.model:
            config.model = DEFAULT_KIMI_MODEL

        super().__init__(
            config=config,
            cache=cache,
            client=client,
            max_tokens=max_tokens,
            structured_output_mode=structured_output_mode,
        )
