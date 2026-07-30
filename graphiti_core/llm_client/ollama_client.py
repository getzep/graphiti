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

# Ollama exposes an OpenAI-compatible API under /v1 on its local HTTP port.
DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434/v1'
# Ollama ignores the API key, but the AsyncOpenAI client requires a non-empty one.
DEFAULT_OLLAMA_API_KEY = 'ollama'


class OllamaClient(OpenAIGenericClient):
    """LLM client for a local `Ollama <https://ollama.com>`_ server.

    Ollama speaks the OpenAI ``/v1/chat/completions`` protocol, so this is a thin
    preset over :class:`OpenAIGenericClient` that fills in Ollama's local endpoint
    and placeholder credentials. Everything else — structured-output handling,
    markdown code-fence stripping, empty-response detection, and the tenacity
    retry wrapper — is inherited.

    Local models are unreliable at honoring ``json_schema`` constrained decoding,
    so this client defaults to the ``json_object`` structured-output fallback: the
    response schema is injected into the prompt and the inherited retry / fence
    handling recovers malformed output. Pass ``structured_output_mode='json_schema'``
    for Ollama builds and models that enforce native structured outputs.
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
            config.base_url = DEFAULT_OLLAMA_BASE_URL
        if not config.api_key:
            config.api_key = DEFAULT_OLLAMA_API_KEY

        super().__init__(
            config=config,
            cache=cache,
            client=client,
            max_tokens=max_tokens,
            structured_output_mode=structured_output_mode,
        )
