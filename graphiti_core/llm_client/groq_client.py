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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import groq
    from groq import AsyncGroq
    from groq.types.chat import ChatCompletionMessageParam
else:
    try:
        import groq
        from groq import AsyncGroq
        from groq.types.chat import ChatCompletionMessageParam
    except ImportError:
        raise ImportError(
            'groq is required for GroqClient. Install it with: pip install graphiti-core[groq]'
        ) from None
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig, ModelSize
from .errors import RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'llama-3.1-70b-versatile'
DEFAULT_MAX_TOKENS = 2048


class GroqClient(LLMClient):
    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        if config is None:
            config = LLMConfig(max_tokens=DEFAULT_MAX_TOKENS)
        elif config.max_tokens is None:
            config.max_tokens = DEFAULT_MAX_TOKENS
        super().__init__(config, cache)

        self.client = AsyncGroq(api_key=config.api_key)

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
        model: str | None = None,
    ) -> dict[str, typing.Any]:
        # Resolve the model before the base class builds its response-cache key.
        if model is None and model_size == ModelSize.small:
            model = self.small_model
        return await super().generate_response(
            messages,
            response_model,
            max_tokens,
            model_size,
            group_id,
            prompt_name,
            attribute_extraction=attribute_extraction,
            model=model,
        )

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
    ) -> dict[str, typing.Any]:
        msgs: list[ChatCompletionMessageParam] = []
        for m in messages:
            if m.role == 'user':
                msgs.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                msgs.append({'role': 'system', 'content': m.content})
        try:
            response = await self.client.chat.completions.create(
                model=model or self.model or DEFAULT_MODEL,
                messages=msgs,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format={'type': 'json_object'},
            )
            result = response.choices[0].message.content or ''
            return json.loads(result)
        except groq.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise
