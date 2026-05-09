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
    from ollama import AsyncClient
else:
    try:
        from ollama import AsyncClient
    except ImportError:
        raise ImportError(
            'ollama is required for OllamaClient. Install it with: pip install graphiti-core[ollama]'
        ) from None
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig, ModelSize
from .errors import RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'qwen3:4b'
DEFAULT_MAX_TOKENS = 8192


class OllamaClient(LLMClient):
    """Ollama async client wrapper for Graphiti.

    This client expects the `ollama` python package to be installed. It uses the
    AsyncClient.chat(...) API to generate chat responses. The response content
    is expected to be JSON which will be parsed and returned as a dict.
    """

    def __init__(self, config: LLMConfig | None = None, cache: bool = False, client: typing.Any | None = None):
        if config is None:
            config = LLMConfig(max_tokens=DEFAULT_MAX_TOKENS)
        elif config.max_tokens is None:
            config.max_tokens = DEFAULT_MAX_TOKENS
        super().__init__(config, cache)

        # Allow injecting a preconfigured AsyncClient for testing
        if client is None:
            # AsyncClient accepts host and other httpx args; pass api_key/base_url when available
            try:
                host = config.base_url.rstrip('/v1') if config.base_url else None
                self.client = AsyncClient(host=host)
            except TypeError as e:
                logger.warning(f"Error creating AsyncClient: {e}")
                self.client = AsyncClient()
        else:
            self.client = client

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        msgs: list[dict[str, str]] = []
        for m in messages:
            if m.role == 'user':
                msgs.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                msgs.append({'role': 'system', 'content': m.content})

        try:
            # Prepare options
            options: dict[str, typing.Any] = {}
            if max_tokens is not None:
                options['max_tokens'] = max_tokens
            if self.temperature is not None:
                options['temperature'] = self.temperature

            # If a response_model is provided, try to get its JSON schema for format
            schema = None
            if response_model is not None:
                try:
                    schema = response_model.model_json_schema()
                except Exception:
                    schema = None
            response = await self.client.chat(
                model=self.model or DEFAULT_MODEL,
                messages=msgs,
                stream=False,
                format=schema,
                options=options,
            )

            # Extract content
            content: str | None = None
            if isinstance(response, dict) and 'message' in response and isinstance(response['message'], dict):
                content = response['message'].get('content')
            else:
                # Some clients return objects with a .message attribute instead of dicts
                msg = getattr(response, 'message', None)

                if isinstance(msg, dict):
                    content = msg.get('content')
                elif msg is not None:
                    content = getattr(msg, 'content', None)

            if content is None:
                # fallback to string
                content = str(response)

            # If structured response requested, validate with pydantic model
            if response_model is not None:
                # Use pydantic v2 model validate json method
                try:
                    validated = response_model.model_validate_json(content)
                    # return model as dict
                    return validated.model_dump()  # type: ignore[attr-defined]
                except Exception as e:
                    logger.error(f'Failed to validate response with response_model: {e}')
                    # fallthrough to try json loads

            # Try parse JSON otherwise
            try:
                return json.loads(content)
            except Exception:
                return {'text': content}
        except Exception as e:
            # map obvious ollama rate limit / response errors to RateLimitError when possible
            err_name = e.__class__.__name__
            status_code = getattr(e, 'status_code', None) or getattr(e, 'status', None)
            if err_name in ('RequestError', 'ResponseError') and status_code == 429:
                raise RateLimitError from e
            logger.error(f'Error in generating LLM response (ollama): {e}')
            raise