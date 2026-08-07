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

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict

from graphiti_core.prompts.lib import (
    BUILTIN_PROMPT_SPECS,
    PromptOverrides,
    create_prompt_library,
    get_prompt_builder,
    validate_prompt_library,
)
from graphiti_core.prompts.models import ChatPrompt, PromptFunction

from .client import LLMClient
from .config import ModelSize


class LLMModelConfig(BaseModel):
    """Named model slot configuration (exact provider model id).

    Reserved for future fields (temperature, max_tokens, structured_output_mode, ...).
    """

    model: str


class _PerCallModelTransport:
    """Wrap an LLMClient to select a model for the duration of a single call.

    Does not break the public LLMClient ABC. Concurrent callers are serialized
    only while temporarily swapping ``client.model`` when the provider has no
    non-mutating per-call model path.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self._lock = asyncio.Lock()

    @property
    def client(self) -> LLMClient:
        return self._client

    async def generate_response(
        self,
        messages: list[Any],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
        model: str | None = None,
    ) -> dict[str, Any]:
        if model is None:
            return await self._client.generate_response(
                messages,
                response_model=response_model,
                max_tokens=max_tokens,
                model_size=model_size,
                group_id=group_id,
                prompt_name=prompt_name,
                attribute_extraction=attribute_extraction,
            )

        async with self._lock:
            previous = getattr(self._client, 'model', None)
            try:
                if hasattr(self._client, 'model'):
                    self._client.model = model  # type: ignore[attr-defined]
                return await self._client.generate_response(
                    messages,
                    response_model=response_model,
                    max_tokens=max_tokens,
                    model_size=model_size,
                    group_id=group_id,
                    prompt_name=prompt_name,
                    attribute_extraction=attribute_extraction,
                )
            finally:
                if hasattr(self._client, 'model'):
                    self._client.model = previous  # type: ignore[attr-defined]


class PromptBoundLLM(BaseModel):
    """Opt-in bundle: single LLMClient transport + prompt library + per-prompt model slots."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Stored as Any so test doubles / structural transports are accepted; runtime expects LLMClient.
    transport: Any
    models: dict[str, LLMModelConfig]
    prompt_models: dict[str, str] = {}
    prompt_library: Any
    prompt_overrides: PromptOverrides | None = None
    model_prompt_overrides: dict[str, PromptOverrides] | None = None

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, '_transport_wrap', _PerCallModelTransport(self.transport))

    def resolve_model_id(self, prompt_name: str) -> str:
        slot = self.prompt_models.get(prompt_name, 'default')
        if slot not in self.models:
            raise ValueError(
                f'Unknown model slot {slot!r} for prompt {prompt_name!r}. '
                f'Known slots: {sorted(self.models)}'
            )
        return self.models[slot].model

    def resolve_builder(self, prompt_name: str, model_id: str) -> PromptFunction:
        """Builder resolution: model override → general override → library method."""
        group_name, method_name = prompt_name.split('.', 1)

        model_overrides = (self.model_prompt_overrides or {}).get(model_id)
        if (
            model_overrides
            and group_name in model_overrides
            and method_name in model_overrides[group_name]
        ):
            return model_overrides[group_name][method_name]

        general = self.prompt_overrides or {}
        if group_name in general and method_name in general[group_name]:
            return general[group_name][method_name]

        return get_prompt_builder(self.prompt_library, prompt_name)

    async def complete(
        self,
        prompt_name: str,
        context: dict[str, Any],
        *,
        response_model: type[BaseModel] | None = None,
        attribute_extraction: bool = False,
        group_id: str | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,  # ignored on bundle path
    ) -> dict[str, Any]:
        del model_size  # routing uses prompt_models slots
        specs = getattr(self.prompt_library, 'specs', BUILTIN_PROMPT_SPECS)
        spec = specs.get(prompt_name) or BUILTIN_PROMPT_SPECS.get(prompt_name)
        if spec is None:
            raise ValueError(f'Unknown prompt_name: {prompt_name}')

        if spec.dynamic_schema:
            if response_model is None:
                raise ValueError(
                    f'prompt {prompt_name} has dynamic_schema=True; response_model is required'
                )
            resolved_model = response_model
        else:
            resolved_model = spec.response_model
            if response_model is not None and response_model is not resolved_model:
                raise ValueError(
                    f'Prompt schema overrides are not allowed for {prompt_name}: '
                    f'got {response_model.__name__}, expected '
                    f'{resolved_model.__name__ if resolved_model else None}'
                )

        model_id = self.resolve_model_id(prompt_name)
        builder = self.resolve_builder(prompt_name, model_id)
        chat_prompt = builder(context)
        if not isinstance(chat_prompt, ChatPrompt):
            raise TypeError(
                f'Prompt builder for {prompt_name} must return ChatPrompt, '
                f'got {type(chat_prompt).__name__}. '
                'Return ChatPrompt(system=..., user=...).'
            )
        messages = chat_prompt.as_messages()

        return await self._transport_wrap.generate_response(  # type: ignore[attr-defined]
            messages,
            response_model=resolved_model,
            max_tokens=max_tokens,
            group_id=group_id,
            prompt_name=prompt_name,
            attribute_extraction=attribute_extraction,
            model=model_id,
        )


def create_prompt_bound_llm(
    client: LLMClient,
    models: dict[str, LLMModelConfig],
    prompt_models: dict[str, str] | None = None,
    prompt_overrides: PromptOverrides | None = None,
    model_prompt_overrides: dict[str, PromptOverrides] | None = None,
    prompt_library: Any | None = None,
) -> PromptBoundLLM:
    """Factory for a PromptBoundLLM wrapping a single-provider LLMClient transport."""
    if 'default' not in models:
        raise ValueError('models must include a "default" slot')

    if prompt_library is None:
        library = create_prompt_library(prompt_overrides)
    else:
        validate_prompt_library(prompt_library)
        library = prompt_library

    # Validate model_prompt_overrides group/function names
    for _model_id, overrides in (model_prompt_overrides or {}).items():
        # Touch create_prompt_library validation by composing once
        create_prompt_library(overrides)

    return PromptBoundLLM(
        transport=client,
        models=models,
        prompt_models=prompt_models or {},
        prompt_library=library,
        prompt_overrides=prompt_overrides,
        model_prompt_overrides=model_prompt_overrides,
    )
