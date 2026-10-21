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

from typing import Any

from pydantic import BaseModel

from graphiti_core.prompts.lib import (
    ChatPromptLibrary,
    create_prompt_library,
    ensure_chat_prompt,
    ensure_prompt_library_wrapped,
    get_prompt_builder,
    resolve_response_model,
)
from graphiti_core.prompts.models import ChatPromptFunction
from graphiti_core.prompts.names import PromptName
from graphiti_core.tracer import Tracer

from .client import LLMClient
from .config import ModelSize
from .prompt_config import (
    LLMModel,
    LLMPromptOverrides,
    LLMTransport,
    PromptRoutes,
    flatten_overrides,
    flatten_routes,
)


def _wrap_builder(builder: ChatPromptFunction, prompt_name: str) -> ChatPromptFunction:
    def _call(context: dict[str, Any]):
        return ensure_chat_prompt(builder(context), prompt_name)

    return _call


class LLMRuntime:
    """Route prompts across one or more provider transports.

    Example::

        from typing import Literal

        from graphiti_core import Graphiti
        from graphiti_core.llm_client.anthropic_client import AnthropicClient
        from graphiti_core.llm_client import (
            LLMRuntime,
            LLMTransport,
            OpenAIClient,
            PromptRoutes,
        )

        OpenAIModels = Literal['gpt-5.1', 'gpt-5-nano']
        openai = LLMTransport[OpenAIModels](
            OpenAIClient(),
            models=['gpt-5.1', 'gpt-5-nano'],
        )
        AnthropicModels = Literal['claude-sonnet-4-5', 'claude-haiku-4-5']
        anthropic = LLMTransport[AnthropicModels](
            AnthropicClient(),
            models=['claude-sonnet-4-5', 'claude-haiku-4-5'],
        )

        runtime = LLMRuntime(
            model=openai.model('gpt-5.1'),
            routes=PromptRoutes(
                extract_nodes=PromptRoutes.ExtractNodes(
                    extract_attributes=openai.model('gpt-5-nano'),
                ),
                dedupe_edges=PromptRoutes.DedupeEdges(
                    resolve_edge=anthropic.model('claude-haiku-4-5'),
                ),
            ),
        )
        graphiti = Graphiti(..., llm_runtime=runtime)

    The runtime uses the routed model ID for each prompt. An unrouted prompt
    uses the default model. ``model_size`` does not select a second model on
    this path. Route prompts that need a smaller model to that model explicitly.
    """

    def __init__(
        self,
        model: LLMModel,
        *,
        routes: PromptRoutes | None = None,
        prompt_overrides: LLMPromptOverrides | None = None,
        library: ChatPromptLibrary | None = None,
    ) -> None:
        if not isinstance(model, LLMModel):
            raise TypeError('model must be an LLMModel instance')
        if routes is not None and not isinstance(routes, PromptRoutes):
            raise TypeError('routes must be a PromptRoutes instance')
        if prompt_overrides is not None and not isinstance(prompt_overrides, LLMPromptOverrides):
            raise TypeError('prompt_overrides must be an LLMPromptOverrides instance')

        resolved_routes = flatten_routes(routes)
        resolved_overrides = flatten_overrides(prompt_overrides)

        if library is None:
            resolved_library = ensure_prompt_library_wrapped(create_prompt_library())
        else:
            resolved_library = ensure_prompt_library_wrapped(library)

        transports = []
        for routed_model in (model, *resolved_routes.values()):
            transport = routed_model.transport
            if not any(transport is existing for existing in transports):
                transports.append(transport)

        self.transports: tuple[LLMTransport[Any], ...] = tuple(transports)
        self.model = model
        self.routes = resolved_routes
        self.prompt_overrides = resolved_overrides
        self.library = resolved_library

    @property
    def client(self) -> LLMClient:
        return self.model.transport.client

    def set_tracer(self, tracer: Tracer) -> None:
        seen_clients: set[int] = set()
        for transport in self.transports:
            client = transport.client
            if id(client) in seen_clients:
                continue
            seen_clients.add(id(client))
            client.set_tracer(tracer)

    def resolve_model(self, prompt_name: str) -> LLMModel:
        """Return the LLMModel that should run ``prompt_name``."""
        routed = self.routes.get(prompt_name)
        if routed is not None:
            return routed
        group_name = prompt_name.split('.', 1)[0]
        routed = self.routes.get(group_name)
        if routed is not None:
            return routed
        return self.model

    def resolve_builder(self, prompt_name: str, model: LLMModel) -> ChatPromptFunction:
        """Builder resolution: model override → general override → library method."""
        override = model.flat_overrides.get(prompt_name)
        if override is None:
            override = self.prompt_overrides.get(prompt_name)
        if override is not None:
            return _wrap_builder(override, prompt_name)
        return get_prompt_builder(self.library, prompt_name)

    async def complete(
        self,
        prompt_name: PromptName,
        context: dict[str, Any],
        *,
        response_model: type[BaseModel] | None = None,
        attribute_extraction: bool = False,
        group_id: str | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        resolved_schema = resolve_response_model(prompt_name, response_model)
        model = self.resolve_model(prompt_name)
        builder = self.resolve_builder(prompt_name, model)
        messages = builder(context).as_messages()
        effective_max_tokens = max_tokens if max_tokens is not None else model.max_tokens

        return await model.transport.client.generate_response(
            messages,
            response_model=resolved_schema,
            max_tokens=effective_max_tokens,
            model_size=model_size,
            group_id=group_id,
            prompt_name=prompt_name,
            attribute_extraction=attribute_extraction,
            model=model.id,
        )
