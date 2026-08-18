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
    PromptLibrary,
    create_prompt_library,
    ensure_chat_prompt,
    ensure_prompt_library_wrapped,
    get_prompt_builder,
    resolve_response_model,
)
from graphiti_core.prompts.models import PromptFunction
from graphiti_core.prompts.names import PromptName
from graphiti_core.tracer import Tracer

from .client import LLMClient
from .config import ModelSize
from .prompt_config import (
    LLMModel,
    LLMPromptOverrides,
    PromptRoutes,
    flatten_overrides,
    flatten_routes,
)


def _wrap_builder(builder: PromptFunction, prompt_name: str) -> PromptFunction:
    def _call(context: dict[str, Any]):
        return ensure_chat_prompt(builder(context), prompt_name)

    return _call


class LLMRuntime:
    """Opt-in LLM runtime: one transport, a required default model, optional routes.

    Example::

        main = LLMModel(id='gpt-4.1', small_id='gpt-4.1-nano')
        nano = LLMModel(
            id='gpt-4.1-nano',
            prompt_overrides=LLMPromptOverrides(
                extract_nodes=LLMPromptOverrides.ExtractNodes(
                    extract_attributes=nano_extract_attrs,
                ),
            ),
        )
        runtime = LLMRuntime(
            transport=OpenAIClient(),
            model=main,
            routes=PromptRoutes(
                extract_nodes=PromptRoutes.ExtractNodes(extract_attributes=nano),
                extract_edges=PromptRoutes.ExtractEdges(extract_attributes=nano),
            ),
            prompt_overrides=LLMPromptOverrides(
                extract_nodes=LLMPromptOverrides.ExtractNodes(
                    extract_message=my_extract,
                ),
            ),
        )
        graphiti = Graphiti(..., llm_runtime=runtime)

    Builder resolution for prompt P on model M:

    1. ``M.prompt_overrides`` for P
    2. else general ``prompt_overrides`` for P
    3. else the builtin / supplied library

    Schemas are never part of that stack. v1 is multi-model on one transport
    that selects models via the ``model`` string attribute. Per-prompt routing
    passes ``model`` / ``small_model`` into ``generate_response``; the transport
    is never cloned or mutated.
    """

    def __init__(
        self,
        transport: LLMClient,
        model: LLMModel,
        *,
        routes: PromptRoutes | None = None,
        prompt_overrides: LLMPromptOverrides | None = None,
        library: PromptLibrary | None = None,
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

        self.transport = transport
        self.model = model
        self.routes = resolved_routes
        self.prompt_overrides = resolved_overrides
        self.library = resolved_library

    def set_tracer(self, tracer: Tracer) -> None:
        self.transport.set_tracer(tracer)

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

    def resolve_builder(self, prompt_name: str, model: LLMModel) -> PromptFunction:
        """Builder resolution: model override → general override → library method."""
        override = model.flat_overrides.get(prompt_name)
        if override is None:
            override = self.prompt_overrides.get(prompt_name)
        if override is not None:
            return _wrap_builder(override, prompt_name)
        return get_prompt_builder(self.library, prompt_name)

    def _small_id_for(self, model: LLMModel) -> str | None:
        """Resolve the per-call ``small_model`` override for ``model``.

        ``None`` means "leave the transport's ``small_model`` alone", which is
        what the default model does when ``small_id`` is omitted so
        ``ModelSize.small`` call sites keep working. Routed models without
        ``small_id`` pin small to their own ``id``.
        """
        if model.small_id is not None:
            return model.small_id
        if model is self.model:
            return None
        return model.id

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

        return await self.transport.generate_response(
            messages,
            response_model=resolved_schema,
            max_tokens=effective_max_tokens,
            model_size=model_size,
            group_id=group_id,
            prompt_name=prompt_name,
            attribute_extraction=attribute_extraction,
            model=model.id,
            small_model=self._small_id_for(model),
        )
