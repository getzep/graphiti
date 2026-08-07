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

from typing import Any

from pydantic import BaseModel, ConfigDict

from graphiti_core.cross_encoder import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.lib import BUILTIN_PROMPT_SPECS, get_prompt_builder
from graphiti_core.prompts.models import ChatPrompt
from graphiti_core.tracer import Tracer


class GraphitiClients(BaseModel):
    driver: GraphDriver
    llm_client: LLMClient
    embedder: EmbedderClient
    cross_encoder: CrossEncoderClient
    tracer: Tracer
    # PromptLibrary is an ABC / duck-typed object; store as Any for Pydantic compatibility.
    prompt_library: Any
    # Optional PromptBoundLLM bundle; when set, complete_prompt routes through it.
    prompt_bound_llm: Any | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def complete_prompt(
        self,
        prompt_name: str,
        context: dict[str, Any],
        *,
        response_model: type[BaseModel] | None = None,
        model_size: ModelSize = ModelSize.medium,
        attribute_extraction: bool = False,
        group_id: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Resolve prompt text + fixed schema and call the LLM.

        Legacy path (no bundle): builder → ChatPrompt.as_messages → llm_client.generate_response.
        Bundle path: PromptBoundLLM.complete (model_size ignored; attribute_extraction honored).
        """
        if self.prompt_bound_llm is not None:
            return await self.prompt_bound_llm.complete(
                prompt_name,
                context,
                response_model=response_model,
                attribute_extraction=attribute_extraction,
                group_id=group_id,
                max_tokens=max_tokens,
                model_size=model_size,
            )

        specs = getattr(self.prompt_library, 'specs', BUILTIN_PROMPT_SPECS)
        spec = specs.get(prompt_name) or BUILTIN_PROMPT_SPECS.get(prompt_name)
        if spec is None:
            raise ValueError(f'Unknown prompt_name: {prompt_name}')

        if spec.dynamic_schema:
            if response_model is None:
                raise ValueError(
                    f'prompt {prompt_name} has dynamic_schema=True; response_model is required'
                )
            resolved_model: type[BaseModel] | None = response_model
        else:
            resolved_model = spec.response_model
            if response_model is not None and response_model is not resolved_model:
                raise ValueError(
                    f'Prompt schema overrides are not allowed for {prompt_name}: '
                    f'got {response_model.__name__}, expected '
                    f'{resolved_model.__name__ if resolved_model else None}'
                )

        builder = get_prompt_builder(self.prompt_library, prompt_name)
        chat_prompt = builder(context)
        if not isinstance(chat_prompt, ChatPrompt):
            raise TypeError(
                f'Prompt builder for {prompt_name} must return ChatPrompt, '
                f'got {type(chat_prompt).__name__}. '
                'Return ChatPrompt(system=..., user=...).'
            )
        messages = chat_prompt.as_messages()

        return await self.llm_client.generate_response(
            messages,
            response_model=resolved_model,
            max_tokens=max_tokens,
            model_size=model_size,
            group_id=group_id,
            prompt_name=prompt_name,
            attribute_extraction=attribute_extraction,
        )
