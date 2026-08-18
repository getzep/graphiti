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
from graphiti_core.llm_client.llm_runtime import LLMRuntime
from graphiti_core.prompts.lib import get_prompt_builder, resolve_response_model
from graphiti_core.prompts.names import PromptName
from graphiti_core.tracer import Tracer


class GraphitiClients(BaseModel):
    driver: GraphDriver
    llm_client: LLMClient
    embedder: EmbedderClient
    cross_encoder: CrossEncoderClient
    tracer: Tracer
    # PromptLibrary is an ABC / duck-typed object; store as Any for Pydantic compatibility.
    prompt_library: Any
    llm_runtime: LLMRuntime | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def complete_prompt(
        self,
        prompt_name: PromptName,
        context: dict[str, Any],
        *,
        response_model: type[BaseModel] | None = None,
        model_size: ModelSize = ModelSize.medium,
        attribute_extraction: bool = False,
        group_id: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Resolve prompt text + fixed schema and call the LLM.

        Legacy path (no runtime): builder → ChatPrompt.as_messages → llm_client.generate_response.
        Runtime path: LLMRuntime.complete. ``model_size`` is forwarded on both paths.
        Schemas come from the immutable builtin registry, not the user library.
        """
        if self.llm_runtime is not None:
            return await self.llm_runtime.complete(
                prompt_name,
                context,
                response_model=response_model,
                attribute_extraction=attribute_extraction,
                group_id=group_id,
                max_tokens=max_tokens,
                model_size=model_size,
            )

        resolved_model = resolve_response_model(prompt_name, response_model)
        builder = get_prompt_builder(self.prompt_library, prompt_name)
        messages = builder(context).as_messages()

        return await self.llm_client.generate_response(
            messages,
            response_model=resolved_model,
            max_tokens=max_tokens,
            model_size=model_size,
            group_id=group_id,
            prompt_name=prompt_name,
            attribute_extraction=attribute_extraction,
        )
