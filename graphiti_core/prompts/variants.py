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

Named prompt variants for provider models.

Use ``transport.model(id, prompt_variant='haiku')`` to select Haiku prompts,
or ``prompt_variant='default'`` to disable automatic model-specific prompts.
Explicit model and runtime prompt overrides take precedence over variants.
"""

from graphiti_core.llm_client.prompt_config import LLMPromptOverrides
from graphiti_core.prompts.haiku import haiku_prompt_overrides


def prompt_overrides_for_variant(name: str) -> LLMPromptOverrides:
    """Return the overrides for ``haiku`` or the unchanged ``default`` library."""
    if name == 'haiku':
        return haiku_prompt_overrides()
    if name == 'default':
        return LLMPromptOverrides()
    raise ValueError(f'Unknown prompt variant: {name!r}')
