"""Named prompt variants for provider models.

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
