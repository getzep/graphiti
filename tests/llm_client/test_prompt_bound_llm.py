"""Tests for PromptBoundLLM routing and override layering."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.llm_client.prompt_bound import (
    LLMModelConfig,
    create_prompt_bound_llm,
)
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.models import ChatPrompt, SystemMessage, UserMessage


def _chat(marker: str) -> ChatPrompt:
    return ChatPrompt(
        system=SystemMessage(content=marker),
        user=UserMessage(content=marker),
    )


@pytest.mark.asyncio
async def test_prompt_bound_routes_to_mapped_model_slot():
    client = MagicMock()
    client.model = 'gpt-4.1'
    client.generate_response = AsyncMock(return_value={'extracted_entities': []})

    bundle = create_prompt_bound_llm(
        client,
        models={
            'default': LLMModelConfig(model='gpt-4.1'),
            'fast': LLMModelConfig(model='gpt-4.1-nano'),
        },
        prompt_models={'extract_nodes.extract_message': 'fast'},
    )

    await bundle.complete(
        'extract_nodes.extract_message',
        {
            'episode_content': 'hi',
            'previous_episodes': [],
            'custom_extraction_instructions': '',
            'entity_types': [],
            'source_description': 't',
        },
    )

    assert client.generate_response.await_count == 1
    assert client.model == 'gpt-4.1'  # restored after call
    kwargs = client.generate_response.await_args.kwargs
    assert kwargs['prompt_name'] == 'extract_nodes.extract_message'
    assert kwargs['response_model'] is ExtractedEntities


@pytest.mark.asyncio
async def test_model_prompt_overrides_beat_general_overrides():
    client = MagicMock()
    client.model = 'gpt-4.1'
    client.generate_response = AsyncMock(return_value={'extracted_entities': []})

    general = {
        'extract_nodes': {'extract_message': lambda ctx: _chat('GENERAL')},
    }
    model_specific = {
        'gpt-4.1-nano': {
            'extract_nodes': {'extract_message': lambda ctx: _chat('MODEL')},
        }
    }

    bundle = create_prompt_bound_llm(
        client,
        models={
            'default': LLMModelConfig(model='gpt-4.1'),
            'fast': LLMModelConfig(model='gpt-4.1-nano'),
        },
        prompt_models={'extract_nodes.extract_message': 'fast'},
        prompt_overrides=general,
        model_prompt_overrides=model_specific,
    )

    await bundle.complete(
        'extract_nodes.extract_message',
        {
            'episode_content': 'hi',
            'previous_episodes': [],
            'custom_extraction_instructions': '',
            'entity_types': [],
            'source_description': 't',
        },
    )
    messages = client.generate_response.await_args.args[0]
    assert messages[0].content.startswith('MODEL')


@pytest.mark.asyncio
async def test_schema_override_rejected_on_bundle_complete():
    from pydantic import BaseModel

    class Other(BaseModel):
        x: int = 1

    client = MagicMock()
    client.model = 'gpt-4.1'
    client.generate_response = AsyncMock(return_value={})
    bundle = create_prompt_bound_llm(
        client,
        models={'default': LLMModelConfig(model='gpt-4.1')},
    )
    with pytest.raises(ValueError, match='schema overrides are not allowed'):
        await bundle.complete(
            'extract_nodes.extract_message',
            {
                'episode_content': 'hi',
                'previous_episodes': [],
                'custom_extraction_instructions': '',
                'entity_types': [],
                'source_description': 't',
            },
            response_model=Other,
        )


@pytest.mark.asyncio
async def test_same_response_model_identity_allowed():
    """Passing the exact fixed response_model is accepted (not an override)."""
    client = MagicMock()
    client.model = 'gpt-4.1'
    client.generate_response = AsyncMock(return_value={'extracted_entities': []})
    bundle = create_prompt_bound_llm(
        client,
        models={'default': LLMModelConfig(model='gpt-4.1')},
    )
    # Same object identity as registry — should succeed
    await bundle.complete(
        'dedupe_edges.resolve_edge',
        {'existing_edges': [], 'edge_invalidation_candidates': [], 'new_edge': 'x'},
    )
    assert client.generate_response.await_count == 1
