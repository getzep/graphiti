"""Tests for GraphitiClients.complete_prompt facade."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.prompts import create_prompt_library, prompt_library
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.models import ChatPrompt, SystemMessage, UserMessage


def _clients(library=None, llm=None) -> GraphitiClients:
    llm_client = llm or MagicMock()
    if not hasattr(llm_client, 'generate_response') or not isinstance(
        llm_client.generate_response, AsyncMock
    ):
        llm_client.generate_response = AsyncMock(return_value={'extracted_entities': []})
    return GraphitiClients.model_construct(
        driver=MagicMock(),
        llm_client=llm_client,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
        tracer=MagicMock(),
        prompt_library=library or prompt_library,
    )


@pytest.mark.asyncio
async def test_complete_prompt_legacy_uses_fixed_schema():
    clients = _clients()
    await clients.complete_prompt(
        'extract_nodes.extract_message',
        {
            'episode_content': 'hi',
            'previous_episodes': [],
            'custom_extraction_instructions': '',
            'entity_types': [],
            'source_description': 't',
        },
    )
    kwargs = clients.llm_client.generate_response.await_args.kwargs
    assert kwargs['response_model'] is ExtractedEntities
    assert kwargs['prompt_name'] == 'extract_nodes.extract_message'


@pytest.mark.asyncio
async def test_complete_prompt_honors_chat_prompt_override():
    lib = create_prompt_library(
        {
            'extract_nodes': {
                'extract_message': lambda ctx: ChatPrompt(
                    system=SystemMessage(content='OVERRIDE'),
                    user=UserMessage(content='u'),
                )
            }
        }
    )
    clients = _clients(lib)
    await clients.complete_prompt(
        'extract_nodes.extract_message',
        {
            'episode_content': 'hi',
            'previous_episodes': [],
            'custom_extraction_instructions': '',
            'entity_types': [],
            'source_description': 't',
        },
    )
    messages = clients.llm_client.generate_response.await_args.args[0]
    assert messages[0].content.startswith('OVERRIDE')


@pytest.mark.asyncio
async def test_complete_prompt_rejects_wrong_schema():
    from pydantic import BaseModel

    class Other(BaseModel):
        x: int = 1

    clients = _clients()
    with pytest.raises(ValueError, match='schema overrides are not allowed'):
        await clients.complete_prompt(
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
