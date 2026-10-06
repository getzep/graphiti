"""Tests that generate_response exposes prompt_name via the current_prompt_name
context var so downstream _generate_response implementations can tag requests."""

import pytest

from graphiti_core.llm_client.client import LLMClient, current_prompt_name
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message


class RecordingClient(LLMClient):
    """Records the prompt_name visible inside _generate_response."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.seen_prompt_name: str | None = 'UNSET'

    async def _generate_response(
        self, messages, response_model=None, max_tokens=0, model_size=None
    ) -> dict:
        self.seen_prompt_name = current_prompt_name.get()
        return {'content': '{}'}


@pytest.mark.asyncio
async def test_generate_response_exposes_prompt_name():
    client = RecordingClient(LLMConfig())
    await client.generate_response(
        [Message(role='user', content='hi')],
        prompt_name='extract_edges.edge',
    )
    assert client.seen_prompt_name == 'extract_edges.edge'


@pytest.mark.asyncio
async def test_contextvar_resets_after_call():
    client = RecordingClient(LLMConfig())
    await client.generate_response([Message(role='user', content='hi')], prompt_name='x')
    # The token is reset in a finally block, so it must not leak.
    assert current_prompt_name.get() is None


@pytest.mark.asyncio
async def test_prompt_name_defaults_to_none():
    client = RecordingClient(LLMConfig())
    await client.generate_response([Message(role='user', content='hi')])
    assert client.seen_prompt_name is None
