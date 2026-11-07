import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import graphiti_core.llm_client.client as client_module
import graphiti_core.llm_client.groq_client as groq_module
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message


@pytest.fixture
def cached_groq(monkeypatch, tmp_path):
    monkeypatch.setattr(client_module, 'DEFAULT_CACHE_DIR', str(tmp_path / 'cache'))
    calls = []
    clients = []

    async def complete(**kwargs):
        calls.append(kwargs)
        content = json.dumps(
            {'max_tokens': kwargs['max_tokens'], 'temperature': kwargs['temperature']}
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(
        groq_module,
        'AsyncGroq',
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=complete))
            )
        ),
    )

    def create(temperature=0):
        client = groq_module.GroqClient(
            LLMConfig(model='test-model', max_tokens=32, temperature=temperature), cache=True
        )
        clients.append(client)
        return client

    yield create, calls
    for client in clients:
        client.cache_dir.close()


def messages():
    return [Message(role='user', content='Return the answer as JSON.')]


@pytest.mark.parametrize('budgets', [(4, 64), (64, 4)])
async def test_response_cache_preserves_requested_token_budget(cached_groq, budgets):
    create, calls = cached_groq
    client = create()

    for budget in budgets:
        result = await client.generate_response(messages(), max_tokens=budget)
        assert result['max_tokens'] == budget
    assert len(calls) == 2

    assert (await client.generate_response(messages(), max_tokens=budgets[-1]))[
        'max_tokens'
    ] == budgets[-1]
    assert len(calls) == 2


async def test_response_cache_separates_clients_with_different_temperatures(cached_groq):
    create, calls = cached_groq
    cold = create(temperature=0)
    warm = create(temperature=1)

    assert (await cold.generate_response(messages()))['temperature'] == 0
    assert (await warm.generate_response(messages()))['temperature'] == 1
    assert len(calls) == 2


async def test_response_cache_reuses_identical_generation_options(cached_groq):
    create, calls = cached_groq
    client = create()

    first = await client.generate_response(messages(), max_tokens=12)
    assert await client.generate_response(messages(), max_tokens=12) == first
    assert len(calls) == 1


async def test_cache_write_does_not_change_generation_identity(cached_groq):
    create, calls = cached_groq
    client = create(temperature=0)
    started = asyncio.Event()
    release = asyncio.Event()
    complete = client.client.chat.completions.create

    async def blocked_completion(**kwargs):
        started.set()
        await release.wait()
        return await complete(**kwargs)

    client.client.chat.completions.create = blocked_completion
    pending = asyncio.create_task(client.generate_response(messages(), max_tokens=12))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        client.temperature = 1
        release.set()
        assert (await asyncio.wait_for(pending, timeout=1))['temperature'] == 0

        warm = create(temperature=1)
        assert (await warm.generate_response(messages(), max_tokens=12))['temperature'] == 1
        assert len(calls) == 2
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
