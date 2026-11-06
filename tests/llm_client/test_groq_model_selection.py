"""Offline tests for Groq model routing and response-cache isolation."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message

groq_module = pytest.importorskip('graphiti_core.llm_client.groq_client')


@pytest.fixture
def mock_groq(monkeypatch):
    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(
        side_effect=lambda **kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({'model': kwargs['model']}))
                )
            ]
        )
    )
    monkeypatch.setattr(groq_module, 'AsyncGroq', lambda **kwargs: sdk)
    return sdk


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('main_model', 'small_model', 'model_size', 'override', 'expected'),
    [
        ('main-model', 'small-model', ModelSize.small, None, 'small-model'),
        (None, 'small-model', ModelSize.small, None, 'small-model'),
        ('main-model', None, ModelSize.small, None, 'main-model'),
        (None, None, ModelSize.small, None, groq_module.DEFAULT_MODEL),
        ('main-model', 'small-model', ModelSize.medium, None, 'main-model'),
        (None, 'small-model', ModelSize.medium, None, groq_module.DEFAULT_MODEL),
        ('main-model', 'small-model', ModelSize.small, 'override', 'override'),
        ('main-model', 'small-model', ModelSize.medium, 'override', 'override'),
    ],
)
async def test_model_selection(mock_groq, main_model, small_model, model_size, override, expected):
    client = groq_module.GroqClient(
        LLMConfig(api_key='offline-test', model=main_model, small_model=small_model)
    )
    result = await client.generate_response(
        [Message(role='user', content='hello')], model_size=model_size, model=override
    )
    assert mock_groq.chat.completions.create.call_args.kwargs['model'] == expected
    assert result == {'model': expected}


@pytest.mark.asyncio
async def test_cache_distinguishes_small_and_main_models(mock_groq, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = groq_module.GroqClient(
        LLMConfig(api_key='offline-test', model='main-model', small_model='small-model'), cache=True
    )
    try:
        responses = []
        for size in (ModelSize.small, ModelSize.medium, ModelSize.small):
            responses.append(
                await client.generate_response(
                    [Message(role='user', content='hello')], model_size=size
                )
            )
        assert responses == [
            {'model': 'small-model'},
            {'model': 'main-model'},
            {'model': 'small-model'},
        ]
        assert mock_groq.chat.completions.create.call_count == 2
    finally:
        client.cache_dir.close()
