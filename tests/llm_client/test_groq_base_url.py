"""Offline regression tests for the optional Groq client."""

import pytest

from graphiti_core.llm_client.config import LLMConfig

GroqClient = pytest.importorskip('graphiti_core.llm_client.groq_client').GroqClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('config_base_url', 'environment_base_url', 'expected_base_url'),
    [
        ('https://gateway.example/v1', None, 'https://gateway.example/v1'),
        ('https://configured.example', 'https://environment.example', 'https://configured.example'),
        (None, 'https://environment.example', 'https://environment.example'),
        (None, None, 'https://api.groq.com'),
    ],
)
async def test_base_url_configuration(
    monkeypatch, config_base_url, environment_base_url, expected_base_url
):
    if environment_base_url is None:
        monkeypatch.delenv('GROQ_BASE_URL', raising=False)
    else:
        monkeypatch.setenv('GROQ_BASE_URL', environment_base_url)
    client = GroqClient(LLMConfig(api_key='offline-test', base_url=config_base_url))
    try:
        assert str(client.client.base_url).rstrip('/') == expected_base_url
    finally:
        await client.client.close()
