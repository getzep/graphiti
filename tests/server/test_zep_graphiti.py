import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'server'))
sys.modules.setdefault(
    'fastapi',
    SimpleNamespace(
        Depends=lambda dependency: dependency,
        HTTPException=Exception,
    ),
)
sys.modules.setdefault(
    'pydantic_settings',
    SimpleNamespace(
        BaseSettings=object,
        SettingsConfigDict=lambda **kwargs: kwargs,
    ),
)

from graph_service.zep_graphiti import get_graphiti  # noqa: E402


@pytest.mark.asyncio
async def test_get_graphiti_applies_small_and_embedding_model_settings():
    settings = SimpleNamespace(
        neo4j_uri='bolt://neo4j:7687',
        neo4j_user='neo4j',
        neo4j_password='password',
        openai_base_url='http://proxy:8081/v1',
        openai_api_key='test-key',
        model_name='MiniMax-M2.7',
        small_model_name='MiniMax-M2.7',
        embedding_model_name='nomic-embed-text',
    )

    client_gen = get_graphiti(settings)
    client = await anext(client_gen)
    try:
        assert client.llm_client.model == 'MiniMax-M2.7'
        assert client.llm_client.small_model == 'MiniMax-M2.7'
        assert client.embedder.config.embedding_model == 'nomic-embed-text'
        assert str(client.llm_client.client.base_url) == 'http://proxy:8081/v1/'
        assert str(client.embedder.client.base_url) == 'http://proxy:8081/v1/'
    finally:
        await client_gen.aclose()
