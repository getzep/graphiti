"""Regression tests for the documented `llm.api_base` / `embedder.api_base` overrides.

Issue: the README's Ollama guide documents a top-level `api_base` under both `llm` and
`embedder`, but the schema had no such field, and pydantic's `extra='ignore'` silently
dropped it — OpenAI clients then hit https://api.openai.com/v1 instead of the configured
local endpoint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.schema import (
    EmbedderConfig,
    EmbedderProvidersConfig,
    GraphitiConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
)
from services.factories import EmbedderFactory, LLMClientFactory


def test_llm_config_accepts_api_base():
    """`llm.api_base` must survive schema validation instead of being dropped."""
    config = LLMConfig(
        provider='openai',
        model='gpt-oss:120b',
        api_base='http://localhost:11434/v1',
    )
    assert config.api_base == 'http://localhost:11434/v1'


def test_embedder_config_accepts_api_base():
    config = EmbedderConfig(provider='openai', api_base='http://localhost:8081')
    assert config.api_base == 'http://localhost:8081'


def test_graphiti_config_keeps_api_base_from_nested_dict():
    config = GraphitiConfig.model_validate(
        {
            'llm': {
                'provider': 'openai',
                'model': 'LocalLLM',
                'api_base': 'https://mydomain.com/v1',
            },
            'embedder': {
                'provider': 'openai',
                'model': 'nomic-embed-text-v1.5.f16.gguf',
                'dimensions': 768,
                'api_base': 'http://192.168.1.10:8081',
            },
        }
    )
    assert config.llm.api_base == 'https://mydomain.com/v1'
    assert config.embedder.api_base == 'http://192.168.1.10:8081'


def test_llm_factory_prefers_api_base_over_provider_api_url():
    config = LLMConfig(
        provider='openai',
        model='LocalLLM',
        api_base='https://mydomain.com/v1',
        providers=LLMProvidersConfig(
            openai=OpenAIProviderConfig(api_key='not-needed', api_url='https://api.openai.com/v1')
        ),
    )
    client = LLMClientFactory.create(config)
    assert client.config.base_url == 'https://mydomain.com/v1'


def test_llm_factory_routes_api_base_to_generic_client():
    """A local endpoint in api_base must be treated as non-OpenAI, like the provider path."""
    config = LLMConfig(
        provider='openai',
        model='LocalLLM',
        api_base='http://localhost:11434/v1',
        providers=LLMProvidersConfig(
            openai=OpenAIProviderConfig(api_key='ollama', api_url='https://api.openai.com/v1')
        ),
    )
    client = LLMClientFactory.create(config)
    assert client.__class__.__name__ == 'OpenAIGenericClient'


def test_embedder_factory_prefers_api_base_over_provider_api_url():
    config = EmbedderConfig(
        provider='openai',
        model='nomic-embed-text-v1.5.f16.gguf',
        dimensions=768,
        api_base='http://192.168.1.10:8081',
        providers=EmbedderProvidersConfig(
            openai=OpenAIProviderConfig(api_key='not-needed', api_url='https://api.openai.com/v1')
        ),
    )
    embedder = EmbedderFactory.create(config)
    assert embedder.config.base_url == 'http://192.168.1.10:8081'


def test_llm_factory_falls_back_to_provider_api_url_without_api_base():
    config = LLMConfig(
        provider='openai',
        model='gpt-6-luna',
        providers=LLMProvidersConfig(
            openai=OpenAIProviderConfig(api_key='sk-test', api_url='https://proxy.example.com/v1')
        ),
    )
    client = LLMClientFactory.create(config)
    assert client.config.base_url == 'https://proxy.example.com/v1'
