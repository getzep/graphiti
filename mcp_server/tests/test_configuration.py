#!/usr/bin/env python3
"""Test script for configuration loading and factory patterns."""

import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
from pydantic import BaseModel

from config.schema import GraphitiConfig
from graphiti_core.prompts.extract_nodes import SummarizedEntities
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from services.instrumented_clients import InstrumentedEmbedderClient, InstrumentedLLMClient
from services.openai_compatible_client import OpenAICompatibleJSONClient
from services.factories import DatabaseDriverFactory, EmbedderFactory, LLMClientFactory


def test_config_loading():
    """Test loading configuration from YAML and environment variables."""
    print('Testing configuration loading...')

    # Test with default config.yaml
    config = GraphitiConfig()

    print('✓ Loaded configuration successfully')
    print(f'  - Server transport: {config.server.transport}')
    print(f'  - LLM provider: {config.llm.provider}')
    print(f'  - LLM model: {config.llm.model}')
    print(f'  - Embedder provider: {config.embedder.provider}')
    print(f'  - Database provider: {config.database.provider}')
    print(f'  - Group ID: {config.graphiti.group_id}')

    # Test environment variable override
    os.environ['LLM__PROVIDER'] = 'anthropic'
    os.environ['LLM__MODEL'] = 'claude-3-opus'
    config2 = GraphitiConfig()

    print('\n✓ Environment variable overrides work')
    print(f'  - LLM provider (overridden): {config2.llm.provider}')
    print(f'  - LLM model (overridden): {config2.llm.model}')

    # Clean up env vars
    del os.environ['LLM__PROVIDER']
    del os.environ['LLM__MODEL']

    assert config is not None
    assert config2 is not None

    # Return the first config for subsequent tests
    return config


def test_llm_factory(config: GraphitiConfig):
    """Test LLM client factory creation."""
    print('\nTesting LLM client factory...')

    # Test OpenAI client creation (if API key is set)
    if (
        config.llm.provider == 'openai'
        and config.llm.providers.openai
        and config.llm.providers.openai.api_key
    ):
        try:
            client = LLMClientFactory.create(config.llm)
            print(f'✓ Created {config.llm.provider} LLM client successfully')
            print(f'  - Model: {client.model}')
            print(f'  - Temperature: {client.temperature}')
        except Exception as e:
            print(f'✗ Failed to create LLM client: {e}')
    else:
        print(f'⚠ Skipping LLM factory test (no API key configured for {config.llm.provider})')

    # Test switching providers
    test_config = config.llm.model_copy()
    test_config.provider = 'gemini'
    if not test_config.providers.gemini:
        from config.schema import GeminiProviderConfig

        test_config.providers.gemini = GeminiProviderConfig(api_key='dummy_value_for_testing')
    else:
        test_config.providers.gemini.api_key = 'dummy_value_for_testing'

    try:
        client = LLMClientFactory.create(test_config)
        print('✓ Factory supports provider switching (tested with Gemini)')
    except Exception as e:
        print(f'✗ Factory provider switching failed: {e}')


def test_embedder_factory(config: GraphitiConfig):
    """Test Embedder client factory creation."""
    print('\nTesting Embedder client factory...')

    # Test OpenAI embedder creation (if API key is set)
    if (
        config.embedder.provider == 'openai'
        and config.embedder.providers.openai
        and config.embedder.providers.openai.api_key
    ):
        try:
            _ = EmbedderFactory.create(config.embedder)
            print(f'✓ Created {config.embedder.provider} Embedder client successfully')
            # The embedder client may not expose model/dimensions as attributes
            print(f'  - Configured model: {config.embedder.model}')
            print(f'  - Configured dimensions: {config.embedder.dimensions}')
        except Exception as e:
            print(f'✗ Failed to create Embedder client: {e}')
    else:
        print(
            f'⚠ Skipping Embedder factory test (no API key configured for {config.embedder.provider})'
        )


async def test_database_factory(config: GraphitiConfig):
    """Test Database driver factory creation."""
    print('\nTesting Database driver factory...')

    # Test Neo4j config creation
    if config.database.provider == 'neo4j' and config.database.providers.neo4j:
        try:
            db_config = DatabaseDriverFactory.create_config(config.database)
            print(f'✓ Created {config.database.provider} configuration successfully')
            print(f'  - URI: {db_config["uri"]}')
            print(f'  - User: {db_config["user"]}')
            print(
                f'  - Password: {"*" * len(db_config["password"]) if db_config["password"] else "None"}'
            )

            # Test actual connection would require initializing Graphiti
            from graphiti_core import Graphiti

            try:
                # This will fail if Neo4j is not running, but tests the config
                graphiti = Graphiti(
                    uri=db_config['uri'],
                    user=db_config['user'],
                    password=db_config['password'],
                )
                await graphiti.driver.client.verify_connectivity()
                print('  ✓ Successfully connected to Neo4j')
                await graphiti.driver.client.close()
            except Exception as e:
                print(f'  ⚠ Could not connect to Neo4j (is it running?): {type(e).__name__}')
        except Exception as e:
            print(f'✗ Failed to create Database configuration: {e}')
    else:
        print(f'⚠ Skipping Database factory test (no configuration for {config.database.provider})')


def test_cli_override():
    """Test CLI argument override functionality."""
    print('\nTesting CLI argument override...')

    # Simulate argparse Namespace
    class Args:
        config = Path('config.yaml')
        transport = 'stdio'
        llm_provider = 'anthropic'
        model = 'claude-3-sonnet'
        temperature = 0.5
        embedder_provider = 'voyage'
        embedder_model = 'voyage-3'
        database_provider = 'falkordb'
        group_id = 'test-group'
        user_id = 'test-user'

    config = GraphitiConfig()
    config.apply_cli_overrides(Args())

    print('✓ CLI overrides applied successfully')
    print(f'  - Transport: {config.server.transport}')
    print(f'  - LLM provider: {config.llm.provider}')
    print(f'  - LLM model: {config.llm.model}')
    print(f'  - Temperature: {config.llm.temperature}')
    print(f'  - Embedder provider: {config.embedder.provider}')
    print(f'  - Database provider: {config.database.provider}')
    print(f'  - Group ID: {config.graphiti.group_id}')
    print(f'  - User ID: {config.graphiti.user_id}')


async def main():
    """Run all tests."""
    print('=' * 60)
    print('Configuration and Factory Pattern Test Suite')
    print('=' * 60)

    try:
        # Test configuration loading
        config = test_config_loading()

        # Test factories
        test_llm_factory(config)
        test_embedder_factory(config)
        await test_database_factory(config)

        # Test CLI overrides
        test_cli_override()

        print('\n' + '=' * 60)
        print('✓ All tests completed successfully!')
        print('=' * 60)

    except Exception as e:
        print(f'\n✗ Test suite failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())


def test_nas_ollama_embedder_config_resolution(monkeypatch):
    """Resolve the confirmed NAS deployment config without changing the live baseline."""
    config_path = Path(__file__).parent.parent / 'config' / 'config-docker-neo4j-external.yaml'

    monkeypatch.setenv('CONFIG_PATH', str(config_path))
    monkeypatch.setenv('LLM_MODEL', 'glm-5')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-glm-key')
    monkeypatch.setenv('OPENAI_API_URL', 'https://open.bigmodel.cn/api/paas/v4/')
    monkeypatch.setenv('EMBEDDER_OPENAI_API_KEY', 'ollama')
    monkeypatch.setenv('EMBEDDER_OPENAI_API_URL', 'http://192.168.123.74:11434/v1')
    monkeypatch.setenv('EMBEDDER_MODEL', 'qwen3-embedding:0.6b')
    monkeypatch.setenv('EMBEDDER_DIMENSIONS', '1024')
    monkeypatch.setenv('NEO4J_PASSWORD', 'test-password')

    config = GraphitiConfig()

    assert config.llm.provider == 'openai'
    assert config.llm.model == 'glm-5'
    assert config.llm.providers.openai.api_url == 'https://open.bigmodel.cn/api/paas/v4/'
    assert config.embedder.provider == 'openai'
    assert config.embedder.model == 'qwen3-embedding:0.6b'
    assert config.embedder.dimensions == 1024
    assert config.embedder.providers.openai.api_key == 'ollama'
    assert config.embedder.providers.openai.api_url == 'http://192.168.123.74:11434/v1'


def test_runtime_config_summary_includes_llm_and_embedder_baseline(monkeypatch):
    """Summarize effective runtime config without exposing secrets."""
    config_path = Path(__file__).parent.parent / 'config' / 'config-docker-neo4j-external.yaml'

    monkeypatch.setenv('CONFIG_PATH', str(config_path))
    monkeypatch.setenv('LLM_MODEL', 'glm-5')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-glm-key')
    monkeypatch.setenv('OPENAI_API_URL', 'https://open.bigmodel.cn/api/paas/v4/')
    monkeypatch.setenv('EMBEDDER_OPENAI_API_KEY', 'ollama')
    monkeypatch.setenv('EMBEDDER_OPENAI_API_URL', 'http://192.168.123.74:11434/v1')
    monkeypatch.setenv('EMBEDDER_MODEL', 'qwen3-embedding:0.6b')
    monkeypatch.setenv('EMBEDDER_DIMENSIONS', '1024')
    monkeypatch.setenv('NEO4J_PASSWORD', 'test-password')

    config = GraphitiConfig()
    graphiti_mcp_server = importlib.import_module('graphiti_mcp_server')

    summary = graphiti_mcp_server.summarize_runtime_config(config)

    assert summary['llm_provider'] == 'openai'
    assert summary['llm_model'] == 'glm-5'
    assert summary['llm_base_url'] == 'https://open.bigmodel.cn/api/paas/v4/'
    assert summary['embedder_provider'] == 'openai'
    assert summary['embedder_model'] == 'qwen3-embedding:0.6b'
    assert summary['embedder_base_url'] == 'http://192.168.123.74:11434/v1'
    assert summary['embedder_dimensions'] == 1024
    assert 'api_key' not in summary


def test_openai_llm_factory_preserves_configured_base_url(monkeypatch):
    """Ensure the OpenAI-compatible LLM factory actually uses the configured API URL."""
    config_path = Path(__file__).parent.parent / 'config' / 'config-docker-neo4j-external.yaml'

    monkeypatch.setenv('CONFIG_PATH', str(config_path))
    monkeypatch.setenv('LLM_MODEL', 'glm-5')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-glm-key')
    monkeypatch.setenv('OPENAI_API_URL', 'https://open.bigmodel.cn/api/paas/v4/')
    monkeypatch.setenv('NEO4J_PASSWORD', 'test-password')

    config = GraphitiConfig()
    client = LLMClientFactory.create(config.llm)

    assert client.config.base_url == 'https://open.bigmodel.cn/api/paas/v4/'
    assert client.model == 'glm-5'


def test_openai_llm_factory_uses_generic_client_for_custom_base_url(monkeypatch):
    """Use the generic OpenAI-compatible client for custom provider endpoints like GLM/Ollama."""
    config_path = Path(__file__).parent.parent / 'config' / 'config-docker-neo4j-external.yaml'

    monkeypatch.setenv('CONFIG_PATH', str(config_path))
    monkeypatch.setenv('LLM_MODEL', 'glm-5')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-glm-key')
    monkeypatch.setenv('OPENAI_API_URL', 'https://open.bigmodel.cn/api/paas/v4/')
    monkeypatch.setenv('NEO4J_PASSWORD', 'test-password')

    config = GraphitiConfig()
    client = LLMClientFactory.create(config.llm)

    assert client.__class__.__name__ == 'InstrumentedLLMClient'
    assert client._inner.__class__.__name__ == 'OpenAICompatibleJSONClient'


def test_anthropic_llm_factory_uses_configured_base_url(monkeypatch):
    """Anthropic-compatible providers must pass through the configured API URL."""
    import services.factories as factories
    from config.schema import AnthropicProviderConfig, LLMConfig as ServerLLMConfig, LLMProvidersConfig

    captured: dict[str, object] = {}

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured['kwargs'] = kwargs

    class FakeAnthropicClient:
        def __init__(self, config=None, cache=False, client=None, max_tokens=None):
            captured['client'] = client
            captured['config'] = config
            self.config = config
            self.model = config.model
            self.small_model = config.model
            self.temperature = config.temperature
            self.max_tokens = config.max_tokens
            self.tracer = None
            self.token_tracker = None

        def set_tracer(self, tracer):
            self.tracer = tracer

    monkeypatch.setattr(factories, 'HAS_ANTHROPIC', True)
    monkeypatch.setattr(factories, 'AsyncAnthropic', FakeAsyncAnthropic, raising=False)
    monkeypatch.setattr(factories, 'AnthropicClient', FakeAnthropicClient, raising=False)

    config = ServerLLMConfig(
        provider='anthropic',
        model='glm-5',
        max_tokens=4096,
        providers=LLMProvidersConfig(
            anthropic=AnthropicProviderConfig(
                api_key='test-anthropic-key',
                api_url='https://open.bigmodel.cn/api/anthropic',
            )
        ),
    )

    _ = factories.LLMClientFactory.create(config)

    assert captured['client'] is not None
    assert captured['kwargs'] == {
        'api_key': 'test-anthropic-key',
        'base_url': 'https://open.bigmodel.cn/api/anthropic',
        'max_retries': 1,
    }


@pytest.mark.asyncio
async def test_openai_compatible_client_uses_json_object_for_structured_output():
    """Structured output for OpenAI-compatible providers should avoid json_schema mode."""

    class StructuredResponse(BaseModel):
        city: str

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"city":"Hangzhou"}'))]
            )

    completions = FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = OpenAICompatibleJSONClient(
        config=importlib.import_module('graphiti_core.llm_client.config').LLMConfig(
            api_key='test-key',
            model='glm-5',
            base_url='https://open.bigmodel.cn/api/paas/v4',
        ),
        client=fake_client,
    )

    response = await client._generate_response(
        messages=[
            importlib.import_module('graphiti_core.prompts.models').Message(
                role='system', content='Return JSON.'
            ),
            importlib.import_module('graphiti_core.prompts.models').Message(
                role='user', content='Return {"city":"Hangzhou"}.'
            ),
        ],
        response_model=StructuredResponse,
    )

    assert response == {'city': 'Hangzhou'}
    assert completions.kwargs['response_format'] == {'type': 'json_object'}


@pytest.mark.asyncio
async def test_openai_compatible_client_normalizes_answer_wrapper_for_single_field_models():
    """Normalize provider-specific `answer` wrappers into the expected top-level schema field."""

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"answer":[{"name":"Codex Smoke Tester","summary":"Prefers jasmine tea."}],"attributes":{}}'
                        )
                    )
                ]
            )

    completions = FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = OpenAICompatibleJSONClient(
        config=importlib.import_module('graphiti_core.llm_client.config').LLMConfig(
            api_key='test-key',
            model='glm-5',
            base_url='https://open.bigmodel.cn/api/paas/v4',
        ),
        client=fake_client,
    )

    response = await client._generate_response(
        messages=[
            importlib.import_module('graphiti_core.prompts.models').Message(
                role='system', content='Return JSON.'
            ),
            importlib.import_module('graphiti_core.prompts.models').Message(
                role='user', content='Return wrapped summary JSON.'
            ),
        ],
        response_model=SummarizedEntities,
    )

    assert response == {
        'summaries': [{'name': 'Codex Smoke Tester', 'summary': 'Prefers jasmine tea.'}]
    }


@pytest.mark.asyncio
async def test_openai_compatible_client_extracts_json_from_fenced_content():
    """Strip markdown fences or extra text before JSON parsing for compatible providers."""

    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='```json\\n{\"answer\": [{\"name\": \"Codex Smoke Tester\", \"entity_type_id\": 1}]}\\n```'
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = OpenAICompatibleJSONClient(
        config=importlib.import_module('graphiti_core.llm_client.config').LLMConfig(
            api_key='test-key',
            model='glm-5',
            base_url='https://open.bigmodel.cn/api/paas/v4',
        ),
        client=fake_client,
    )

    response = await client._generate_response(
        messages=[
            importlib.import_module('graphiti_core.prompts.models').Message(
                role='system', content='Return JSON.'
            ),
            importlib.import_module('graphiti_core.prompts.models').Message(
                role='user', content='Return entities.'
            ),
        ],
        response_model=ExtractedEntities,
    )

    assert response == {
        'extracted_entities': [{'name': 'Codex Smoke Tester', 'entity_type_id': 1}]
    }


@pytest.mark.asyncio
async def test_openai_compatible_client_wraps_list_for_single_field_models():
    """If a compatible provider returns a bare list, wrap it into the only schema field."""

    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='[{\"name\": \"Codex Smoke Tester\", \"entity_type_id\": 1}]'
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = OpenAICompatibleJSONClient(
        config=importlib.import_module('graphiti_core.llm_client.config').LLMConfig(
            api_key='test-key',
            model='glm-5',
            base_url='https://open.bigmodel.cn/api/paas/v4',
        ),
        client=fake_client,
    )

    response = await client._generate_response(
        messages=[
            importlib.import_module('graphiti_core.prompts.models').Message(
                role='system', content='Return JSON.'
            ),
            importlib.import_module('graphiti_core.prompts.models').Message(
                role='user', content='Return entities.'
            ),
        ],
        response_model=ExtractedEntities,
    )

    assert response == {
        'extracted_entities': [{'name': 'Codex Smoke Tester', 'entity_type_id': 1}]
    }


@pytest.mark.asyncio
async def test_openai_compatible_client_maps_single_list_field_to_expected_schema():
    """Map provider keys like `entities` onto the expected single-field schema name."""

    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"entities":[{"name":"Codex Smoke Tester","entity_type_id":1}]}'
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = OpenAICompatibleJSONClient(
        config=importlib.import_module('graphiti_core.llm_client.config').LLMConfig(
            api_key='test-key',
            model='glm-5',
            base_url='https://open.bigmodel.cn/api/paas/v4',
        ),
        client=fake_client,
    )

    response = await client._generate_response(
        messages=[
            importlib.import_module('graphiti_core.prompts.models').Message(
                role='system', content='Return JSON.'
            ),
            importlib.import_module('graphiti_core.prompts.models').Message(
                role='user', content='Return entities.'
            ),
        ],
        response_model=ExtractedEntities,
    )

    assert response == {
        'extracted_entities': [{'name': 'Codex Smoke Tester', 'entity_type_id': 1}]
    }


@pytest.mark.asyncio
async def test_openai_compatible_client_appends_schema_instruction_for_response_models():
    """Preserve schema guidance so compatible providers are more likely to emit valid JSON."""

    class StructuredResponse(BaseModel):
        city: str

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"city":"Hangzhou"}'))]
            )

    completions = FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = OpenAICompatibleJSONClient(
        config=importlib.import_module('graphiti_core.llm_client.config').LLMConfig(
            api_key='test-key',
            model='glm-5',
            base_url='https://open.bigmodel.cn/api/paas/v4',
        ),
        client=fake_client,
    )

    messages = [
        importlib.import_module('graphiti_core.prompts.models').Message(
            role='system', content='Return JSON.'
        ),
        importlib.import_module('graphiti_core.prompts.models').Message(
            role='user', content='Return the city.'
        ),
    ]

    await client.generate_response(messages, response_model=StructuredResponse)

    user_message = completions.kwargs['messages'][-1]['content']
    assert 'Respond with a JSON object in the following format' in user_message
    assert '"city"' in user_message


@pytest.mark.asyncio
async def test_openai_compatible_client_logs_prompt_timing(caplog):
    """Log prompt timing so slow Graphiti stages can be identified in NAS logs."""

    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = OpenAICompatibleJSONClient(
        config=importlib.import_module('graphiti_core.llm_client.config').LLMConfig(
            api_key='test-key',
            model='glm-5',
            base_url='https://open.bigmodel.cn/api/paas/v4',
        ),
        client=fake_client,
    )

    messages = [
        importlib.import_module('graphiti_core.prompts.models').Message(
            role='system', content='Return JSON.'
        ),
        importlib.import_module('graphiti_core.prompts.models').Message(
            role='user', content='Return {"ok": true}.'
        ),
    ]

    with caplog.at_level('INFO'):
        response = await client.generate_response(messages, prompt_name='smoke.prompt')

    assert response == {'ok': True}
    assert 'smoke.prompt' in caplog.text
    assert 'llm timing' in caplog.text.lower()


@pytest.mark.asyncio
async def test_instrumented_llm_client_logs_prompt_timing(caplog):
    """Log timing in the mcp_server layer so the NAS container actually emits prompt latency."""

    class FakeLLM:
        config = importlib.import_module('graphiti_core.llm_client.config').LLMConfig(
            api_key='test-key',
            model='glm-5',
            base_url='https://open.bigmodel.cn/api/paas/v4',
        )
        model = 'glm-5'
        small_model = 'glm-5'
        temperature = 0
        max_tokens = 4096
        tracer = None
        token_tracker = None

        async def generate_response(self, *args, **kwargs):
            return {'ok': True}

        def set_tracer(self, tracer):
            return None

    client = InstrumentedLLMClient(FakeLLM())

    assert isinstance(client, LLMClient)

    with caplog.at_level('INFO'):
        response = await client.generate_response(
            [], model_size=ModelSize.small, prompt_name='smoke.prompt'
        )

    assert response == {'ok': True}
    assert 'LLM timing' in caplog.text
    assert 'smoke.prompt' in caplog.text
    assert 'glm-5' in caplog.text


@pytest.mark.asyncio
async def test_openai_embedder_logs_timing(caplog):
    """Log embedding timing so local Ollama latency can be compared against LLM latency."""

    class FakeEmbeddings:
        async def create(self, **kwargs):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

    fake_client = SimpleNamespace(embeddings=FakeEmbeddings())
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key='ollama',
            base_url='http://127.0.0.1:11434/v1',
            embedding_model='qwen3-embedding:0.6b',
            embedding_dim=3,
        ),
        client=fake_client,
    )

    with caplog.at_level('INFO'):
        result = await embedder.create('hello')

    assert result == [0.1, 0.2, 0.3]
    assert 'embedder timing' in caplog.text.lower()
    assert 'qwen3-embedding:0.6b' in caplog.text


@pytest.mark.asyncio
async def test_instrumented_embedder_client_logs_timing(caplog):
    """Log embedder latency in the mcp_server layer so the NAS container can expose it."""

    class FakeEmbedder:
        async def create(self, input_data):
            return [0.1, 0.2, 0.3]

        async def create_batch(self, input_data_list):
            return [[0.1, 0.2, 0.3] for _ in input_data_list]

    embedder = InstrumentedEmbedderClient(
        inner=FakeEmbedder(),
        model_name='qwen3-embedding:0.6b',
        dimensions=1024,
    )

    assert isinstance(embedder, EmbedderClient)

    with caplog.at_level('INFO'):
        result = await embedder.create('hello')

    assert result == [0.1, 0.2, 0.3]
    assert 'Embedder timing' in caplog.text
    assert 'qwen3-embedding:0.6b' in caplog.text


@pytest.mark.asyncio
async def test_embedder_connectivity_failure_reports_model_and_base_url(monkeypatch):
    """Surface a precise error when the configured embedder endpoint is unreachable."""
    config_path = Path(__file__).parent.parent / 'config' / 'config-docker-neo4j-external.yaml'

    monkeypatch.setenv('CONFIG_PATH', str(config_path))
    monkeypatch.setenv('LLM_MODEL', 'glm-5')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-glm-key')
    monkeypatch.setenv('OPENAI_API_URL', 'https://open.bigmodel.cn/api/paas/v4/')
    monkeypatch.setenv('EMBEDDER_OPENAI_API_KEY', 'ollama')
    monkeypatch.setenv('EMBEDDER_OPENAI_API_URL', 'http://192.168.123.74:11434/v1')
    monkeypatch.setenv('EMBEDDER_MODEL', 'qwen3-embedding:0.6b')
    monkeypatch.setenv('EMBEDDER_DIMENSIONS', '1024')
    monkeypatch.setenv('NEO4J_PASSWORD', 'test-password')

    config = GraphitiConfig()
    graphiti_mcp_server = importlib.import_module('graphiti_mcp_server')

    class BrokenEmbedder:
        async def create(self, input_data):
            raise RuntimeError('connection refused')

    monkeypatch.setattr(
        graphiti_mcp_server.EmbedderFactory,
        'create',
        lambda _: BrokenEmbedder(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await graphiti_mcp_server.verify_embedder_connectivity(config)

    message = str(exc_info.value)
    assert 'qwen3-embedding:0.6b' in message
    assert 'http://192.168.123.74:11434/v1' in message
    assert 'ollama' not in message
