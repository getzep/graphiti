"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from graphiti_core.config import (
    DatabaseConfig,
    EmbedderConfig,
    EmbedderProvider,
    GraphitiConfig,
    LLMProvider,
    LLMProviderConfig,
    RerankerConfig,
)
from graphiti_core.config.providers import DatabaseProvider, RerankerProvider


class TestLLMProviderConfig:
    def test_openai_defaults(self):
        """Test OpenAI provider defaults are set correctly."""
        config = LLMProviderConfig(provider=LLMProvider.OPENAI)

        assert config.provider == LLMProvider.OPENAI
        assert config.model == 'gpt-4.1-mini'
        assert config.small_model == 'gpt-4.1-nano'
        assert config.temperature == 1.0
        assert config.max_tokens == 8192

    def test_anthropic_defaults(self):
        """Test Anthropic provider defaults are set correctly."""
        config = LLMProviderConfig(provider=LLMProvider.ANTHROPIC)

        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model == 'claude-sonnet-4-5-latest'
        assert config.small_model == 'claude-haiku-4-5-latest'

    def test_azure_openai_requires_base_url(self):
        """Test Azure OpenAI provider requires base_url."""
        with pytest.raises(ValueError, match='base_url is required'):
            LLMProviderConfig(provider=LLMProvider.AZURE_OPENAI)

    def test_azure_openai_valid_config(self):
        """Test valid Azure OpenAI configuration."""
        config = LLMProviderConfig(
            provider=LLMProvider.AZURE_OPENAI,
            base_url='https://my-resource.openai.azure.com',
            azure_deployment='gpt-4-deployment',
            api_key='test-key',
        )

        assert config.provider == LLMProvider.AZURE_OPENAI
        assert config.base_url == 'https://my-resource.openai.azure.com'
        assert config.azure_deployment == 'gpt-4-deployment'

    def test_litellm_requires_model(self):
        """Test LiteLLM provider requires litellm_model."""
        with pytest.raises(ValueError, match='litellm_model is required'):
            LLMProviderConfig(provider=LLMProvider.LITELLM)

    def test_litellm_valid_config(self):
        """Test valid LiteLLM configuration."""
        config = LLMProviderConfig(
            provider=LLMProvider.LITELLM,
            litellm_model='azure/gpt-4',
        )

        assert config.provider == LLMProvider.LITELLM
        assert config.litellm_model == 'azure/gpt-4'

    def test_custom_provider_requires_client_class(self):
        """Test custom provider requires custom_client_class."""
        with pytest.raises(ValueError, match='custom_client_class is required'):
            LLMProviderConfig(provider=LLMProvider.CUSTOM)

    def test_api_key_from_env(self, monkeypatch):
        """Test API key is loaded from environment."""
        monkeypatch.setenv('OPENAI_API_KEY', 'test-api-key')

        config = LLMProviderConfig(provider=LLMProvider.OPENAI)

        assert config.api_key == 'test-api-key'


class TestEmbedderConfig:
    def test_openai_defaults(self):
        """Test OpenAI embedder defaults."""
        config = EmbedderConfig(provider=EmbedderProvider.OPENAI)

        assert config.provider == EmbedderProvider.OPENAI
        assert config.model == 'text-embedding-3-small'
        assert config.dimensions == 1536

    def test_voyage_defaults(self):
        """Test Voyage AI embedder defaults."""
        config = EmbedderConfig(provider=EmbedderProvider.VOYAGE)

        assert config.provider == EmbedderProvider.VOYAGE
        assert config.model == 'voyage-3'
        assert config.dimensions == 1024

    def test_azure_requires_base_url(self):
        """Test Azure embedder requires base_url."""
        with pytest.raises(ValueError, match='base_url is required'):
            EmbedderConfig(provider=EmbedderProvider.AZURE_OPENAI)

    def test_custom_embedder_requires_class(self):
        """Test custom embedder requires custom_client_class."""
        with pytest.raises(ValueError, match='custom_client_class is required'):
            EmbedderConfig(provider=EmbedderProvider.CUSTOM)


class TestGraphitiConfig:
    def test_default_config(self):
        """Test default configuration."""
        config = GraphitiConfig()

        assert config.llm.provider == LLMProvider.OPENAI
        assert config.embedder.provider == EmbedderProvider.OPENAI
        assert config.database.provider == DatabaseProvider.NEO4J
        assert config.store_raw_episode_content is True

    def test_yaml_round_trip(self):
        """Test saving and loading configuration from YAML."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'test_config.yaml'

            # Create and save config
            original_config = GraphitiConfig(
                llm=LLMProviderConfig(
                    provider=LLMProvider.ANTHROPIC,
                    model='claude-sonnet-4-5-latest',
                    temperature=0.7,
                ),
                embedder=EmbedderConfig(
                    provider=EmbedderProvider.VOYAGE,
                    model='voyage-3',
                ),
                store_raw_episode_content=False,
            )

            original_config.to_yaml(config_path)

            # Load config back
            loaded_config = GraphitiConfig.from_yaml(config_path)

            assert loaded_config.llm.provider == LLMProvider.ANTHROPIC
            assert loaded_config.llm.model == 'claude-sonnet-4-5-latest'
            assert loaded_config.llm.temperature == 0.7
            assert loaded_config.embedder.provider == EmbedderProvider.VOYAGE
            assert loaded_config.embedder.model == 'voyage-3'
            assert loaded_config.store_raw_episode_content is False

    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            GraphitiConfig.from_yaml('nonexistent.yaml')

    def test_from_env_with_config_path(self, monkeypatch):
        """Test loading config from environment variable."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'graphiti.yaml'

            # Create config file
            config = GraphitiConfig(
                llm=LLMProviderConfig(provider=LLMProvider.GEMINI),
            )
            config.to_yaml(config_path)

            # Set environment variable
            monkeypatch.setenv('GRAPHITI_CONFIG_PATH', str(config_path))

            # Load from environment
            loaded_config = GraphitiConfig.from_env()

            assert loaded_config.llm.provider == LLMProvider.GEMINI

    def test_from_env_default_files(self):
        """Test loading from default config files."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / '.graphiti.yaml'

            # Create config in temp dir
            config = GraphitiConfig(
                llm=LLMProviderConfig(provider=LLMProvider.GROQ),
            )
            config.to_yaml(config_path)

            # Change to temp dir and load
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)
                loaded_config = GraphitiConfig.from_env()
                assert loaded_config.llm.provider == LLMProvider.GROQ
            finally:
                os.chdir(original_dir)

    def test_from_env_no_config_returns_defaults(self, monkeypatch):
        """Test loading from environment without config returns defaults."""
        # Make sure env var is not set
        monkeypatch.delenv('GRAPHITI_CONFIG_PATH', raising=False)

        config = GraphitiConfig.from_env()

        # Should return default config
        assert config.llm.provider == LLMProvider.OPENAI
        assert config.embedder.provider == EmbedderProvider.OPENAI


class TestDatabaseConfig:
    def test_neo4j_config(self):
        """Test Neo4j database configuration."""
        config = DatabaseConfig(
            provider=DatabaseProvider.NEO4J,
            uri='bolt://localhost:7687',
            user='neo4j',
            password='password',
            database='graphiti',
        )

        assert config.provider == DatabaseProvider.NEO4J
        assert config.uri == 'bolt://localhost:7687'
        assert config.user == 'neo4j'
        assert config.database == 'graphiti'

    def test_custom_database_requires_driver_class(self):
        """Test custom database provider requires custom_driver_class."""
        with pytest.raises(ValueError, match='custom_driver_class is required'):
            DatabaseConfig(provider=DatabaseProvider.CUSTOM)


class TestRerankerConfig:
    def test_default_config(self):
        """Test default reranker configuration."""
        config = RerankerConfig()

        assert config.provider == RerankerProvider.OPENAI

    def test_api_key_from_env(self, monkeypatch):
        """Test reranker API key from environment."""
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

        config = RerankerConfig(provider=RerankerProvider.OPENAI)

        assert config.api_key == 'test-key'
