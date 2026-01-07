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

import pytest

from graphiti_core.config import (
    DatabaseConfig,
    EmbedderConfig,
    EmbedderProvider,
    LLMProvider,
    LLMProviderConfig,
    RerankerConfig,
)
from graphiti_core.config.factory import (
    create_database_driver,
    create_embedder,
    create_llm_client,
    create_reranker,
)
from graphiti_core.config.providers import DatabaseProvider, RerankerProvider
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder
from graphiti_core.llm_client.openai_client import OpenAIClient


class TestCreateLLMClient:
    def test_create_openai_client(self, monkeypatch):
        """Test creating OpenAI LLM client."""
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

        config = LLMProviderConfig(provider=LLMProvider.OPENAI)
        client = create_llm_client(config)

        assert isinstance(client, OpenAIClient)
        assert client.model == 'gpt-4.1-mini'
        assert client.small_model == 'gpt-4.1-nano'

    def test_create_azure_openai_client(self, monkeypatch):
        """Test creating Azure OpenAI LLM client."""
        from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient

        monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'test-key')

        config = LLMProviderConfig(
            provider=LLMProvider.AZURE_OPENAI,
            base_url='https://my-resource.openai.azure.com',
            azure_deployment='gpt-4-deployment',
        )
        client = create_llm_client(config)

        assert isinstance(client, AzureOpenAILLMClient)

    def test_create_anthropic_client_missing_dep(self):
        """Test creating Anthropic client without dependency raises error."""
        config = LLMProviderConfig(provider=LLMProvider.ANTHROPIC)

        # This test will pass if anthropic is installed (in dev env)
        # or raise ImportError if not installed
        try:
            client = create_llm_client(config)
            from graphiti_core.llm_client.anthropic_client import AnthropicClient

            assert isinstance(client, AnthropicClient)
        except ImportError as e:
            assert 'anthropic' in str(e).lower()

    def test_create_litellm_client_missing_dep(self):
        """Test creating LiteLLM client without dependency raises error."""
        config = LLMProviderConfig(
            provider=LLMProvider.LITELLM,
            litellm_model='gpt-4',
        )

        # This will raise ImportError if litellm is not installed
        try:
            client = create_llm_client(config)
            from graphiti_core.llm_client.litellm_client import LiteLLMClient

            assert isinstance(client, LiteLLMClient)
        except ImportError as e:
            assert 'litellm' in str(e).lower()

    def test_unsupported_provider_raises_error(self):
        """Test unsupported provider raises ValueError."""
        # We need to bypass pydantic validation to test this
        config = LLMProviderConfig(provider=LLMProvider.OPENAI)
        config.provider = 'unsupported'  # type: ignore

        with pytest.raises(ValueError, match='Unsupported LLM provider'):
            create_llm_client(config)


class TestCreateEmbedder:
    def test_create_openai_embedder(self, monkeypatch):
        """Test creating OpenAI embedder."""
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

        config = EmbedderConfig(provider=EmbedderProvider.OPENAI)
        embedder = create_embedder(config)

        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.config.embedding_model == 'text-embedding-3-small'

    def test_create_voyage_embedder_missing_dep(self):
        """Test creating Voyage embedder without dependency."""
        config = EmbedderConfig(provider=EmbedderProvider.VOYAGE)

        try:
            embedder = create_embedder(config)
            from graphiti_core.embedder.voyage import VoyageEmbedder

            assert isinstance(embedder, VoyageEmbedder)
        except ImportError as e:
            assert 'voyageai' in str(e).lower()

    def test_create_azure_openai_embedder(self, monkeypatch):
        """Test creating Azure OpenAI embedder."""
        from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient

        monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'test-key')

        config = EmbedderConfig(
            provider=EmbedderProvider.AZURE_OPENAI,
            base_url='https://my-resource.openai.azure.com',
            azure_deployment='embedding-deployment',
        )
        embedder = create_embedder(config)

        assert isinstance(embedder, AzureOpenAIEmbedderClient)


class TestCreateReranker:
    def test_create_openai_reranker(self):
        """Test creating OpenAI reranker."""
        config = RerankerConfig(provider=RerankerProvider.OPENAI)
        reranker = create_reranker(config)

        assert isinstance(reranker, OpenAIRerankerClient)


class TestCreateDatabaseDriver:
    def test_create_neo4j_driver(self):
        """Test creating Neo4j driver."""
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        config = DatabaseConfig(
            provider=DatabaseProvider.NEO4J,
            uri='bolt://localhost:7687',
            user='neo4j',
            password='password',
        )
        driver = create_database_driver(config)

        assert isinstance(driver, Neo4jDriver)

    def test_neo4j_driver_missing_uri(self):
        """Test Neo4j driver without URI raises error."""
        config = DatabaseConfig(provider=DatabaseProvider.NEO4J)

        with pytest.raises(ValueError, match='uri is required'):
            create_database_driver(config)

    def test_create_falkordb_driver_missing_dep(self):
        """Test creating FalkorDB driver without dependency."""
        config = DatabaseConfig(
            provider=DatabaseProvider.FALKORDB,
            uri='redis://localhost:6379',
        )

        try:
            driver = create_database_driver(config)
            from graphiti_core.driver.falkor_driver import FalkorDriver

            assert isinstance(driver, FalkorDriver)
        except ImportError as e:
            assert 'falkordb' in str(e).lower()
