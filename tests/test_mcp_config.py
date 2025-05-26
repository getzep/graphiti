"""
Test MCP server configuration, especially OpenRouter integration.
"""

import os
import pytest
from unittest.mock import patch

# Import the configuration classes from the MCP server
import sys
sys.path.append('mcp_server')

from graphiti_mcp_server import GraphitiLLMConfig, GraphitiEmbedderConfig, GraphitiConfig


class TestGraphitiLLMConfig:
    """Test the LLM configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GraphitiLLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4.1-mini"
        assert config.small_model == "gpt-4.1-nano"
        assert config.temperature == 0.0
        assert config.base_url is None
        assert config.api_key is None

    @patch.dict(os.environ, {
        'LLM_PROVIDER': 'openrouter',
        'OPENROUTER_API_KEY': 'sk-or-v1-test-key',
        'MODEL_NAME': 'meta-llama/llama-3.1-70b-instruct',
        'SMALL_MODEL_NAME': 'meta-llama/llama-3.1-8b-instruct',
        'LLM_TEMPERATURE': '0.2'
    })
    def test_openrouter_config_from_env(self):
        """Test OpenRouter configuration from environment variables."""
        config = GraphitiLLMConfig.from_env()
        assert config.provider == "openrouter"
        assert config.api_key == "sk-or-v1-test-key"
        assert config.model == "meta-llama/llama-3.1-70b-instruct"
        assert config.small_model == "meta-llama/llama-3.1-8b-instruct"
        assert config.temperature == 0.2
        assert config.base_url == "https://openrouter.ai/api/v1"

    @patch.dict(os.environ, {
        'LLM_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'sk-test-openai-key',
        'MODEL_NAME': 'gpt-4o',
        'OPENAI_BASE_URL': 'https://custom.openai.com/v1'
    })
    def test_openai_config_from_env(self):
        """Test OpenAI configuration from environment variables."""
        config = GraphitiLLMConfig.from_env()
        assert config.provider == "openai"
        assert config.api_key == "sk-test-openai-key"
        assert config.model == "gpt-4o"
        assert config.base_url == "https://custom.openai.com/v1"

    @patch.dict(os.environ, {
        'LLM_PROVIDER': 'azure',
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_OPENAI_DEPLOYMENT_NAME': 'gpt-4o-deployment',
        'OPENAI_API_KEY': 'azure-test-key'
    })
    def test_azure_config_from_env(self):
        """Test Azure OpenAI configuration from environment variables."""
        config = GraphitiLLMConfig.from_env()
        assert config.provider == "azure"
        assert config.api_key == "azure-test-key"
        assert config.azure_openai_endpoint == "https://test.openai.azure.com/"

    def test_api_key_selection_openrouter(self):
        """Test that OpenRouter provider uses OPENROUTER_API_KEY."""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openrouter',
            'OPENROUTER_API_KEY': 'sk-or-v1-openrouter-key',
            'OPENAI_API_KEY': 'sk-openai-key'
        }):
            config = GraphitiLLMConfig.from_env()
            assert config.api_key == "sk-or-v1-openrouter-key"

    def test_api_key_selection_openai(self):
        """Test that OpenAI provider uses OPENAI_API_KEY."""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'OPENROUTER_API_KEY': 'sk-or-v1-openrouter-key',
            'OPENAI_API_KEY': 'sk-openai-key'
        }):
            config = GraphitiLLMConfig.from_env()
            assert config.api_key == "sk-openai-key"


class TestGraphitiEmbedderConfig:
    """Test the embedder configuration class."""
    
    def test_default_config(self):
        """Test default embedder configuration."""
        config = GraphitiEmbedderConfig()
        assert config.model == "text-embedding-3-small"
        assert config.base_url is None
        assert config.api_key is None

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-openai-key',
        'EMBEDDER_MODEL_NAME': 'text-embedding-3-large',
        'EMBEDDER_BASE_URL': 'https://custom.openai.com/v1'
    })
    def test_embedder_config_from_env(self):
        """Test embedder configuration from environment variables."""
        config = GraphitiEmbedderConfig.from_env()
        assert config.api_key == "sk-openai-key"
        assert config.model == "text-embedding-3-large"
        assert config.base_url == "https://custom.openai.com/v1"

    @patch.dict(os.environ, {
        'EMBEDDER_PROVIDER': 'openrouter',
        'OPENROUTER_API_KEY': 'sk-or-key',
        'OPENAI_API_KEY': 'sk-openai-key'
    })
    def test_openrouter_embedder_fallback(self):
        """Test that OpenRouter embedder falls back to OpenAI (since OpenRouter doesn't support embeddings)."""
        config = GraphitiEmbedderConfig.from_env()
        # Should fall back to openai since OpenRouter doesn't support embeddings
        assert config.provider == "openai"
        # Should use OPENAI_API_KEY, not OPENROUTER_API_KEY
        assert config.api_key == "sk-openai-key"

    @patch.dict(os.environ, {
        'EMBEDDER_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'sk-openai-key'
    })
    def test_explicit_openai_embedder(self):
        """Test explicit OpenAI embedder configuration."""
        config = GraphitiEmbedderConfig.from_env()
        assert config.provider == "openai"
        assert config.api_key == "sk-openai-key"


class TestGraphitiConfig:
    """Test the main Graphiti configuration class."""
    
    @patch.dict(os.environ, {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USER': 'neo4j',
        'NEO4J_PASSWORD': 'password',
        'LLM_PROVIDER': 'openrouter',
        'OPENROUTER_API_KEY': 'sk-or-v1-test-key',
        'MODEL_NAME': 'meta-llama/llama-3.1-70b-instruct'
    })
    def test_full_config_from_env(self):
        """Test full configuration from environment variables."""
        config = GraphitiConfig.from_env()
        assert config.neo4j.uri == "bolt://localhost:7687"
        assert config.neo4j.user == "neo4j"
        assert config.neo4j.password == "password"
        assert config.llm.provider == "openrouter"
        assert config.llm.api_key == "sk-or-v1-test-key"
        assert config.llm.model == "meta-llama/llama-3.1-70b-instruct"

    @patch.dict(os.environ, {
        'LLM_PROVIDER': 'openrouter',
        'OPENROUTER_API_KEY': 'sk-or-v1-llm-key',
        'MODEL_NAME': 'meta-llama/llama-3.1-70b-instruct',
        'EMBEDDER_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'sk-openai-embedder-key',
        'EMBEDDER_MODEL_NAME': 'text-embedding-3-small'
    })
    def test_mixed_provider_config(self):
        """Test mixed provider configuration (OpenRouter for LLM, OpenAI for embeddings)."""
        config = GraphitiConfig.from_env()
        assert config.llm.provider == "openrouter"
        assert config.llm.api_key == "sk-or-v1-llm-key"
        assert config.embedder.provider == "openai"
        assert config.embedder.api_key == "sk-openai-embedder-key"


if __name__ == "__main__":
    pytest.main([__file__]) 
