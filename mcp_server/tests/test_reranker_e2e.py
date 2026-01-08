"""End-to-end tests for reranker configuration."""

import os
import sys
from pathlib import Path

import pytest

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestRerankerE2E:
    """End-to-end tests for reranker configuration."""

    def test_load_config_local_yaml(self):
        """Test loading reranker config from config-local.yaml."""
        from config.schema import GraphitiConfig

        # Set config path to config-local.yaml
        config_path = Path(__file__).parent.parent / 'config' / 'config-local.yaml'
        os.environ['CONFIG_PATH'] = str(config_path)

        config = GraphitiConfig()

        # Verify reranker config is loaded
        assert config.reranker is not None
        assert config.reranker.enabled is True
        assert config.reranker.type == 'cross_encoder'
        assert config.reranker.provider == 'openai'
        assert config.reranker.model == 'qwen3-rerank'
        assert config.reranker.providers.openai is not None
        assert config.reranker.providers.openai.api_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1'

    def test_reranker_factory_with_config_local(self):
        """Test RerankerFactory creates client from config-local.yaml."""
        from config.schema import GraphitiConfig
        from services.factories import RerankerFactory

        # Set config path to config-local.yaml
        config_path = Path(__file__).parent.parent / 'config' / 'config-local.yaml'
        os.environ['CONFIG_PATH'] = str(config_path)

        config = GraphitiConfig()

        # Create reranker client
        try:
            reranker_client = RerankerFactory.create(config.reranker)
            # If successful, should have a rank method
            if reranker_client is not None:
                assert hasattr(reranker_client, 'rank')
                assert callable(getattr(reranker_client, 'rank'))
        except Exception as e:
            # If it fails, it should be a clear error message
            error_msg = str(e).lower()
            # Should fail with API key validation or import error, not silently
            assert 'api key' in error_msg or 'not available' in error_msg or 'import' in error_msg

    @pytest.mark.asyncio
    async def test_graphiti_service_initialization_with_reranker(self):
        """Test GraphitiService initialization with reranker config."""
        from config.schema import GraphitiConfig
        from graphiti_mcp_server import GraphitiService

        # Set config path to config-local.yaml
        config_path = Path(__file__).parent.parent / 'config' / 'config-local.yaml'
        os.environ['CONFIG_PATH'] = str(config_path)

        config = GraphitiConfig()

        # Create service (but don't fully initialize to avoid DB connection)
        service = GraphitiService(config, semaphore_limit=1)

        # Verify config is set
        assert service.config.reranker.enabled is True
        assert service.config.reranker.type == 'cross_encoder'

    def test_reranker_client_creation_with_qwen3_rerank(self):
        """Test that reranker client is created correctly with qwen3-rerank model."""
        from config.schema import GraphitiConfig
        from services.factories import RerankerFactory

        # Set config path to config-local.yaml
        config_path = Path(__file__).parent.parent / 'config' / 'config-local.yaml'
        os.environ['CONFIG_PATH'] = str(config_path)

        config = GraphitiConfig()

        # Verify model is qwen3-rerank
        assert config.reranker.model == 'qwen3-rerank'
        assert config.reranker.providers.openai.api_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1'

        # Create reranker client
        reranker = RerankerFactory.create(config.reranker)

        # Should create OpenAIRerankerClient
        assert reranker is not None
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        assert isinstance(reranker, OpenAIRerankerClient)

        # Verify the client has the correct model configured
        assert reranker.config.model == 'qwen3-rerank'
        assert 'dashscope' in reranker.config.base_url or reranker.config.base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
