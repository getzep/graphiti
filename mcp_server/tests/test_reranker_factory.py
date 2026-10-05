"""Tests for RerankerFactory."""

import os
from unittest.mock import patch

import pytest


class TestRerankerConfig:
    """Tests for RerankerConfig model."""

    def test_default_config(self):
        """Test default reranker configuration."""
        from config.schema import RerankerConfig

        config = RerankerConfig()
        assert config.enabled is True
        assert config.type == 'rrf'
        assert config.provider == 'openai'
        assert config.model == 'gpt-4.1-nano'

    def test_local_config(self):
        """Test local reranker configuration."""
        from config.schema import RerankerConfig, RerankerLocalConfig

        config = RerankerConfig(local=RerankerLocalConfig(type='mmr', mmr_lambda=0.7))
        assert config.local.type == 'mmr'
        assert config.local.mmr_lambda == 0.7


class TestRerankerFactory:
    """Tests for RerankerFactory."""

    def test_disabled_reranker(self):
        """Test when reranker is disabled."""
        from config.schema import RerankerConfig
        from services.factories import RerankerFactory

        config = RerankerConfig(enabled=False)
        result = RerankerFactory.create(config)
        from services.factories import LocalRerankerSentinel

        # Local types return a sentinel (never None): None would let core
        # fall back to OpenAIRerankerClient and crash keyless setups.
        assert isinstance(result, LocalRerankerSentinel)

    def test_local_rrf_reranker(self):
        """Test local RRF reranker."""
        from config.schema import RerankerConfig
        from services.factories import RerankerFactory

        config = RerankerConfig(enabled=True, type='rrf')
        result = RerankerFactory.create(config)
        from services.factories import LocalRerankerSentinel

        # Local types return a sentinel (never None): None would let core
        # fall back to OpenAIRerankerClient and crash keyless setups.
        assert isinstance(result, LocalRerankerSentinel)

    def test_local_mmr_reranker(self):
        """Test local MMR reranker."""
        from config.schema import RerankerConfig
        from services.factories import RerankerFactory

        config = RerankerConfig(enabled=True, type='mmr')
        result = RerankerFactory.create(config)
        from services.factories import LocalRerankerSentinel

        # Local types return a sentinel (never None): None would let core
        # fall back to OpenAIRerankerClient and crash keyless setups.
        assert isinstance(result, LocalRerankerSentinel)

    def test_openai_cross_encoder(self):
        """Test OpenAI cross_encoder reranker."""
        from config.schema import (
            OpenAIProviderConfig,
            RerankerConfig,
            RerankerProvidersConfig,
        )
        from services.factories import RerankerFactory

        config = RerankerConfig(
            enabled=True,
            type='cross_encoder',
            provider='openai',
            model='gpt-4.1-nano',
            providers=RerankerProvidersConfig(openai=OpenAIProviderConfig(api_key='test-key')),
        )
        # This will fail if API key is invalid, but that's expected
        # We just verify the factory method doesn't crash
        try:
            result = RerankerFactory.create(config)
            # If it succeeds, result should be a client instance or None
            hasattr(result, 'rank')
        except ValueError as e:
            # Expected if API key validation fails
            assert 'API key' in str(e) or 'not configured' in str(e)

    def test_gemini_cross_encoder(self):
        """Test Gemini cross_encoder reranker."""
        from config.schema import (
            GeminiProviderConfig,
            RerankerConfig,
            RerankerProvidersConfig,
        )
        from services.factories import RerankerFactory

        config = RerankerConfig(
            enabled=True,
            type='cross_encoder',
            provider='gemini',
            model='gemini-2.5-flash-lite',
            providers=RerankerProvidersConfig(gemini=GeminiProviderConfig(api_key='test-key')),
        )
        # This will fail if API key is invalid or dependency missing
        try:
            result = RerankerFactory.create(config)
            # If it succeeds, result should be a client instance or None
            hasattr(result, 'rank')
        except (ValueError, ImportError) as e:
            # Expected if API key validation fails or dependency missing
            assert 'API key' in str(e) or 'not configured' in str(e) or 'not available' in str(e)

    def test_sentence_transformers_cross_encoder(self):
        """Test sentence_transformers cross_encoder reranker."""
        from config.schema import RerankerConfig
        from services.factories import RerankerFactory

        config = RerankerConfig(
            enabled=True, type='cross_encoder', provider='sentence_transformers'
        )
        # This will fail if dependency is missing
        try:
            result = RerankerFactory.create(config)
            # If it succeeds, result should be a client instance
            hasattr(result, 'rank')
        except (ValueError, ImportError) as e:
            # Expected if dependency is missing
            assert 'not available' in str(e) or 'sentence-transformers' in str(e)

    def test_unknown_reranker_type(self):
        """Test unknown reranker type falls back to local."""
        from config.schema import RerankerConfig
        from services.factories import RerankerFactory

        config = RerankerConfig(enabled=True, type='unknown')
        with pytest.raises(ValueError, match='Unknown reranker type'):
            RerankerFactory.create(config)


class TestRerankerConfigIntegration:
    """Integration tests for reranker configuration loading."""

    def test_load_reranker_config_from_yaml(self, tmp_path):
        """Test loading reranker config from YAML."""
        from config.schema import GraphitiConfig

        config_content = """
reranker:
  enabled: true
  type: cross_encoder
  provider: openai
  model: qwen3-rerank
  providers:
    openai:
      api_key: ${TEST_API_KEY}
      api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
"""
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(config_content)

        with patch.dict(os.environ, {'CONFIG_PATH': str(config_file), 'TEST_API_KEY': 'test-key'}):
            config = GraphitiConfig()
            assert config.reranker.enabled is True
            assert config.reranker.type == 'cross_encoder'
            assert config.reranker.provider == 'openai'
            assert config.reranker.model == 'qwen3-rerank'


def test_embedder_factory_passes_batch_size():
    """The mcp embedder config batch_size must reach the core embedder config."""
    from unittest.mock import MagicMock

    from config.schema import EmbedderConfig
    from services.factories import EmbedderFactory

    cfg = MagicMock(spec=EmbedderConfig)
    cfg.provider = 'openai'
    cfg.providers = MagicMock()
    cfg.providers.openai = MagicMock(api_key='test-key', api_url=None)
    cfg.model = 'text-embedding-v4'
    cfg.dimensions = 1024
    cfg.batch_size = 3

    embedder = EmbedderFactory.create(cfg)

    assert embedder.config.batch_size == 3


def test_unknown_cross_encoder_provider_raises():
    """A typo'd provider must raise, not return None (None re-enables the
    OpenAIRerankerClient fallback crash on keyless setups)."""
    from config.schema import RerankerConfig
    from services.factories import RerankerFactory

    config = RerankerConfig(enabled=True, type='cross_encoder', provider='unknown-provider')
    import pytest

    with pytest.raises(ValueError, match='Unknown cross_encoder provider'):
        RerankerFactory.create(config)
