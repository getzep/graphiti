"""Tests for LLMClientFactory configuration forwarding."""

from unittest.mock import MagicMock, patch

import pytest

from config.schema import LLMConfig, LLMProvidersConfig, OpenAIProviderConfig


def _make_openai_llm_config(
    model: str = 'gpt-4o',
    small_model: str | None = None,
    api_url: str = 'https://api.openai.com/v1',
) -> LLMConfig:
    """Helper to build an LLMConfig with an OpenAI provider."""
    return LLMConfig(
        provider='openai',
        model=model,
        small_model=small_model,
        providers=LLMProvidersConfig(
            openai=OpenAIProviderConfig(api_key='sk-test', api_url=api_url)
        ),
    )


@pytest.fixture
def capture_core_config():
    """Patch CoreLLMConfig and OpenAIClient, returning the captured kwargs."""
    captured = {}

    def fake_llm_config(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with (
        patch('graphiti_core.llm_client.config.LLMConfig', side_effect=fake_llm_config),
        patch('graphiti_core.llm_client.OpenAIClient', return_value=MagicMock()),
    ):
        yield captured


class TestLLMClientFactoryBaseUrl:
    """Tests for forwarding api_url to CoreLLMConfig as base_url."""

    def test_custom_base_url_is_forwarded(self, capture_core_config):
        """When api_url is a custom endpoint, base_url must be passed to CoreLLMConfig."""
        from services.factories import LLMClientFactory

        config = _make_openai_llm_config(api_url='http://localhost:4000/v1')
        LLMClientFactory.create(config)

        assert capture_core_config.get('base_url') == 'http://localhost:4000/v1'

    def test_default_openai_url_is_normalised_to_none(self, capture_core_config):
        """When api_url is the OpenAI default, base_url should be None."""
        from services.factories import LLMClientFactory

        config = _make_openai_llm_config(api_url='https://api.openai.com/v1')
        LLMClientFactory.create(config)

        assert capture_core_config.get('base_url') is None


class TestLLMClientFactorySmallModel:
    """Tests for small_model config forwarding."""

    def test_small_model_field_defaults_to_none(self):
        """small_model should default to None when not configured."""
        config = LLMConfig(provider='openai', model='gpt-4o')
        assert config.small_model is None

    def test_explicit_small_model_is_forwarded(self, capture_core_config):
        """When small_model is set in config, it must be passed to CoreLLMConfig."""
        from services.factories import LLMClientFactory

        config = _make_openai_llm_config(model='gpt-4o', small_model='gpt-4o-mini')
        LLMClientFactory.create(config)

        assert capture_core_config.get('model') == 'gpt-4o'
        assert capture_core_config.get('small_model') == 'gpt-4o-mini'

    def test_small_model_falls_back_to_model_when_not_set(self, capture_core_config):
        """When small_model is not configured, it should fall back to model."""
        from services.factories import LLMClientFactory

        config = _make_openai_llm_config(model='gpt-4o', small_model=None)
        LLMClientFactory.create(config)

        assert capture_core_config.get('small_model') == 'gpt-4o'


class TestLLMClientFactoryErrors:
    """Error-path tests for LLMClientFactory."""

    def test_missing_openai_provider_raises(self):
        """Factory must raise ValueError when OpenAI provider config is absent."""
        from services.factories import LLMClientFactory

        config = LLMConfig(provider='openai', model='gpt-4o')

        with pytest.raises(ValueError, match='OpenAI provider configuration not found'):
            LLMClientFactory.create(config)
