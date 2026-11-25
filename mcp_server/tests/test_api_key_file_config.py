"""Unit tests for file-backed provider API keys."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.schema import (  # noqa: E402
    AnthropicProviderConfig,
    AzureOpenAIProviderConfig,
    GeminiProviderConfig,
    GraphitiConfig,
    GroqProviderConfig,
    OpenAIProviderConfig,
    VoyageProviderConfig,
)

PROVIDER_CONFIGS = (
    OpenAIProviderConfig,
    AzureOpenAIProviderConfig,
    AnthropicProviderConfig,
    GeminiProviderConfig,
    GroqProviderConfig,
    VoyageProviderConfig,
)


@pytest.mark.parametrize('provider_config', PROVIDER_CONFIGS)
def test_api_key_is_loaded_from_file(provider_config, tmp_path):
    api_key_file = tmp_path / 'api-key'
    api_key_file.write_text('secret-from-file\n', encoding='utf-8')

    config = provider_config(api_key_file=api_key_file)

    assert config.api_key == 'secret-from-file'


def test_api_key_file_path_supports_environment_expansion(tmp_path, monkeypatch):
    api_key_file = tmp_path / 'api-key'
    api_key_file.write_text('secret-from-file\n', encoding='utf-8')
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(
        """
llm:
  providers:
    openai:
      api_key_file: ${TEST_OPENAI_API_KEY_FILE}
embedder:
  providers:
    openai:
      api_key_file: ${TEST_OPENAI_API_KEY_FILE}
""",
        encoding='utf-8',
    )
    monkeypatch.setenv('CONFIG_PATH', str(config_file))
    monkeypatch.setenv('TEST_OPENAI_API_KEY_FILE', str(api_key_file))

    config = GraphitiConfig()

    assert config.llm.providers.openai.api_key == 'secret-from-file'
    assert config.embedder.providers.openai.api_key == 'secret-from-file'


def test_inline_api_key_and_api_key_file_are_mutually_exclusive(tmp_path):
    api_key_file = tmp_path / 'api-key'
    api_key_file.write_text('secret-from-file', encoding='utf-8')

    with pytest.raises(ValidationError, match='api_key and api_key_file are mutually exclusive'):
        OpenAIProviderConfig(api_key='inline-secret', api_key_file=api_key_file)


def test_missing_api_key_file_has_clear_validation_error(tmp_path):
    api_key_file = tmp_path / 'missing-api-key'

    with pytest.raises(ValidationError, match='Unable to read api_key_file'):
        OpenAIProviderConfig(api_key_file=api_key_file)


def test_empty_api_key_file_has_clear_validation_error(tmp_path):
    api_key_file = tmp_path / 'api-key'
    api_key_file.write_text('\n', encoding='utf-8')

    with pytest.raises(ValidationError, match='api_key_file must not be empty'):
        OpenAIProviderConfig(api_key_file=api_key_file)
