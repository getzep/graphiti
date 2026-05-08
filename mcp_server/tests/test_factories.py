from config.schema import (
    AnthropicProviderConfig,
    GeminiProviderConfig,
    GroqProviderConfig,
    LLMConfig,
    LLMProvidersConfig,
)
from services import factories


class DummyClient:
    def __init__(self, config, **kwargs):
        self.config = config
        self.temperature = config.temperature
        self.kwargs = kwargs


def _make_llm_config(provider: str, temperature: float | None) -> LLMConfig:
    providers = LLMProvidersConfig()
    if provider == 'anthropic':
        providers.anthropic = AnthropicProviderConfig(api_key='test-anthropic-key')
        model = 'claude-sonnet-4-5-20250929'
    elif provider == 'gemini':
        providers.gemini = GeminiProviderConfig(api_key='test-gemini-key')
        model = 'gemini-2.5-pro'
    elif provider == 'groq':
        providers.groq = GroqProviderConfig(
            api_key='test-groq-key',
            api_url='https://api.groq.com/openai/v1',
        )
        model = 'llama-3.3-70b-versatile'
    else:
        raise ValueError(f'Unsupported provider for test: {provider}')

    return LLMConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=2048,
        providers=providers,
    )


def test_anthropic_uses_core_default_temperature_when_unspecified(monkeypatch):
    monkeypatch.setattr(factories, 'HAS_ANTHROPIC', True)
    monkeypatch.setattr(factories, 'AnthropicClient', DummyClient, raising=False)

    client = factories.LLMClientFactory.create(_make_llm_config('anthropic', temperature=None))

    assert client.temperature == 1


def test_gemini_uses_core_default_temperature_when_unspecified(monkeypatch):
    monkeypatch.setattr(factories, 'HAS_GEMINI', True)
    monkeypatch.setattr(factories, 'GeminiClient', DummyClient, raising=False)

    client = factories.LLMClientFactory.create(_make_llm_config('gemini', temperature=None))

    assert client.temperature == 1


def test_groq_uses_core_default_temperature_when_unspecified(monkeypatch):
    monkeypatch.setattr(factories, 'HAS_GROQ', True)
    monkeypatch.setattr(factories, 'GroqClient', DummyClient, raising=False)

    client = factories.LLMClientFactory.create(_make_llm_config('groq', temperature=None))

    assert client.temperature == 1


def test_explicit_temperature_is_preserved_for_non_openai_providers(monkeypatch):
    monkeypatch.setattr(factories, 'HAS_ANTHROPIC', True)
    monkeypatch.setattr(factories, 'AnthropicClient', DummyClient, raising=False)

    client = factories.LLMClientFactory.create(_make_llm_config('anthropic', temperature=0.42))

    assert client.temperature == 0.42
