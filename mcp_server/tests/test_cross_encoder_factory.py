from config.schema import (
    AnthropicProviderConfig,
    GeminiProviderConfig,
    LLMConfig,
    LLMProvidersConfig,
)
from services.factories import CrossEncoderFactory, NoOpCrossEncoderClient


def test_gemini_cross_encoder_does_not_default_to_openai():
    config = LLMConfig(
        provider='gemini',
        model='gemini-2.5-flash-lite',
        providers=LLMProvidersConfig(gemini=GeminiProviderConfig(api_key='test-key')),
    )

    client = CrossEncoderFactory.create(config)
    assert client.__class__.__name__ != 'OpenAIRerankerClient'


def test_anthropic_cross_encoder_uses_noop():
    config = LLMConfig(
        provider='anthropic',
        model='claude-3-5-sonnet-latest',
        providers=LLMProvidersConfig(anthropic=AnthropicProviderConfig(api_key='test-key')),
    )

    client = CrossEncoderFactory.create(config)
    assert isinstance(client, NoOpCrossEncoderClient)
