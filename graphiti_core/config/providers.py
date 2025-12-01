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

from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers for Graphiti."""

    OPENAI = 'openai'
    AZURE_OPENAI = 'azure_openai'
    ANTHROPIC = 'anthropic'
    GEMINI = 'gemini'
    GROQ = 'groq'
    LITELLM = 'litellm'  # Unified interface for multiple providers
    CUSTOM = 'custom'


class EmbedderProvider(str, Enum):
    """Supported embedding providers for Graphiti."""

    OPENAI = 'openai'
    AZURE_OPENAI = 'azure_openai'
    VOYAGE = 'voyage'
    GEMINI = 'gemini'
    CUSTOM = 'custom'


class DatabaseProvider(str, Enum):
    """Supported graph database providers for Graphiti."""

    NEO4J = 'neo4j'
    FALKORDB = 'falkordb'
    NEPTUNE = 'neptune'
    CUSTOM = 'custom'


class RerankerProvider(str, Enum):
    """Supported reranker providers for Graphiti."""

    OPENAI = 'openai'
    AZURE_OPENAI = 'azure_openai'
    CUSTOM = 'custom'


# Provider-specific default models
DEFAULT_MODELS = {
    LLMProvider.OPENAI: {
        'model': 'gpt-4.1-mini',
        'small_model': 'gpt-4.1-nano',
    },
    LLMProvider.AZURE_OPENAI: {
        'model': 'gpt-4.1-mini',
        'small_model': 'gpt-4.1-nano',
    },
    LLMProvider.ANTHROPIC: {
        'model': 'claude-sonnet-4-5-latest',
        'small_model': 'claude-haiku-4-5-latest',
    },
    LLMProvider.GEMINI: {
        'model': 'gemini-2.5-flash',
        'small_model': 'gemini-2.5-flash',
    },
    LLMProvider.GROQ: {
        'model': 'llama-3.1-70b-versatile',
        'small_model': 'llama-3.1-8b-instant',
    },
}

DEFAULT_EMBEDDINGS = {
    EmbedderProvider.OPENAI: {
        'model': 'text-embedding-3-small',
        'dimensions': 1536,
    },
    EmbedderProvider.AZURE_OPENAI: {
        'model': 'text-embedding-3-small',
        'dimensions': 1536,
    },
    EmbedderProvider.VOYAGE: {
        'model': 'voyage-3',
        'dimensions': 1024,
    },
    EmbedderProvider.GEMINI: {
        'model': 'text-embedding-004',
        'dimensions': 768,
    },
}
