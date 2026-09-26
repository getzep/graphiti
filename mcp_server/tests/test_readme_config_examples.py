#!/usr/bin/env python3
"""Regression fence: the config examples in mcp_server/README.md must match the real schema.

The YAML is extracted from README.md at runtime (never hand-copied) so the documented
examples cannot silently drift away from `config.schema` / `services.factories` again.
See getzep/graphiti#1556.
"""

import re
import sys
from pathlib import Path

import pytest
import yaml

# Add the src directory to the path (mirrors the other factory tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.embedder.openai import OpenAIEmbedder
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from config.schema import EmbedderConfig, EmbedderProvidersConfig, LLMConfig
from services.factories import EmbedderFactory, LLMClientFactory

README_PATH = Path(__file__).parent.parent / 'README.md'

OLLAMA_SECTION_RE = re.compile(
    r'^###\s+Using Ollama for Local LLM\s*$.*?^```yaml\s*$(?P<yaml>.*?)^```\s*$',
    re.MULTILINE | re.DOTALL,
)

YAML_FENCE_RE = re.compile(r'^```yaml\s*$(?P<yaml>.*?)^```\s*$', re.MULTILINE | re.DOTALL)

OLLAMA_API_URL = 'http://localhost:11434/v1'


def _readme_text() -> str:
    """Read README.md relative to this test file, not the current working directory."""
    assert README_PATH.is_file(), f'README not found at {README_PATH}'
    return README_PATH.read_text(encoding='utf-8')


def _ollama_example() -> dict:
    """Extract and parse the YAML block from the 'Using Ollama for Local LLM' section."""
    match = OLLAMA_SECTION_RE.search(_readme_text())
    assert match is not None, 'no yaml fence found under "### Using Ollama for Local LLM"'
    data = yaml.safe_load(match.group('yaml'))
    assert isinstance(data, dict), 'Ollama example did not parse into a mapping'
    return data


@pytest.fixture
def ollama_example() -> dict:
    """Provide the parsed README Ollama config example."""
    return _ollama_example()


class TestReadmeOllamaExample:
    """The documented Ollama example must build working clients through the real factories."""

    def test_example_declares_llm_and_embedder_sections(self, ollama_example):
        """The extractor found a block that actually configures both an LLM and an embedder."""
        assert 'llm' in ollama_example, f'no llm section in {sorted(ollama_example)}'
        assert 'embedder' in ollama_example, f'no embedder section in {sorted(ollama_example)}'

    def test_llm_example_nests_ollama_endpoint_under_openai_provider(self, ollama_example):
        """The Ollama endpoint must land on `llm.providers.openai.api_url`, not a dropped key."""
        llm = LLMConfig(**ollama_example['llm'])
        assert llm.providers.openai is not None, (
            'llm.providers.openai is None - the documented endpoint keys are silently '
            'ignored by LLMConfig (extra="ignore")'
        )
        assert llm.providers.openai.api_url == OLLAMA_API_URL

    def test_llm_example_builds_chat_completions_client(self, ollama_example):
        """A localhost endpoint must route through `is_non_openai_provider` to the generic client."""
        llm = LLMConfig(**ollama_example['llm'])
        client = LLMClientFactory.create(llm)
        assert isinstance(client, OpenAIGenericClient), (
            f'expected OpenAIGenericClient (Chat Completions) for {OLLAMA_API_URL}, '
            f'got {type(client).__name__}'
        )

    def test_embedder_example_builds_openai_compatible_embedder(self, ollama_example):
        """The documented embedder section must construct without network or a real API key."""
        embedder = EmbedderFactory.create(EmbedderConfig(**ollama_example['embedder']))
        assert isinstance(embedder, OpenAIEmbedder), (
            f'expected an OpenAI-compatible embedder, got {type(embedder).__name__}'
        )


class TestReadmeEmbedderProviders:
    """Every embedder provider named anywhere in the README must exist in the schema."""

    def test_all_documented_embedder_providers_are_supported(self):
        """`provider:` values under any README `embedder:` block must be schema fields."""
        supported = set(EmbedderProvidersConfig.model_fields)
        documented = set()
        for match in YAML_FENCE_RE.finditer(_readme_text()):
            data = yaml.safe_load(match.group('yaml'))
            if not isinstance(data, dict):
                continue
            embedder = data.get('embedder')
            if isinstance(embedder, dict) and 'provider' in embedder:
                documented.add(str(embedder['provider']))

        assert documented, 'no embedder provider found in any README yaml block'
        unsupported = sorted(documented - supported)
        assert not unsupported, (
            f'README documents unsupported embedder provider(s) {unsupported}; '
            f'supported: {sorted(supported)}'
        )
