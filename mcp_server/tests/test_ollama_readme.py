"""Check that both clients in the documented Ollama configuration use its local API."""

from pathlib import Path

import yaml
from graphiti_core.embedder.openai import OpenAIEmbedder
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from config.schema import EmbedderConfig, LLMConfig
from services.factories import EmbedderFactory, LLMClientFactory


def ollama_example():
    readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
    section = readme.split('### Using Ollama for Local LLM', 1)[1]
    return yaml.safe_load(section.split('```yaml', 1)[1].split('```', 1)[0])


def test_ollama_example_routes_chat_to_local_server():
    config = LLMConfig.model_validate(ollama_example()['llm'])
    client = LLMClientFactory.create(config)
    assert isinstance(client, OpenAIGenericClient)
    assert str(client.client.base_url).rstrip('/') == 'http://localhost:11434/v1'


def test_ollama_example_routes_embeddings_to_local_server():
    config = EmbedderConfig.model_validate(ollama_example()['embedder'])
    client = EmbedderFactory.create(config)
    assert isinstance(client, OpenAIEmbedder)
    assert str(client.client.base_url).rstrip('/') == 'http://localhost:11434/v1'
