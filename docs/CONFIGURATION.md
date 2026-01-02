# Graphiti Configuration Guide

This guide explains how to configure Graphiti using the new unified configuration system introduced in v0.24.2.

## Overview

Graphiti now supports flexible configuration through:
- **YAML configuration files** for declarative setup
- **Environment variables** for secrets and deployment-specific settings
- **Programmatic configuration** for dynamic setups
- **Backward compatibility** with existing initialization patterns

## Quick Start

### Option 1: YAML Configuration (Recommended)

Create a `.graphiti.yaml` file in your project root:

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-5-latest
  temperature: 0.7

embedder:
  provider: voyage
  model: voyage-3

database:
  provider: neo4j
  uri: "bolt://localhost:7687"
  user: neo4j
  password: password
```

Then initialize Graphiti:

```python
from graphiti_core import Graphiti
from graphiti_core.config import GraphitiConfig

# Load from default .graphiti.yaml
config = GraphitiConfig.from_env()
graphiti = Graphiti.from_config(config)

# Or load from specific file
config = GraphitiConfig.from_yaml("path/to/config.yaml")
graphiti = Graphiti.from_config(config)
```

### Option 2: Programmatic Configuration

```python
from graphiti_core import Graphiti
from graphiti_core.config import (
    GraphitiConfig,
    LLMProviderConfig,
    EmbedderConfig,
    DatabaseConfig,
    LLMProvider,
    EmbedderProvider,
)

config = GraphitiConfig(
    llm=LLMProviderConfig(
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-5-latest",
    ),
    embedder=EmbedderConfig(
        provider=EmbedderProvider.VOYAGE,
    ),
    database=DatabaseConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
    ),
)

graphiti = Graphiti.from_config(config)
```

### Option 3: Traditional Initialization (Backward Compatible)

```python
from graphiti_core import Graphiti

# Still works as before!
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
)
```

## Supported Providers

### LLM Providers

| Provider | Value | Install Command | Notes |
|----------|-------|----------------|-------|
| OpenAI | `openai` | Built-in | Default provider |
| Azure OpenAI | `azure_openai` | Built-in | Requires `base_url` and `azure_deployment` |
| Anthropic | `anthropic` | `pip install graphiti-core[anthropic]` | Claude models |
| Google Gemini | `gemini` | `pip install graphiti-core[google-genai]` | Gemini models |
| Groq | `groq` | `pip install graphiti-core[groq]` | Fast inference |
| LiteLLM | `litellm` | `pip install graphiti-core[litellm]` | Unified interface for 100+ providers |
| Custom | `custom` | - | Bring your own client |

### Embedder Providers

| Provider | Value | Install Command | Default Model |
|----------|-------|----------------|---------------|
| OpenAI | `openai` | Built-in | text-embedding-3-small |
| Azure OpenAI | `azure_openai` | Built-in | text-embedding-3-small |
| Voyage AI | `voyage` | `pip install graphiti-core[voyageai]` | voyage-3 |
| Google Gemini | `gemini` | `pip install graphiti-core[google-genai]` | text-embedding-004 |
| Custom | `custom` | - | Bring your own embedder |

### Database Providers

| Provider | Value | Install Command | Notes |
|----------|-------|----------------|-------|
| Neo4j | `neo4j` | Built-in | Default provider |
| FalkorDB | `falkordb` | `pip install graphiti-core[falkordb]` | Redis-based graph DB |
| Neptune | `neptune` | `pip install graphiti-core[neptune]` | AWS Neptune |
| Custom | `custom` | - | Bring your own driver |

## Configuration Examples

### Azure OpenAI

```yaml
llm:
  provider: azure_openai
  base_url: "https://your-resource.openai.azure.com"
  azure_deployment: "gpt-4-deployment"
  azure_api_version: "2024-10-21"
  # api_key via AZURE_OPENAI_API_KEY env var

embedder:
  provider: azure_openai
  base_url: "https://your-resource.openai.azure.com"
  azure_deployment: "embedding-deployment"
  model: text-embedding-3-small

database:
  provider: neo4j
  uri: "bolt://localhost:7687"
  user: neo4j
  password: password
```

### LiteLLM for Multi-Cloud

LiteLLM provides a unified interface to 100+ LLM providers:

```yaml
llm:
  provider: litellm
  litellm_model: "azure/gpt-4-deployment"  # Azure OpenAI
  # Or: "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"  # AWS Bedrock
  # Or: "ollama/llama2"  # Local Ollama
  # Or: "vertex_ai/gemini-pro"  # Google Vertex AI
  base_url: "https://your-resource.openai.azure.com"
  api_key: "your-key"
```

### Local Models with Ollama

```yaml
llm:
  provider: litellm
  litellm_model: "ollama/llama2"
  base_url: "http://localhost:11434"
  temperature: 0.8
  max_tokens: 4096

embedder:
  provider: openai  # Or use local embeddings
  model: text-embedding-3-small
```

### Anthropic + Voyage AI

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-5-latest
  small_model: claude-haiku-4-5-latest
  temperature: 0.7

embedder:
  provider: voyage
  model: voyage-3
  dimensions: 1024
```

### Google Gemini

```yaml
llm:
  provider: gemini
  model: gemini-2.5-flash
  temperature: 0.9

embedder:
  provider: gemini
  model: text-embedding-004
  dimensions: 768
```

## Environment Variables

API keys and secrets can be provided via environment variables:

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Voyage AI | `VOYAGE_API_KEY` |

Configuration file location:
- `GRAPHITI_CONFIG_PATH`: Path to configuration YAML file

Example:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GRAPHITI_CONFIG_PATH="/path/to/config.yaml"
```

## Configuration Reference

### LLMProviderConfig

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `provider` | `LLMProvider` | LLM provider to use | `openai` |
| `model` | `str` | Model name | Provider default |
| `small_model` | `str` | Smaller model for simple tasks | Provider default |
| `api_key` | `str` | API key (or from env) | From environment |
| `base_url` | `str` | API base URL | Provider default |
| `temperature` | `float` | Sampling temperature (0-2) | `1.0` |
| `max_tokens` | `int` | Maximum tokens in response | `8192` |
| `azure_deployment` | `str` | Azure deployment name | `None` |
| `azure_api_version` | `str` | Azure API version | `2024-10-21` |
| `litellm_model` | `str` | LiteLLM model string | `None` |

### EmbedderConfig

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `provider` | `EmbedderProvider` | Embedder provider | `openai` |
| `model` | `str` | Embedding model name | Provider default |
| `api_key` | `str` | API key (or from env) | From environment |
| `base_url` | `str` | API base URL | Provider default |
| `dimensions` | `int` | Embedding dimensions | Provider default |
| `azure_deployment` | `str` | Azure deployment name | `None` |
| `azure_api_version` | `str` | Azure API version | `2024-10-21` |

### DatabaseConfig

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `provider` | `DatabaseProvider` | Database provider | `neo4j` |
| `uri` | `str` | Database connection URI | `None` |
| `user` | `str` | Database username | `None` |
| `password` | `str` | Database password | `None` |
| `database` | `str` | Database name | Provider default |

### GraphitiConfig

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `llm` | `LLMProviderConfig` | LLM configuration | Default config |
| `embedder` | `EmbedderConfig` | Embedder configuration | Default config |
| `reranker` | `RerankerConfig` | Reranker configuration | Default config |
| `database` | `DatabaseConfig` | Database configuration | Default config |
| `store_raw_episode_content` | `bool` | Store raw episode content | `True` |
| `max_coroutines` | `int` | Max concurrent operations | `None` |

## Migration Guide

### From Traditional Initialization

**Before:**
```python
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
)
```

**After (with config):**
```python
from graphiti_core.config import GraphitiConfig, DatabaseConfig

config = GraphitiConfig(
    database=DatabaseConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
    )
)
graphiti = Graphiti.from_config(config)
```

**Or with YAML:**
```yaml
# .graphiti.yaml
database:
  uri: "bolt://localhost:7687"
  user: neo4j
  password: password
```

```python
config = GraphitiConfig.from_env()
graphiti = Graphiti.from_config(config)
```

### From Custom Clients

**Before:**
```python
from graphiti_core.llm_client.anthropic_client import AnthropicClient

llm = AnthropicClient(...)
graphiti = Graphiti(
    uri="...",
    llm_client=llm,
)
```

**After:**
```python
from graphiti_core.config import GraphitiConfig, LLMProviderConfig, LLMProvider

config = GraphitiConfig(
    llm=LLMProviderConfig(provider=LLMProvider.ANTHROPIC),
)
graphiti = Graphiti.from_config(config)
```

## Benefits

1. **Centralized Configuration**: All settings in one place
2. **Environment-Specific Configs**: Different configs for dev/staging/prod
3. **No Code Changes**: Switch providers via config file
4. **Better Validation**: Pydantic validates all settings
5. **Multi-Provider Support**: Easy integration with LiteLLM
6. **Backward Compatible**: Existing code continues to work

## Troubleshooting

### Missing Dependencies

If you get an import error for a provider:

```
ImportError: Anthropic client requires anthropic package.
Install with: pip install graphiti-core[anthropic]
```

Install the required optional dependency:
```bash
pip install graphiti-core[anthropic]
pip install graphiti-core[voyageai]
pip install graphiti-core[litellm]
```

### Azure OpenAI Configuration

Azure OpenAI requires:
- `base_url`: Your Azure endpoint
- `azure_deployment`: Deployment name (not model name)
- `api_key`: Azure API key (or via `AZURE_OPENAI_API_KEY`)

### LiteLLM Model Format

LiteLLM uses specific format for models:
- Azure: `azure/deployment-name`
- Bedrock: `bedrock/model-id`
- Ollama: `ollama/model-name`
- Vertex AI: `vertex_ai/model-name`

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for more providers.

## Examples

See `examples/graphiti_config_example.yaml` for a complete configuration example with multiple provider options.

## Related Issues

This feature addresses:
- #1004: Azure OpenAI support
- #1006: Azure OpenAI reranker support
- #1007: vLLM/OpenAI-compatible provider stability
- #1074: Ollama embeddings support

## Further Reading

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Pydantic Settings Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Neo4j Connection Strings](https://neo4j.com/docs/operations-manual/current/configuration/connectors/)
