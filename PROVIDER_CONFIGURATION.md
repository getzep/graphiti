# Provider Configuration System

This document describes the new provider configuration system that replaces hardcoded "ghost variables" with configurable defaults.

## Overview

Previously, each LLM provider client had hardcoded model names and configuration values that could not be easily customized without modifying the source code. This created several issues:

1. **Maintenance burden**: Updating to newer models required code changes
2. **Limited flexibility**: Users couldn't easily switch to different models
3. **Provider constraints**: Some models (like Gemini's flash-lite) have specific location constraints that differed from defaults
4. **Hidden configurations**: Token limits and other operational parameters were buried in the code

## New Configuration System

The new system introduces a centralized `provider_defaults.py` module that:

1. **Centralizes all provider defaults** in a single location
2. **Supports environment variable overrides** for easy customization
3. **Maintains backward compatibility** with existing configurations
4. **Provides provider-specific configurations** for different LLM providers

## Environment Variables

You can now override any provider default using environment variables with the following pattern:

```bash
# For OpenAI
export OPENAI_DEFAULT_MODEL="gpt-4"
export OPENAI_DEFAULT_SMALL_MODEL="gpt-4-mini"
export OPENAI_DEFAULT_MAX_TOKENS="8192"
export OPENAI_DEFAULT_TEMPERATURE="0.0"
export OPENAI_EXTRACT_EDGES_MAX_TOKENS="16384"

# For Gemini
export GEMINI_DEFAULT_MODEL="gemini-2.5-flash"
export GEMINI_DEFAULT_SMALL_MODEL="gemini-2.5-flash-lite"
export GEMINI_DEFAULT_MAX_TOKENS="8192"
export GEMINI_DEFAULT_TEMPERATURE="0.0"
export GEMINI_EXTRACT_EDGES_MAX_TOKENS="16384"

# For Anthropic
export ANTHROPIC_DEFAULT_MODEL="claude-3-5-sonnet-latest"
export ANTHROPIC_DEFAULT_SMALL_MODEL="claude-3-5-haiku-latest"
export ANTHROPIC_DEFAULT_MAX_TOKENS="8192"
export ANTHROPIC_DEFAULT_TEMPERATURE="0.0"
export ANTHROPIC_EXTRACT_EDGES_MAX_TOKENS="16384"

# For Groq
export GROQ_DEFAULT_MODEL="llama-3.1-70b-versatile"
export GROQ_DEFAULT_SMALL_MODEL="llama-3.1-8b-instant"
export GROQ_DEFAULT_MAX_TOKENS="8192"
export GROQ_DEFAULT_TEMPERATURE="0.0"
export GROQ_EXTRACT_EDGES_MAX_TOKENS="16384"

# General configuration (for edge operations)
export EXTRACT_EDGES_MAX_TOKENS="16384"
```

## Supported Providers

The system currently supports the following providers:

- **openai**: OpenAI GPT models
- **gemini**: Google Gemini models
- **anthropic**: Anthropic Claude models
- **groq**: Groq models
- **azure_openai**: Azure OpenAI models

## Usage Examples

### Basic Usage

The configuration system works transparently with existing code:

```python
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig

# Uses default models (configurable via environment variables)
client = OpenAIClient()

# Or with explicit configuration (still uses provider defaults as fallback)
config = LLMConfig(model="gpt-4", small_model="gpt-4-mini")
client = OpenAIClient(config)
```

### Customizing Model Defaults

Instead of hardcoding model names in your application, you can now use environment variables:

```bash
# Set up your preferred models
export OPENAI_DEFAULT_MODEL="gpt-4"
export OPENAI_DEFAULT_SMALL_MODEL="gpt-4-mini"

# Your application will automatically use these defaults
python your_app.py
```

### Provider-Specific Configuration

Each provider can have different default models and configurations:

```python
from graphiti_core.llm_client.provider_defaults import get_provider_defaults

# Get defaults for a specific provider
openai_defaults = get_provider_defaults('openai')
print(f"OpenAI default model: {openai_defaults.model}")
print(f"OpenAI small model: {openai_defaults.small_model}")

gemini_defaults = get_provider_defaults('gemini')
print(f"Gemini default model: {gemini_defaults.model}")
print(f"Gemini small model: {gemini_defaults.small_model}")
```

## Migration Guide

### Before (with ghost variables)

```python
# In gemini_client.py
DEFAULT_MODEL = 'gemini-2.5-flash'
DEFAULT_SMALL_MODEL = 'models/gemini-2.5-flash-lite-preview-06-17'

def _get_model_for_size(self, model_size: ModelSize) -> str:
    if model_size == ModelSize.small:
        return self.small_model or DEFAULT_SMALL_MODEL
    else:
        return self.model or DEFAULT_MODEL
```

### After (with configurable defaults)

```python
# Configuration is now externalized
from .provider_defaults import get_model_for_size

def _get_model_for_size(self, model_size: ModelSize) -> str:
    return get_model_for_size(
        provider='gemini',
        model_size=model_size.value,
        user_model=self.model,
        user_small_model=self.small_model
    )
```

## Benefits

1. **No More Ghost Variables**: All defaults are now configurable
2. **Easy Model Updates**: Update models via environment variables
3. **Provider Flexibility**: Each provider can have optimized defaults
4. **Backward Compatibility**: Existing code continues to work unchanged
5. **Environment-Specific Configuration**: Different environments can use different models
6. **Reduced Maintenance**: No need to modify source code for model updates

## Implementation Details

The new system is implemented in `graphiti_core/llm_client/provider_defaults.py` and includes:

- `ProviderDefaults` dataclass for configuration structure
- `get_provider_defaults()` function with environment variable support
- `get_model_for_size()` centralized model selection logic
- `get_extract_edges_max_tokens_default()` for operational parameters

All existing LLM client implementations have been updated to use this new system while maintaining full backward compatibility.