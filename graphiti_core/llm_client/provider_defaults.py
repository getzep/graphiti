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

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ProviderDefaults:
    """
    Configuration for provider-specific model defaults.
    
    This class replaces hardcoded DEFAULT_MODEL constants with configurable
    provider-specific defaults that can be overridden via environment variables.
    """
    model: str
    small_model: str
    max_tokens: int = 8192
    temperature: float = 0.0
    extract_edges_max_tokens: int = 16384


# Provider-specific default configurations
# These can be overridden via environment variables (see get_provider_defaults)
PROVIDER_DEFAULTS: Dict[str, ProviderDefaults] = {
    'openai': ProviderDefaults(
        model='gpt-4.1-mini',
        small_model='gpt-4.1-nano',
        extract_edges_max_tokens=16384,
    ),
    'gemini': ProviderDefaults(
        model='gemini-2.5-flash',
        small_model='models/gemini-2.5-flash-lite-preview-06-17',
        extract_edges_max_tokens=16384,
    ),
    'anthropic': ProviderDefaults(
        model='claude-3-7-sonnet-latest',
        small_model='claude-3-7-haiku-latest',
        extract_edges_max_tokens=16384,
    ),
    'groq': ProviderDefaults(
        model='llama-3.1-70b-versatile',
        small_model='llama-3.1-8b-instant',
        extract_edges_max_tokens=16384,
    ),
    'azure_openai': ProviderDefaults(
        model='gpt-4.1-mini',
        small_model='gpt-4.1-nano',
        extract_edges_max_tokens=16384,
    ),
}


def get_provider_defaults(provider: str) -> ProviderDefaults:
    """
    Get provider-specific defaults with optional environment variable overrides.
    
    Environment variables can override defaults using the pattern:
    - {PROVIDER}_DEFAULT_MODEL
    - {PROVIDER}_DEFAULT_SMALL_MODEL
    - {PROVIDER}_DEFAULT_MAX_TOKENS
    - {PROVIDER}_DEFAULT_TEMPERATURE
    - {PROVIDER}_EXTRACT_EDGES_MAX_TOKENS
    
    Args:
        provider: The provider name (e.g., 'openai', 'gemini', 'anthropic', etc.)
        
    Returns:
        ProviderDefaults object with defaults for the specified provider
        
    Raises:
        ValueError: If the provider is not supported
    """
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(PROVIDER_DEFAULTS.keys())}")
    
    defaults = PROVIDER_DEFAULTS[provider]
    
    # Check for environment variable overrides
    env_prefix = provider.upper()
    
    model = os.getenv(f'{env_prefix}_DEFAULT_MODEL', defaults.model)
    small_model = os.getenv(f'{env_prefix}_DEFAULT_SMALL_MODEL', defaults.small_model)
    max_tokens = int(os.getenv(f'{env_prefix}_DEFAULT_MAX_TOKENS', str(defaults.max_tokens)))
    temperature = float(os.getenv(f'{env_prefix}_DEFAULT_TEMPERATURE', str(defaults.temperature)))
    extract_edges_max_tokens = int(os.getenv(f'{env_prefix}_EXTRACT_EDGES_MAX_TOKENS', str(defaults.extract_edges_max_tokens)))
    
    return ProviderDefaults(
        model=model,
        small_model=small_model,
        max_tokens=max_tokens,
        temperature=temperature,
        extract_edges_max_tokens=extract_edges_max_tokens
    )


def get_model_for_size(provider: str, model_size: str, user_model: Optional[str] = None, user_small_model: Optional[str] = None) -> str:
    """
    Get the appropriate model name based on the requested size and provider.
    
    This function replaces the _get_model_for_size methods in individual clients
    with a centralized implementation that uses configurable provider defaults.
    
    Args:
        provider: The provider name (e.g., 'openai', 'gemini', 'anthropic', etc.)
        model_size: The size of the model requested ('small' or 'medium')
        user_model: User-configured model override
        user_small_model: User-configured small model override
        
    Returns:
        The appropriate model name for the requested size
    """
    defaults = get_provider_defaults(provider)
    
    if model_size == 'small':
        return user_small_model or defaults.small_model
    else:
        return user_model or defaults.model


def get_extract_edges_max_tokens(provider: str) -> int:
    """
    Get the maximum tokens for edge extraction operations.
    
    This function replaces hardcoded extract_edges_max_tokens constants
    with configurable provider-specific defaults.
    
    Args:
        provider: The provider name (e.g., 'openai', 'gemini', 'anthropic', etc.)
        
    Returns:
        The maximum tokens for edge extraction operations
    """
    defaults = get_provider_defaults(provider)
    return defaults.extract_edges_max_tokens


def get_extract_edges_max_tokens_default() -> int:
    """
    Get the default maximum tokens for edge extraction operations.
    
    This function provides a configurable default that can be overridden
    via the EXTRACT_EDGES_MAX_TOKENS environment variable.
    
    Returns:
        The maximum tokens for edge extraction operations
    """
    return int(os.getenv('EXTRACT_EDGES_MAX_TOKENS', '16384'))