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

import importlib
import logging
from typing import TYPE_CHECKING

from ..cross_encoder.client import CrossEncoderClient
from ..cross_encoder.openai_reranker_client import OpenAIRerankerClient
from ..driver.driver import GraphDriver
from ..embedder.client import EmbedderClient
from ..llm_client.client import LLMClient
from ..llm_client.config import LLMConfig
from .providers import DatabaseProvider, EmbedderProvider, LLMProvider, RerankerProvider

if TYPE_CHECKING:
    from .settings import DatabaseConfig, EmbedderConfig, LLMProviderConfig, RerankerConfig

logger = logging.getLogger(__name__)


def create_llm_client(config: 'LLMProviderConfig') -> LLMClient:
    """Create an LLM client based on configuration.

    Args:
        config: LLM provider configuration

    Returns:
        Configured LLM client instance

    Raises:
        ValueError: If provider is not supported or configuration is invalid
        ImportError: If required dependencies for the provider are not installed
    """
    # Create LLMConfig from provider config
    llm_config = LLMConfig(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        small_model=config.small_model,
    )

    if config.provider == LLMProvider.OPENAI:
        from ..llm_client.openai_client import OpenAIClient

        return OpenAIClient(llm_config)

    elif config.provider == LLMProvider.AZURE_OPENAI:
        from ..llm_client.azure_openai_client import AzureOpenAILLMClient

        if not config.azure_deployment:
            raise ValueError('azure_deployment is required for Azure OpenAI provider')

        return AzureOpenAILLMClient(
            llm_config,
            azure_deployment=config.azure_deployment,
            azure_api_version=config.azure_api_version or '2024-10-21',
        )

    elif config.provider == LLMProvider.ANTHROPIC:
        try:
            from ..llm_client.anthropic_client import AnthropicClient

            return AnthropicClient(llm_config)
        except ImportError as e:
            raise ImportError(
                'Anthropic client requires anthropic package. '
                'Install with: pip install graphiti-core[anthropic]'
            ) from e

    elif config.provider == LLMProvider.GEMINI:
        try:
            from ..llm_client.gemini_client import GeminiClient

            return GeminiClient(llm_config)
        except ImportError as e:
            raise ImportError(
                'Gemini client requires google-genai package. '
                'Install with: pip install graphiti-core[google-genai]'
            ) from e

    elif config.provider == LLMProvider.GROQ:
        try:
            from ..llm_client.groq_client import GroqClient

            return GroqClient(llm_config)
        except ImportError as e:
            raise ImportError(
                'Groq client requires groq package. Install with: pip install graphiti-core[groq]'
            ) from e

    elif config.provider == LLMProvider.LITELLM:
        try:
            from ..llm_client.litellm_client import LiteLLMClient

            # For LiteLLM, use the litellm_model if provided
            if config.litellm_model:
                llm_config.model = config.litellm_model
            return LiteLLMClient(llm_config)
        except ImportError as e:
            raise ImportError(
                'LiteLLM client requires litellm package. '
                'Install with: pip install graphiti-core[litellm]'
            ) from e

    elif config.provider == LLMProvider.CUSTOM:
        if not config.custom_client_class:
            raise ValueError('custom_client_class is required for custom LLM provider')

        # Import and instantiate custom client class
        module_name, class_name = config.custom_client_class.rsplit('.', 1)
        module = importlib.import_module(module_name)
        client_class = getattr(module, class_name)
        return client_class(llm_config)

    else:
        raise ValueError(f'Unsupported LLM provider: {config.provider}')


def create_embedder(config: 'EmbedderConfig') -> EmbedderClient:
    """Create an embedder client based on configuration.

    Args:
        config: Embedder configuration

    Returns:
        Configured embedder client instance

    Raises:
        ValueError: If provider is not supported or configuration is invalid
        ImportError: If required dependencies for the provider are not installed
    """
    if config.provider == EmbedderProvider.OPENAI:
        from ..embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        embedder_config = OpenAIEmbedderConfig(
            api_key=config.api_key,
            embedding_model=config.model or 'text-embedding-3-small',
            embedding_dim=config.dimensions or 1536,
        )
        return OpenAIEmbedder(config=embedder_config)

    elif config.provider == EmbedderProvider.AZURE_OPENAI:
        from openai import AsyncAzureOpenAI

        from ..embedder.azure_openai import AzureOpenAIEmbedderClient  # type: ignore

        if not config.base_url:
            raise ValueError('base_url is required for Azure OpenAI embedder')

        azure_client = AsyncAzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.base_url,
            api_version=config.azure_api_version or '2024-10-21',
        )

        return AzureOpenAIEmbedderClient(  # type: ignore
            azure_client=azure_client,
            model=config.azure_deployment or config.model or 'text-embedding-3-small',
        )

    elif config.provider == EmbedderProvider.VOYAGE:
        try:
            from ..embedder.voyage import VoyageEmbedder, VoyageEmbedderConfig  # type: ignore

            voyage_config = VoyageEmbedderConfig(  # type: ignore
                api_key=config.api_key,
                embedding_model=config.model or 'voyage-3',
            )
            return VoyageEmbedder(config=voyage_config)  # type: ignore
        except ImportError as e:
            raise ImportError(
                'Voyage embedder requires voyageai package. '
                'Install with: pip install graphiti-core[voyageai]'
            ) from e

    elif config.provider == EmbedderProvider.GEMINI:
        try:
            from ..embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig  # type: ignore

            gemini_config = GeminiEmbedderConfig(  # type: ignore
                api_key=config.api_key,
                embedding_model=config.model or 'text-embedding-004',
            )
            return GeminiEmbedder(config=gemini_config)  # type: ignore
        except ImportError as e:
            raise ImportError(
                'Gemini embedder requires google-genai package. '
                'Install with: pip install graphiti-core[google-genai]'
            ) from e

    elif config.provider == EmbedderProvider.CUSTOM:
        if not config.custom_client_class:
            raise ValueError('custom_client_class is required for custom embedder provider')

        # Import and instantiate custom embedder class
        module_name, class_name = config.custom_client_class.rsplit('.', 1)
        module = importlib.import_module(module_name)
        embedder_class = getattr(module, class_name)
        return embedder_class(api_key=config.api_key, model=config.model)

    else:
        raise ValueError(f'Unsupported embedder provider: {config.provider}')


def create_reranker(config: 'RerankerConfig') -> CrossEncoderClient:
    """Create a reranker/cross-encoder client based on configuration.

    Args:
        config: Reranker configuration

    Returns:
        Configured reranker client instance

    Raises:
        ValueError: If provider is not supported or configuration is invalid
    """
    if config.provider in (RerankerProvider.OPENAI, RerankerProvider.AZURE_OPENAI):
        return OpenAIRerankerClient()

    elif config.provider == RerankerProvider.CUSTOM:
        if not config.custom_client_class:
            raise ValueError('custom_client_class is required for custom reranker provider')

        # Import and instantiate custom reranker class
        module_name, class_name = config.custom_client_class.rsplit('.', 1)
        module = importlib.import_module(module_name)
        reranker_class = getattr(module, class_name)
        return reranker_class()

    else:
        raise ValueError(f'Unsupported reranker provider: {config.provider}')


def create_database_driver(config: 'DatabaseConfig') -> GraphDriver:
    """Create a graph database driver based on configuration.

    Args:
        config: Database configuration

    Returns:
        Configured database driver instance

    Raises:
        ValueError: If provider is not supported or configuration is invalid
        ImportError: If required dependencies for the provider are not installed
    """
    if config.provider == DatabaseProvider.NEO4J:
        from ..driver.neo4j_driver import Neo4jDriver

        if not config.uri:
            raise ValueError('uri is required for Neo4j database')

        return Neo4jDriver(
            uri=config.uri,
            user=config.user,
            password=config.password,
            database=config.database,  # type: ignore
        )

    elif config.provider == DatabaseProvider.FALKORDB:
        try:
            from ..driver.falkor_driver import FalkorDriver  # type: ignore

            if not config.uri:
                raise ValueError('uri is required for FalkorDB database')

            return FalkorDriver(  # type: ignore
                uri=config.uri,
                user=config.user,
                password=config.password,
                database=config.database,  # type: ignore
            )
        except ImportError as e:
            raise ImportError(
                'FalkorDB driver requires falkordb package. '
                'Install with: pip install graphiti-core[falkordb]'
            ) from e

    elif config.provider == DatabaseProvider.NEPTUNE:
        try:
            from ..driver.neptune_driver import NeptuneDriver  # type: ignore

            if not config.uri:
                raise ValueError('uri is required for Neptune database')

            # Neptune driver has different signature - add type ignore
            return NeptuneDriver(config.uri)  # type: ignore
        except ImportError as e:
            raise ImportError(
                'Neptune driver requires langchain-aws and related packages. '
                'Install with: pip install graphiti-core[neptune]'
            ) from e

    elif config.provider == DatabaseProvider.CUSTOM:
        if not config.custom_driver_class:
            raise ValueError('custom_driver_class is required for custom database provider')

        # Import and instantiate custom driver class
        module_name, class_name = config.custom_driver_class.rsplit('.', 1)
        module = importlib.import_module(module_name)
        driver_class = getattr(module, class_name)
        return driver_class(uri=config.uri, user=config.user, password=config.password)

    else:
        raise ValueError(f'Unsupported database provider: {config.provider}')
