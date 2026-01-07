"""Factory classes for creating LLM, Embedder, and Database clients."""

from openai import AsyncAzureOpenAI

from config.schema import (
    DatabaseConfig,
    EmbedderConfig,
    LLMConfig,
)

# Try to import FalkorDriver if available
try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: F401

    HAS_FALKOR = True
except ImportError:
    HAS_FALKOR = False

# Kuzu support removed - FalkorDB is now the default
from graphiti_core.cross_encoder import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig

# Try to import Gemini reranker if available
try:
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

    HAS_GEMINI_RERANKER = True
except ImportError:
    HAS_GEMINI_RERANKER = False

# Try to import OpenAI reranker
try:
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

    HAS_OPENAI_RERANKER = True
except ImportError:
    HAS_OPENAI_RERANKER = False

# Try to import additional providers if available
try:
    from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient

    HAS_AZURE_EMBEDDER = True
except ImportError:
    HAS_AZURE_EMBEDDER = False

try:
    from graphiti_core.embedder.gemini import GeminiEmbedder

    HAS_GEMINI_EMBEDDER = True
except ImportError:
    HAS_GEMINI_EMBEDDER = False

try:
    from graphiti_core.embedder.voyage import VoyageAIEmbedder

    HAS_VOYAGE_EMBEDDER = True
except ImportError:
    HAS_VOYAGE_EMBEDDER = False

try:
    from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient

    HAS_AZURE_LLM = True
except ImportError:
    HAS_AZURE_LLM = False

try:
    from graphiti_core.llm_client.anthropic_client import AnthropicClient

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from graphiti_core.llm_client.gemini_client import GeminiClient

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from graphiti_core.llm_client.groq_client import GroqClient

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
from utils.utils import create_azure_credential_token_provider


def _is_vertex_ai_mode() -> bool:
    """Check if Vertex AI mode is enabled via environment variable."""
    import os
    return os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').lower() == 'true'


def _validate_api_key(provider_name: str, api_key: str | None, logger) -> str | None:
    """Validate API key is present.

    Args:
        provider_name: Name of the provider (e.g., 'OpenAI', 'Anthropic')
        api_key: The API key to validate
        logger: Logger instance for output

    Returns:
        The validated API key, or None if Vertex AI mode is enabled for Gemini

    Raises:
        ValueError: If API key is None or empty (unless Vertex AI mode for Gemini)
    """
    # Allow None for Gemini when using Vertex AI (ADC authentication)
    if not api_key and 'Gemini' in provider_name and _is_vertex_ai_mode():
        logger.info(f'Creating {provider_name} client with Vertex AI (ADC authentication)')
        return None

    if not api_key:
        raise ValueError(
            f'{provider_name} API key is not configured. Please set the appropriate environment variable.'
        )

    logger.info(f'Creating {provider_name} client')

    return api_key


class LLMClientFactory:
    """Factory for creating LLM clients based on configuration."""

    @staticmethod
    def create(config: LLMConfig) -> LLMClient:
        """Create an LLM client based on the configured provider."""
        import logging

        logger = logging.getLogger(__name__)

        provider = config.provider.lower()

        match provider:
            case 'openai':
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI', api_key, logger)

                from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                # Determine appropriate small model based on main model type
                is_reasoning_model = (
                    config.model.startswith('gpt-5')
                    or config.model.startswith('o1')
                    or config.model.startswith('o3')
                )
                small_model = (
                    'gpt-5-nano' if is_reasoning_model else 'gpt-4.1-mini'
                )  # Use reasoning model for small tasks if main model is reasoning

                llm_config = CoreLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    small_model=small_model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

                # Only pass reasoning/verbosity parameters for reasoning models (gpt-5 family)
                if is_reasoning_model:
                    return OpenAIClient(config=llm_config, reasoning='minimal', verbosity='low')
                else:
                    # For non-reasoning models, explicitly pass None to disable these parameters
                    return OpenAIClient(config=llm_config, reasoning=None, verbosity=None)

            case 'azure_openai':
                if not HAS_AZURE_LLM:
                    raise ValueError(
                        'Azure OpenAI LLM client not available in current graphiti-core version'
                    )
                if not config.providers.azure_openai:
                    raise ValueError('Azure OpenAI provider configuration not found')
                azure_config = config.providers.azure_openai

                if not azure_config.api_url:
                    raise ValueError('Azure OpenAI API URL is required')

                # Handle Azure AD authentication if enabled
                api_key: str | None = None
                azure_ad_token_provider = None
                if azure_config.use_azure_ad:
                    logger.info('Creating Azure OpenAI LLM client with Azure AD authentication')
                    azure_ad_token_provider = create_azure_credential_token_provider()
                else:
                    api_key = azure_config.api_key
                    _validate_api_key('Azure OpenAI', api_key, logger)

                # Create the Azure OpenAI client first
                azure_client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=azure_config.api_url,
                    api_version=azure_config.api_version,
                    azure_deployment=azure_config.deployment_name,
                    azure_ad_token_provider=azure_ad_token_provider,
                )

                # Then create the LLMConfig
                from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                llm_config = CoreLLMConfig(
                    api_key=api_key,
                    base_url=azure_config.api_url,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

                return AzureOpenAILLMClient(
                    azure_client=azure_client,
                    config=llm_config,
                    max_tokens=config.max_tokens,
                )

            case 'anthropic':
                if not HAS_ANTHROPIC:
                    raise ValueError(
                        'Anthropic client not available in current graphiti-core version'
                    )
                if not config.providers.anthropic:
                    raise ValueError('Anthropic provider configuration not found')

                api_key = config.providers.anthropic.api_key
                _validate_api_key('Anthropic', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return AnthropicClient(config=llm_config)

            case 'gemini':
                if not HAS_GEMINI:
                    raise ValueError('Gemini client not available in current graphiti-core version')
                if not config.providers.gemini:
                    raise ValueError('Gemini provider configuration not found')

                api_key = config.providers.gemini.api_key
                _validate_api_key('Gemini', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return GeminiClient(config=llm_config)

            case 'groq':
                if not HAS_GROQ:
                    raise ValueError('Groq client not available in current graphiti-core version')
                if not config.providers.groq:
                    raise ValueError('Groq provider configuration not found')

                api_key = config.providers.groq.api_key
                _validate_api_key('Groq', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    base_url=config.providers.groq.api_url,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return GroqClient(config=llm_config)

            case _:
                raise ValueError(f'Unsupported LLM provider: {provider}')


class EmbedderFactory:
    """Factory for creating Embedder clients based on configuration."""

    @staticmethod
    def create(config: EmbedderConfig) -> EmbedderClient:
        """Create an Embedder client based on the configured provider."""
        import logging

        logger = logging.getLogger(__name__)

        provider = config.provider.lower()

        match provider:
            case 'openai':
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI Embedder', api_key, logger)

                from graphiti_core.embedder.openai import OpenAIEmbedderConfig

                embedder_config = OpenAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model,
                )
                return OpenAIEmbedder(config=embedder_config)

            case 'azure_openai':
                if not HAS_AZURE_EMBEDDER:
                    raise ValueError(
                        'Azure OpenAI embedder not available in current graphiti-core version'
                    )
                if not config.providers.azure_openai:
                    raise ValueError('Azure OpenAI provider configuration not found')
                azure_config = config.providers.azure_openai

                if not azure_config.api_url:
                    raise ValueError('Azure OpenAI API URL is required')

                # Handle Azure AD authentication if enabled
                api_key: str | None = None
                azure_ad_token_provider = None
                if azure_config.use_azure_ad:
                    logger.info(
                        'Creating Azure OpenAI Embedder client with Azure AD authentication'
                    )
                    azure_ad_token_provider = create_azure_credential_token_provider()
                else:
                    api_key = azure_config.api_key
                    _validate_api_key('Azure OpenAI Embedder', api_key, logger)

                # Create the Azure OpenAI client first
                azure_client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=azure_config.api_url,
                    api_version=azure_config.api_version,
                    azure_deployment=azure_config.deployment_name,
                    azure_ad_token_provider=azure_ad_token_provider,
                )

                return AzureOpenAIEmbedderClient(
                    azure_client=azure_client,
                    model=config.model or 'text-embedding-3-small',
                )

            case 'gemini':
                if not HAS_GEMINI_EMBEDDER:
                    raise ValueError(
                        'Gemini embedder not available in current graphiti-core version'
                    )
                if not config.providers.gemini:
                    raise ValueError('Gemini provider configuration not found')

                api_key = config.providers.gemini.api_key
                _validate_api_key('Gemini Embedder', api_key, logger)

                from graphiti_core.embedder.gemini import GeminiEmbedderConfig

                gemini_config = GeminiEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model or 'models/text-embedding-004',
                    embedding_dim=config.dimensions or 768,
                )
                return GeminiEmbedder(config=gemini_config)

            case 'voyage':
                if not HAS_VOYAGE_EMBEDDER:
                    raise ValueError(
                        'Voyage embedder not available in current graphiti-core version'
                    )
                if not config.providers.voyage:
                    raise ValueError('Voyage provider configuration not found')

                api_key = config.providers.voyage.api_key
                _validate_api_key('Voyage Embedder', api_key, logger)

                from graphiti_core.embedder.voyage import VoyageAIEmbedderConfig

                voyage_config = VoyageAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model or 'voyage-3',
                    embedding_dim=config.dimensions or 1024,
                )
                return VoyageAIEmbedder(config=voyage_config)

            case _:
                raise ValueError(f'Unsupported Embedder provider: {provider}')


class DatabaseDriverFactory:
    """Factory for creating Database drivers based on configuration.

    Note: This returns configuration dictionaries that can be passed to Graphiti(),
    not driver instances directly, as the drivers require complex initialization.
    """

    @staticmethod
    def create_config(config: DatabaseConfig) -> dict:
        """Create database configuration dictionary based on the configured provider."""
        provider = config.provider.lower()

        match provider:
            case 'neo4j':
                # Use Neo4j config if provided, otherwise use defaults
                if config.providers.neo4j:
                    neo4j_config = config.providers.neo4j
                else:
                    # Create default Neo4j configuration
                    from config.schema import Neo4jProviderConfig

                    neo4j_config = Neo4jProviderConfig()

                # Check for environment variable overrides (for CI/CD compatibility)
                import os

                uri = os.environ.get('NEO4J_URI', neo4j_config.uri)
                username = os.environ.get('NEO4J_USER', neo4j_config.username)
                password = os.environ.get('NEO4J_PASSWORD', neo4j_config.password)

                return {
                    'uri': uri,
                    'user': username,
                    'password': password,
                    # Note: database and use_parallel_runtime would need to be passed
                    # to the driver after initialization if supported
                }

            case 'falkordb':
                if not HAS_FALKOR:
                    raise ValueError(
                        'FalkorDB driver not available in current graphiti-core version'
                    )

                # Use FalkorDB config if provided, otherwise use defaults
                if config.providers.falkordb:
                    falkor_config = config.providers.falkordb
                else:
                    # Create default FalkorDB configuration
                    from config.schema import FalkorDBProviderConfig

                    falkor_config = FalkorDBProviderConfig()

                # Check for environment variable overrides (for CI/CD compatibility)
                import os
                from urllib.parse import urlparse

                uri = os.environ.get('FALKORDB_URI', falkor_config.uri)
                password = os.environ.get('FALKORDB_PASSWORD', falkor_config.password)

                # Parse the URI to extract host and port
                parsed = urlparse(uri)
                host = parsed.hostname or 'localhost'
                port = parsed.port or 6379

                return {
                    'driver': 'falkordb',
                    'host': host,
                    'port': port,
                    'password': password,
                    'database': falkor_config.database,
                }

            case _:
                raise ValueError(f'Unsupported Database provider: {provider}')


class CrossEncoderFactory:
    """Factory for creating CrossEncoder/Reranker clients based on LLM provider."""

    @staticmethod
    def create(llm_provider: str, llm_config) -> CrossEncoderClient | None:
        """Create a CrossEncoder client based on the LLM provider.

        Uses the same provider as the LLM to ensure consistent authentication.
        Returns None if no suitable reranker is available.
        """
        import logging

        logger = logging.getLogger(__name__)

        provider = llm_provider.lower()

        match provider:
            case 'gemini':
                if not HAS_GEMINI_RERANKER:
                    logger.warning('Gemini reranker not available, skipping cross-encoder')
                    return None

                api_key = None
                if llm_config and llm_config.providers.gemini:
                    api_key = llm_config.providers.gemini.api_key

                # Allow None for Vertex AI mode
                if not api_key and _is_vertex_ai_mode():
                    logger.info('Creating Gemini Reranker with Vertex AI (ADC authentication)')
                    api_key = None
                elif not api_key:
                    logger.warning('Gemini API key not configured, skipping cross-encoder')
                    return None

                reranker_config = GraphitiLLMConfig(api_key=api_key)
                return GeminiRerankerClient(config=reranker_config)

            case 'openai':
                if not HAS_OPENAI_RERANKER:
                    logger.warning('OpenAI reranker not available, skipping cross-encoder')
                    return None

                if not llm_config or not llm_config.providers.openai:
                    logger.warning('OpenAI config not found, skipping cross-encoder')
                    return None

                api_key = llm_config.providers.openai.api_key
                if not api_key:
                    logger.warning('OpenAI API key not configured, skipping cross-encoder')
                    return None

                reranker_config = GraphitiLLMConfig(api_key=api_key)
                return OpenAIRerankerClient(config=reranker_config)

            case _:
                logger.info(f'No reranker available for provider: {provider}')
                return None
