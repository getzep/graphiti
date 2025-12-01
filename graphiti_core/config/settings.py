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
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from .providers import (
    DEFAULT_EMBEDDINGS,
    DEFAULT_MODELS,
    DatabaseProvider,
    EmbedderProvider,
    LLMProvider,
    RerankerProvider,
)


class LLMProviderConfig(BaseModel):
    """Configuration for LLM provider.

    This configuration supports multiple LLM providers including OpenAI, Azure OpenAI,
    Anthropic, Gemini, Groq, and generic providers via LiteLLM.

    Examples:
        >>> # OpenAI configuration
        >>> config = LLMProviderConfig(
        ...     provider=LLMProvider.OPENAI,
        ...     api_key='sk-...',
        ... )

        >>> # Azure OpenAI configuration
        >>> config = LLMProviderConfig(
        ...     provider=LLMProvider.AZURE_OPENAI,
        ...     api_key='...',
        ...     base_url='https://your-resource.openai.azure.com',
        ...     azure_deployment='your-deployment-name',
        ... )

        >>> # Anthropic configuration
        >>> config = LLMProviderConfig(
        ...     provider=LLMProvider.ANTHROPIC,
        ...     model='claude-sonnet-4-5-latest',
        ... )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description='The LLM provider to use',
    )
    model: str | None = Field(
        default=None,
        description='The model name to use. If not provided, uses provider default.',
    )
    small_model: str | None = Field(
        default=None,
        description='Smaller/faster model for simpler tasks. If not provided, uses provider default.',
    )
    api_key: str | None = Field(
        default=None,
        description='API key for the provider. Falls back to environment variables if not provided.',
    )
    base_url: str | None = Field(
        default=None,
        description='Base URL for the API. Required for Azure OpenAI and custom endpoints.',
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description='Temperature for response generation',
    )
    max_tokens: int = Field(
        default=8192,
        gt=0,
        description='Maximum tokens for response generation',
    )

    # Azure-specific fields
    azure_deployment: str | None = Field(
        default=None,
        description='Azure OpenAI deployment name (required for Azure provider)',
    )
    azure_api_version: str | None = Field(
        default='2024-10-21',
        description='Azure OpenAI API version',
    )

    # LiteLLM-specific fields
    litellm_model: str | None = Field(
        default=None,
        description='Full LiteLLM model string (e.g., "azure/gpt-4", "bedrock/claude-3")',
    )

    # Custom provider fields
    custom_client_class: str | None = Field(
        default=None,
        description='Fully qualified class name for custom LLM client',
    )

    @model_validator(mode='after')
    def set_defaults_and_validate(self) -> 'LLMProviderConfig':
        """Set provider-specific defaults and validate configuration."""
        # Set default models if not provided
        if self.model is None and self.provider in DEFAULT_MODELS:
            self.model = DEFAULT_MODELS[self.provider]['model']

        if self.small_model is None and self.provider in DEFAULT_MODELS:
            self.small_model = DEFAULT_MODELS[self.provider]['small_model']

        # Set API key from environment if not provided
        if self.api_key is None:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.getenv('OPENAI_API_KEY')
            elif self.provider == LLMProvider.AZURE_OPENAI:
                self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
            elif self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.getenv('ANTHROPIC_API_KEY')
            elif self.provider == LLMProvider.GEMINI:
                self.api_key = os.getenv('GOOGLE_API_KEY')
            elif self.provider == LLMProvider.GROQ:
                self.api_key = os.getenv('GROQ_API_KEY')

        # Validate Azure-specific requirements
        if self.provider == LLMProvider.AZURE_OPENAI:
            if not self.base_url:
                raise ValueError('base_url is required for Azure OpenAI provider')
            if not self.azure_deployment and not self.model:
                raise ValueError(
                    'Either azure_deployment or model must be provided for Azure OpenAI'
                )

        # Validate LiteLLM requirements
        if self.provider == LLMProvider.LITELLM and not self.litellm_model:
            raise ValueError('litellm_model is required for LiteLLM provider')

        # Validate custom provider requirements
        if self.provider == LLMProvider.CUSTOM and not self.custom_client_class:
            raise ValueError('custom_client_class is required for custom provider')

        return self


class EmbedderConfig(BaseModel):
    """Configuration for embedding provider.

    Examples:
        >>> # OpenAI embeddings
        >>> config = EmbedderConfig(
        ...     provider=EmbedderProvider.OPENAI,
        ... )

        >>> # Voyage AI embeddings
        >>> config = EmbedderConfig(
        ...     provider=EmbedderProvider.VOYAGE,
        ...     model='voyage-3',
        ... )
    """

    provider: EmbedderProvider = Field(
        default=EmbedderProvider.OPENAI,
        description='The embedder provider to use',
    )
    model: str | None = Field(
        default=None,
        description='The embedding model name. If not provided, uses provider default.',
    )
    api_key: str | None = Field(
        default=None,
        description='API key for the provider. Falls back to environment variables if not provided.',
    )
    base_url: str | None = Field(
        default=None,
        description='Base URL for the API. Required for Azure OpenAI.',
    )
    dimensions: int | None = Field(
        default=None,
        description='Embedding dimensions. If not provided, uses provider default.',
    )

    # Azure-specific fields
    azure_deployment: str | None = Field(
        default=None,
        description='Azure OpenAI deployment name (required for Azure provider)',
    )
    azure_api_version: str | None = Field(
        default='2024-10-21',
        description='Azure OpenAI API version',
    )

    # Custom provider fields
    custom_client_class: str | None = Field(
        default=None,
        description='Fully qualified class name for custom embedder client',
    )

    @model_validator(mode='after')
    def set_defaults_and_validate(self) -> 'EmbedderConfig':
        """Set provider-specific defaults and validate configuration."""
        # Set default model and dimensions if not provided
        if self.provider in DEFAULT_EMBEDDINGS:
            if self.model is None:
                self.model = DEFAULT_EMBEDDINGS[self.provider]['model']
            if self.dimensions is None:
                self.dimensions = DEFAULT_EMBEDDINGS[self.provider]['dimensions']

        # Set API key from environment if not provided
        if self.api_key is None:
            if self.provider == EmbedderProvider.OPENAI:
                self.api_key = os.getenv('OPENAI_API_KEY')
            elif self.provider == EmbedderProvider.AZURE_OPENAI:
                self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
            elif self.provider == EmbedderProvider.VOYAGE:
                self.api_key = os.getenv('VOYAGE_API_KEY')
            elif self.provider == EmbedderProvider.GEMINI:
                self.api_key = os.getenv('GOOGLE_API_KEY')

        # Validate Azure-specific requirements
        if self.provider == EmbedderProvider.AZURE_OPENAI and not self.base_url:
            raise ValueError('base_url is required for Azure OpenAI embedder')

        # Validate custom provider requirements
        if self.provider == EmbedderProvider.CUSTOM and not self.custom_client_class:
            raise ValueError('custom_client_class is required for custom embedder')

        return self


class RerankerConfig(BaseModel):
    """Configuration for reranker/cross-encoder provider.

    Examples:
        >>> config = RerankerConfig(
        ...     provider=RerankerProvider.OPENAI,
        ... )
    """

    provider: RerankerProvider = Field(
        default=RerankerProvider.OPENAI,
        description='The reranker provider to use',
    )
    api_key: str | None = Field(
        default=None,
        description='API key for the provider. Falls back to environment variables if not provided.',
    )
    base_url: str | None = Field(
        default=None,
        description='Base URL for the API.',
    )

    # Azure-specific fields
    azure_deployment: str | None = Field(
        default=None,
        description='Azure OpenAI deployment name (required for Azure provider)',
    )

    # Custom provider fields
    custom_client_class: str | None = Field(
        default=None,
        description='Fully qualified class name for custom reranker client',
    )

    @model_validator(mode='after')
    def set_defaults(self) -> 'RerankerConfig':
        """Set provider-specific defaults."""
        # Set API key from environment if not provided
        if self.api_key is None:
            if self.provider == RerankerProvider.OPENAI:
                self.api_key = os.getenv('OPENAI_API_KEY')
            elif self.provider == RerankerProvider.AZURE_OPENAI:
                self.api_key = os.getenv('AZURE_OPENAI_API_KEY')

        return self


class DatabaseConfig(BaseModel):
    """Configuration for graph database.

    Examples:
        >>> # Neo4j configuration
        >>> config = DatabaseConfig(
        ...     provider=DatabaseProvider.NEO4J,
        ...     uri='bolt://localhost:7687',
        ...     user='neo4j',
        ...     password='password',
        ... )

        >>> # FalkorDB configuration
        >>> config = DatabaseConfig(
        ...     provider=DatabaseProvider.FALKORDB,
        ...     uri='redis://localhost:6379',
        ... )
    """

    provider: DatabaseProvider = Field(
        default=DatabaseProvider.NEO4J,
        description='The graph database provider to use',
    )
    uri: str | None = Field(
        default=None,
        description='Database connection URI',
    )
    user: str | None = Field(
        default=None,
        description='Database username',
    )
    password: str | None = Field(
        default=None,
        description='Database password',
    )
    database: str | None = Field(
        default=None,
        description='Database name. Uses provider default if not specified.',
    )

    # Custom provider fields
    custom_driver_class: str | None = Field(
        default=None,
        description='Fully qualified class name for custom database driver',
    )

    @model_validator(mode='after')
    def validate_database_config(self) -> 'DatabaseConfig':
        """Validate database configuration."""
        if self.provider == DatabaseProvider.CUSTOM and not self.custom_driver_class:
            raise ValueError('custom_driver_class is required for custom database provider')
        return self


class GraphitiConfig(BaseModel):
    """Main Graphiti configuration.

    This is the primary configuration class that aggregates all provider configurations.
    It supports loading from YAML files, environment variables, and programmatic configuration.

    Examples:
        >>> # Programmatic configuration
        >>> config = GraphitiConfig(
        ...     llm=LLMProviderConfig(provider=LLMProvider.ANTHROPIC),
        ...     embedder=EmbedderConfig(provider=EmbedderProvider.VOYAGE),
        ...     database=DatabaseConfig(
        ...         uri='bolt://localhost:7687',
        ...         user='neo4j',
        ...         password='password',
        ...     ),
        ... )

        >>> # Load from YAML file
        >>> config = GraphitiConfig.from_yaml('graphiti.yaml')

        >>> # Load from environment (looks for GRAPHITI_CONFIG_PATH)
        >>> config = GraphitiConfig.from_env()
    """

    llm: LLMProviderConfig = Field(
        default_factory=LLMProviderConfig,
        description='LLM provider configuration',
    )
    embedder: EmbedderConfig = Field(
        default_factory=EmbedderConfig,
        description='Embedder provider configuration',
    )
    reranker: RerankerConfig = Field(
        default_factory=RerankerConfig,
        description='Reranker provider configuration',
    )
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description='Database provider configuration',
    )

    # General settings
    store_raw_episode_content: bool = Field(
        default=True,
        description='Whether to store raw episode content in the database',
    )
    max_coroutines: int | None = Field(
        default=None,
        description='Maximum number of concurrent operations',
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'GraphitiConfig':
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            GraphitiConfig instance loaded from the file

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the YAML file is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Configuration file not found: {path}')

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        return cls(**config_dict)

    @classmethod
    def from_env(cls, env_var: str = 'GRAPHITI_CONFIG_PATH') -> 'GraphitiConfig':
        """Load configuration from a YAML file specified in an environment variable.

        Args:
            env_var: Name of the environment variable containing the config file path

        Returns:
            GraphitiConfig instance loaded from the file, or default config if env var not set

        Raises:
            FileNotFoundError: If the specified config file doesn't exist
        """
        config_path = os.getenv(env_var)
        if config_path:
            return cls.from_yaml(config_path)

        # Look for default config files in current directory
        for default_file in ['.graphiti.yaml', '.graphiti.yml', 'graphiti.yaml', 'graphiti.yml']:
            if Path(default_file).exists():
                return cls.from_yaml(default_file)

        # Return default configuration
        return cls()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path where the configuration file should be saved
        """
        path = Path(path)
        # Use json mode to convert enums to their values
        config_dict = self.model_dump(exclude_none=True, mode='json')

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
