"""Configuration schemas with pydantic-settings and YAML support."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for loading from YAML files."""

    def __init__(self, settings_cls: type[BaseSettings], config_path: Path | None = None):
        super().__init__(settings_cls)
        self.config_path = config_path or Path('config.yaml')

    def _expand_env_vars(self, value: Any) -> Any:
        """Recursively expand environment variables in configuration values."""
        if isinstance(value, str):
            # Support ${VAR} and ${VAR:default} syntax
            import re

            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(3) if match.group(3) is not None else ''
                return os.environ.get(var_name, default_value)

            pattern = r'\$\{([^:}]+)(:([^}]*))?\}'

            # Check if the entire value is a single env var expression
            full_match = re.fullmatch(pattern, value)
            if full_match:
                result = replacer(full_match)
                # Convert boolean-like strings to actual booleans
                if isinstance(result, str):
                    lower_result = result.lower().strip()
                    if lower_result in ('true', '1', 'yes', 'on'):
                        return True
                    elif lower_result in ('false', '0', 'no', 'off'):
                        return False
                    elif lower_result == '':
                        # Empty string means env var not set - return None for optional fields
                        return None
                return result
            else:
                # Otherwise, do string substitution (keep as strings for partial replacements)
                return re.sub(pattern, replacer, value)
        elif isinstance(value, dict):
            return {k: self._expand_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._expand_env_vars(item) for item in value]
        return value

    def get_field_value(self, field_name: str, field_info: Any) -> Any:
        """Get field value from YAML config."""
        return None

    def __call__(self) -> dict[str, Any]:
        """Load and parse YAML configuration."""
        if not self.config_path.exists():
            return {}

        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f) or {}

        # Expand environment variables
        return self._expand_env_vars(raw_config)


class ServerConfig(BaseModel):
    """Server configuration."""

    transport: str = Field(
        default='http',
        description='Transport type: http (default, recommended), stdio, or sse (deprecated)',
    )
    host: str = Field(default='0.0.0.0', description='Server host')
    port: int = Field(default=8000, description='Server port')


class OpenAIProviderConfig(BaseModel):
    """OpenAI provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.openai.com/v1'
    organization_id: str | None = None


class AzureOpenAIProviderConfig(BaseModel):
    """Azure OpenAI provider configuration."""

    api_key: str | None = None
    api_url: str | None = None
    api_version: str = '2024-10-21'
    deployment_name: str | None = None
    use_azure_ad: bool = False


class AnthropicProviderConfig(BaseModel):
    """Anthropic provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.anthropic.com'
    max_retries: int = 3


class GeminiProviderConfig(BaseModel):
    """Gemini provider configuration."""

    api_key: str | None = None
    project_id: str | None = None
    location: str = 'us-central1'


class GroqProviderConfig(BaseModel):
    """Groq provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.groq.com/openai/v1'


class VoyageProviderConfig(BaseModel):
    """Voyage AI provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.voyageai.com/v1'
    model: str = 'voyage-3'


class LLMProvidersConfig(BaseModel):
    """LLM providers configuration."""

    openai: OpenAIProviderConfig | None = None
    azure_openai: AzureOpenAIProviderConfig | None = None
    anthropic: AnthropicProviderConfig | None = None
    gemini: GeminiProviderConfig | None = None
    groq: GroqProviderConfig | None = None


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str = Field(default='openai', description='LLM provider')
    model: str = Field(default='gpt-4o-mini', description='Model name')
    temperature: float | None = Field(
        default=None, description='Temperature (optional, defaults to None for reasoning models)'
    )
    max_tokens: int = Field(default=4096, description='Max tokens')
    providers: LLMProvidersConfig = Field(default_factory=LLMProvidersConfig)


class EmbedderProvidersConfig(BaseModel):
    """Embedder providers configuration."""

    openai: OpenAIProviderConfig | None = None
    azure_openai: AzureOpenAIProviderConfig | None = None
    gemini: GeminiProviderConfig | None = None
    voyage: VoyageProviderConfig | None = None


class EmbedderConfig(BaseModel):
    """Embedder configuration."""

    provider: str = Field(default='openai', description='Embedder provider')
    model: str = Field(default='text-embedding-3-small', description='Model name')
    dimensions: int = Field(default=1536, description='Embedding dimensions')
    providers: EmbedderProvidersConfig = Field(default_factory=EmbedderProvidersConfig)


class Neo4jProviderConfig(BaseModel):
    """Neo4j provider configuration."""

    uri: str = 'bolt://localhost:7687'
    username: str = 'neo4j'
    password: str | None = None
    database: str = 'neo4j'
    use_parallel_runtime: bool = False


class FalkorDBProviderConfig(BaseModel):
    """FalkorDB provider configuration."""

    uri: str = 'redis://localhost:6379'
    password: str | None = None
    database: str = 'default_db'


class DatabaseProvidersConfig(BaseModel):
    """Database providers configuration."""

    neo4j: Neo4jProviderConfig | None = None
    falkordb: FalkorDBProviderConfig | None = None


class DatabaseConfig(BaseModel):
    """Database configuration."""

    provider: str = Field(default='falkordb', description='Database provider')
    providers: DatabaseProvidersConfig = Field(default_factory=DatabaseProvidersConfig)


class EntityTypeConfig(BaseModel):
    """Entity type configuration."""

    name: str
    description: str


class GraphitiAppConfig(BaseModel):
    """Graphiti-specific configuration."""

    group_id: str = Field(default='main', description='Group ID')
    episode_id_prefix: str | None = Field(default='', description='Episode ID prefix')
    user_id: str = Field(default='mcp_user', description='User ID')
    entity_types: list[EntityTypeConfig] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        """Convert None to empty string for episode_id_prefix."""
        if self.episode_id_prefix is None:
            self.episode_id_prefix = ''


class GraphitiConfig(BaseSettings):
    """Graphiti configuration with YAML and environment support."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    graphiti: GraphitiAppConfig = Field(default_factory=GraphitiAppConfig)

    # Additional server options
    destroy_graph: bool = Field(default=False, description='Clear graph on startup')

    model_config = SettingsConfigDict(
        env_prefix='',
        env_nested_delimiter='__',
        case_sensitive=False,
        extra='ignore',
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to include YAML."""
        config_path = Path(os.environ.get('CONFIG_PATH', 'config/config.yaml'))
        yaml_settings = YamlSettingsSource(settings_cls, config_path)
        # Priority: CLI args (init) > env vars > yaml > defaults
        return (init_settings, env_settings, yaml_settings, dotenv_settings)

    def apply_cli_overrides(self, args) -> None:
        """Apply CLI argument overrides to configuration."""
        # Override server settings
        if hasattr(args, 'transport') and args.transport:
            self.server.transport = args.transport

        # Override LLM settings
        if hasattr(args, 'llm_provider') and args.llm_provider:
            self.llm.provider = args.llm_provider
        if hasattr(args, 'model') and args.model:
            self.llm.model = args.model
        if hasattr(args, 'temperature') and args.temperature is not None:
            self.llm.temperature = args.temperature

        # Override embedder settings
        if hasattr(args, 'embedder_provider') and args.embedder_provider:
            self.embedder.provider = args.embedder_provider
        if hasattr(args, 'embedder_model') and args.embedder_model:
            self.embedder.model = args.embedder_model

        # Override database settings
        if hasattr(args, 'database_provider') and args.database_provider:
            self.database.provider = args.database_provider

        # Override Graphiti settings
        if hasattr(args, 'group_id') and args.group_id:
            self.graphiti.group_id = args.group_id
        if hasattr(args, 'user_id') and args.user_id:
            self.graphiti.user_id = args.user_id
