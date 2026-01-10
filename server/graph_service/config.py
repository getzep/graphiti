import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


def _resolve_database_provider() -> str:
    """Resolve database provider from multiple possible environment variables."""
    # Check for explicit DATABASE_PROVIDER first
    provider = os.environ.get('DATABASE_PROVIDER', '').strip().lower()
    if provider:
        return provider
    # Check legacy environment variables for backward compatibility
    db_backend = os.environ.get('db_backend', '').strip().lower()
    if db_backend:
        return db_backend
    graphiti_backend = os.environ.get('GRAPHITI_BACKEND', '').strip().lower()
    if graphiti_backend:
        return graphiti_backend
    # Default to 'falkordb'
    return 'falkordb'


class Settings(BaseSettings):
    openai_api_key: str = Field(default='')
    openai_base_url: str | None = Field(None)
    model_name: str | None = Field(None)
    embedding_model_name: str | None = Field(None)
    # Database provider: 'neo4j' or 'falkordb'
    # Supports both DATABASE_PROVIDER and legacy db_backend/GRAPHITI_BACKEND env vars
    database_provider: str = Field(default='falkordb')
    # Neo4j settings (used when database_provider='neo4j')
    neo4j_uri: str | None = Field(None)
    neo4j_user: str | None = Field(None)
    neo4j_password: str | None = Field(None)
    # FalkorDB settings (used when database_provider='falkordb')
    falkordb_host: str = Field(default='localhost')
    falkordb_port: int = Field(default=6380)
    falkordb_password: str | None = Field(None)
    falkordb_database: str = Field(default='default_db')

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    @model_validator(mode='after')
    def resolve_database_provider(self):
        """Resolve database provider from environment variables, supporting legacy vars."""
        # Always check environment variables to support legacy db_backend/GRAPHITI_BACKEND
        # This allows docker-compose.yml to work with existing environment variable names
        resolved = _resolve_database_provider()
        # Override with resolved value from environment (unless explicitly set via DATABASE_PROVIDER)
        # This ensures legacy env vars (db_backend, GRAPHITI_BACKEND) take precedence
        if resolved and resolved != self.database_provider.lower():
            self.database_provider = resolved
        # Normalize to lowercase
        self.database_provider = self.database_provider.lower()
        return self


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
