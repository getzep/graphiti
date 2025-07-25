from functools import lru_cache
from typing import Annotated
import os

from fastapi import Depends
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: str | None = Field(None, env="OPENAI_BASE_URL")
    model_name: str | None = Field(None, env="MODEL_NAME")
    embedding_model_name: str | None = Field(None, env="EMBEDDING_MODEL_NAME")
    # Backend selection: 'neo4j' or 'falkordb'
    db_backend: str | None = Field(None, alias="GRAPHITI_BACKEND", env="GRAPHITI_BACKEND", description="Database backend: 'neo4j' or 'falkordb'")
    # Neo4j settings (optional, validated below)
    neo4j_uri: str | None = Field(None, env="NEO4J_URI")
    neo4j_user: str | None = Field(None, env="NEO4J_USER")
    neo4j_password: str | None = Field(None, env="NEO4J_PASSWORD")
    # FalkorDB settings (optional, validated below)
    falkordb_host: str | None = Field(None, env="FALKORDB_HOST")
    falkordb_port: int | None = Field(None, env="FALKORDB_PORT")
    falkordb_database: str = Field("default_db", env="FALKORDB_DATABASE")
    falkordb_user: str | None = Field(None, env="FALKORDB_USER")
    falkordb_password: str | None = Field(None, env="FALKORDB_PASSWORD")

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', env_prefix='', extra='ignore', env_priority='env', populate_by_name=True)

    @model_validator(mode="after")
    def check_backend_fields(self):
        # Set default if not provided
        if self.db_backend is None:
            object.__setattr__(self, 'db_backend', 'neo4j')
        if self.db_backend == 'neo4j':
            missing = [k for k in ['neo4j_uri', 'neo4j_user', 'neo4j_password'] if getattr(self, k) is None]
            if missing:
                raise ValueError(f"Missing required Neo4j settings: {', '.join(missing)}")
        elif self.db_backend == 'falkordb':
            missing = [k for k in ['falkordb_host', 'falkordb_port'] if getattr(self, k) is None]
            if missing:
                raise ValueError(f"Missing required FalkorDB settings: {', '.join(missing)}")
        else:
            raise ValueError("db_backend must be either 'neo4j' or 'falkordb'")
        return self


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
