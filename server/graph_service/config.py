from functools import lru_cache
from typing import Annotated

from fastapi import Depends  # type: ignore
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    # LLM settings
    openai_api_key: str
    openai_base_url: str | None = None
    model_name: str | None = None
    embedding_model_name: str | None = None
    
    # Backend selection: 'neo4j' or 'falkordb'
    db_backend: str | None = None
    
    # Neo4j settings (optional, validated below)
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    
    # FalkorDB settings (optional, validated below)
    falkordb_host: str = ""
    falkordb_port: int = 6379
    falkordb_user: str | None = None
    falkordb_password: str | None = None

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        env_prefix='',
        extra='ignore',
        env_priority='env',
        populate_by_name=True
    )

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