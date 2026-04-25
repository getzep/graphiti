from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = Field(None)
    model_name: str | None = Field(None)
    embedding_model_name: str | None = Field(None)

    # Graph database provider
    graph_db_provider: Literal['neo4j', 'falkordb', 'neptune', 'kuzu'] = Field('neo4j')

    # Neo4j settings
    neo4j_uri: str | None = Field(None)
    neo4j_user: str | None = Field(None)
    neo4j_password: str | None = Field(None)
    neo4j_database: str = Field('neo4j')

    # FalkorDB settings
    falkordb_host: str = Field('localhost')
    falkordb_port: int = Field(6379)
    falkordb_username: str | None = Field(None)
    falkordb_password: str | None = Field(None)
    falkordb_database: str = Field('default_db')

    # Neptune settings
    neptune_host: str | None = Field(None)
    neptune_port: int = Field(8182)
    aoss_host: str | None = Field(None)
    aoss_port: int = Field(443)

    # Kuzu settings
    kuzu_db: str = Field(':memory:')
    kuzu_max_concurrent_queries: int = Field(1)

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
