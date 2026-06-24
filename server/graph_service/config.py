from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = Field(None)
    model_name: str | None = Field(None)
    embedding_model_name: str | None = Field(None)
    neo4j_uri: str | None = Field(None)
    neo4j_user: str | None = Field(None)
    neo4j_password: str | None = Field(None)
    falkordb_host: str | None = Field(None)
    falkordb_port: int | None = Field(None)
    falkordb_database: str | None = Field(None)
    db_backend: str = Field('neo4j')

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
