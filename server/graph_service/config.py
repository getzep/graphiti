from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

from graphiti_core import Graphiti


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = Field(None)
    model_name: str | None = Field(None)
    embedding_model_name: str | None = Field(None)
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


@lru_cache
def get_settings():
    return Settings()


ZepEnvDep = Annotated[Settings, Depends(get_settings)]


@lru_cache
def get_graphiti(settings: ZepEnvDep) -> Graphiti:
    """Get a Graphiti instance."""
    return Graphiti(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        openai_api_key=settings.openai_api_key,
        openai_base_url=settings.openai_base_url,
        model_name=settings.model_name,
        embedding_model_name=settings.embedding_model_name,
    )
