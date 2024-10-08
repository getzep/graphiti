from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    openai_api_key: str | None = Field(None)
    openai_base_url: str | None = Field(None)
    
    groq_api_key: str | None = Field(None)
    anthropic_api_key: str | None = Field(None)
    voyage_api_key: str | None = Field(None)

    llm_provider: Literal['openai', 'groq', 'anthropic'] | None = Field(None)
    model_name: str | None = Field(None)
    
    embedding_provider: Literal['openai', 'voyage'] | None = Field(None)
    embedding_model_name: str | None = Field(None)

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


@lru_cache
def get_settings():
    return Settings()


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
