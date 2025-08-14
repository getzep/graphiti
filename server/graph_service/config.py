from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    # Azure OpenAI settings
    api_key: str = Field(alias='API_KEY')
    api_version: str | None = Field(None, alias='API_VERSION')
    llm_endpoint: str | None = Field(None, alias='LLM_ENDPOINT')
    embedding_endpoint: str | None = Field(None, alias='EMBEDDING_ENDPOINT')
    embedding_model: str | None = Field(None, alias='EMBEDDING_MODEL')
    model_name: str | None = Field(None, alias='MODEL_NAME')
    small_model_name: str | None = Field(None, alias='SMALL_MODEL_NAME')
    
    # Legacy OpenAI settings (optional for backward compatibility)
    openai_api_key: str | None = Field(None)
    openai_base_url: str | None = Field(None)
    
    # Neo4j settings
    neo4j_uri: str = Field(alias='NEO4J_URI')
    neo4j_user: str = Field(alias='NEO4J_USER')
    neo4j_password: str = Field(alias='NEO4J_PASSWORD')

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
