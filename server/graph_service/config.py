from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends
from graphiti_core.embedder.client import EMBEDDING_DIM
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    openai_api_key: str | None = Field(
        None,
        validation_alias=AliasChoices('ARK_API_KEY', 'OPENAI_API_KEY', 'openai_api_key'),
        repr=False,
    )
    openai_base_url: str | None = Field(
        None,
        validation_alias=AliasChoices('ARK_BASE_URL', 'OPENAI_BASE_URL', 'openai_base_url'),
    )
    model_name: str | None = Field(
        None,
        validation_alias=AliasChoices(
            'ARK_CHAT_MODEL',
            'OPENAI_MODEL',
            'OPENAI_MODEL_NAME',
            'MODEL_NAME',
            'model_name',
        ),
    )
    llm_temperature: float = Field(
        0.0,
        validation_alias=AliasChoices(
            'ARK_TEMPERATURE', 'OPENAI_TEMPERATURE', 'LLM_TEMPERATURE', 'llm_temperature'
        ),
    )
    llm_max_tokens: int = Field(
        8192,
        ge=256,
        validation_alias=AliasChoices(
            'ARK_MAX_TOKENS', 'OPENAI_MAX_TOKENS', 'LLM_MAX_TOKENS', 'llm_max_tokens'
        ),
    )
    structured_output_mode: Literal['json_schema', 'json_object', 'prompt_only'] = Field(
        'prompt_only',
        validation_alias=AliasChoices(
            'ARK_STRUCTURED_OUTPUT_MODE',
            'OPENAI_STRUCTURED_OUTPUT_MODE',
            'STRUCTURED_OUTPUT_MODE',
            'structured_output_mode',
        ),
    )

    embedding_provider: Literal['local_hash', 'openai'] = Field('local_hash')
    embedding_api_key: str | None = Field(
        None,
        validation_alias=AliasChoices(
            'ARK_EMBEDDING_API_KEY',
            'OPENAI_EMBEDDING_API_KEY',
            'EMBEDDING_API_KEY',
            'embedding_api_key',
        ),
        repr=False,
    )
    embedding_base_url: str | None = Field(
        None,
        validation_alias=AliasChoices(
            'ARK_EMBEDDING_BASE_URL',
            'OPENAI_EMBEDDING_BASE_URL',
            'EMBEDDING_BASE_URL',
            'embedding_base_url',
        ),
    )
    embedding_model_name: str | None = Field(
        None,
        validation_alias=AliasChoices(
            'ARK_EMBEDDING_MODEL',
            'OPENAI_EMBEDDING_MODEL',
            'OPENAI_EMBEDDING_MODEL_NAME',
            'EMBEDDING_MODEL_NAME',
            'embedding_model_name',
        ),
    )
    embedding_dim: int = Field(
        EMBEDDING_DIM,
        ge=32,
        validation_alias=AliasChoices('EMBEDDING_DIM', 'embedding_dim'),
    )

    neo4j_uri: str | None = Field(None)
    neo4j_user: str | None = Field(None)
    neo4j_password: str | None = Field(None, repr=False)
    neo4j_database: str = Field('neo4j')
    falkordb_host: str | None = Field(None)
    falkordb_port: int | None = Field(None)
    falkordb_database: str | None = Field(None)
    db_backend: Literal['neo4j', 'falkordb'] = Field('neo4j')

    source_state_path: Path = Field(Path('data/source_state.db'))
    upload_root: Path = Field(Path('data/uploads'))
    max_upload_bytes: int = Field(25 * 1024 * 1024, ge=1024)
    sync_concurrency: int = Field(1, ge=1, le=8)

    oauth_public_base_url: str | None = Field(None)
    oauth_token_encryption_key: str | None = Field(None, repr=False)
    oauth_state_ttl_seconds: int = Field(600, ge=60, le=1800)
    oauth_cookie_secure: bool = Field(False)

    feishu_app_id: str | None = Field(None)
    feishu_app_secret: str | None = Field(None, repr=False)
    feishu_base_url: str = Field('https://open.feishu.cn/open-apis')
    feishu_authorize_url: str = Field(
        'https://accounts.feishu.cn/open-apis/authen/v1/authorize'
    )
    feishu_token_url: str = Field('https://accounts.feishu.cn/oauth/v3/token')
    feishu_oauth_scopes: str = Field(
        'offline_access space:document:retrieve docx:document:readonly drive:file:readonly'
    )

    meego_plugin_id: str | None = Field(None)
    meego_plugin_secret: str | None = Field(None, repr=False)
    meego_user_key: str | None = Field(None, repr=False)
    meego_base_url: str = Field('https://project.feishu.cn/open_api')
    meego_host: str = Field('project.feishu.cn')
    mcp_public_url: str = Field('http://localhost:8001/mcp/')

    model_config = SettingsConfigDict(
        extra='ignore',
        populate_by_name=True,
        env_ignore_empty=True,
    )


@lru_cache
def get_settings():
    return Settings(_env_file='.env')  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
