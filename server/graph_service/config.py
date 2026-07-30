from functools import lru_cache
from typing import Annotated, Any, Literal

from fastapi import Depends
from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


def _blank_to_none(value: Any) -> Any:
    """Treat an env var that is set but empty as unset."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _strip(value: Any) -> Any:
    """Trim surrounding whitespace from an env var that has to be present."""
    return value.strip() if isinstance(value, str) else value


# A .env file and a dashboard UI both make it easy to define a variable with an empty
# value, which is not the same thing as leaving it out: '' arrives here as a real setting
# and gets passed on to the clients. A blank model_name would be sent to OpenAI as the
# model to use, and a blank falkordb_port would fail validation before the app can boot.
#
# One case this cannot reach: the OpenAI SDK falls back to reading OPENAI_BASE_URL from
# the environment when it is handed base_url=None, so a blank OPENAI_BASE_URL still ends
# up as '' inside the client, and requests go out with no host. Leave it unset, not empty.
OptionalStr = Annotated[str | None, BeforeValidator(_blank_to_none)]
OptionalInt = Annotated[int | None, BeforeValidator(_blank_to_none)]

# The same idea for a setting that has to be present: strip first, so whitespace fails min_length
# rather than passing as a one-space secret, and a key pasted with a trailing newline survives.
RequiredStr = Annotated[str, BeforeValidator(_strip)]

# Entropy floor for graphiti_api_key. auth.py's budget stops a stranger working through the
# keyspace; this stops a key already at the front of it. Named so tests/test_auth.py can pin the
# boundary against it rather than restating the number.
MIN_API_KEY_LENGTH = 16


class Settings(BaseSettings):
    # Rejected when blank rather than defaulted, so a missing key fails the deploy at
    # startup instead of turning every background ingestion into a 401 nobody is watching.
    openai_api_key: RequiredStr = Field(min_length=1)
    openai_base_url: OptionalStr = None
    model_name: OptionalStr = None
    embedding_model_name: OptionalStr = None
    neo4j_uri: OptionalStr = None
    neo4j_user: OptionalStr = None
    neo4j_password: OptionalStr = None
    falkordb_host: OptionalStr = None
    falkordb_port: OptionalInt = None
    falkordb_username: OptionalStr = None
    falkordb_password: OptionalStr = None
    falkordb_database: OptionalStr = None
    # Only these two backends are wired up in zep_graphiti, so a typo should be a startup
    # error naming the valid values, not a silent fall-through to the Neo4j branch.
    db_backend: Literal['neo4j', 'falkordb'] = 'neo4j'
    # Bearer token for every endpoint except /healthcheck, mandatory: this API writes to a shared
    # graph and spends openai_api_key on every episode. Missing fails startup, because an open
    # service and one that 401s everything both look healthy to Render.
    #
    # Printable ASCII is what survives an HTTP header: values travel as latin-1 and clients
    # disagree about encoding anything outside it, so a key with an accent authenticates for some
    # and not others. Enforced here rather than in auth.py so a bad rotation fails the deploy
    # naming the problem, instead of 401ing the operator who set it.
    #
    # Caveat: pydantic echoes the offending value into the deploy log. Only ever a key that never
    # authenticated, and scrubbing it means catching ValidationError and degrading every other
    # setting's message.
    graphiti_api_key: RequiredStr = Field(min_length=MIN_API_KEY_LENGTH, pattern=r'^[\x20-\x7e]+$')

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
