import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from graph_service.config import ZepEnvDep

bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    settings: ZepEnvDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> None:
    """Fail-closed bearer-token guard for all ingest/retrieve endpoints.

    A request is only authorized when an ``api_key`` is configured and the caller
    presents a matching ``Authorization: Bearer <api_key>`` credential. If no key
    is configured the service fails closed and rejects every request.
    """
    if (
        settings.api_key is None
        or credentials is None
        or not secrets.compare_digest(credentials.credentials, settings.api_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Missing or invalid API key',
            headers={'WWW-Authenticate': 'Bearer'},
        )
