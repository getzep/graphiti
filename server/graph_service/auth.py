"""Bearer-token auth for the graph endpoints.

Every endpoint except /healthcheck requires GRAPHITI_API_KEY, with no way to switch it off: the
API writes to a shared graph and spends the deployment's OpenAI key. config.py requires the key
at startup, so this module can assume there is one.
"""

import hashlib
import secrets
import time
from collections import deque
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from graph_service.config import ZepEnvDep

# auto_error=False so a missing or non-Bearer header arrives as None: HTTPBearer would answer 403
# on its own, and the right answer is a 401 naming the scheme to retry with.
_bearer = HTTPBearer(auto_error=False, description='The GRAPHITI_API_KEY set on this service.')

# Rejections allowed per window before answering 429, so the key can't be worked through.
#
# Global, not per-IP: behind Render's proxy every request carries the proxy's address, and
# trusting X-Forwarded-For would hand an attacker the bucket key. It can't lock out a real client,
# because require_api_key checks the key before the budget. In-process, so N instances allow N
# times this — a shared counter would put FalkorDB on the auth path.
MAX_FAILED_AUTH = 10
FAILED_AUTH_WINDOW_SECONDS = 60

# The last MAX_FAILED_AUTH rejection times, oldest first, so the budget is spent exactly when the
# deque is full and its oldest entry is still in the window. monotonic, so an NTP correction can't
# resize the window. No lock: single event loop.
_recent_rejections: deque[float] = deque(maxlen=MAX_FAILED_AUTH)


def _seconds_until_budget_frees(now: float) -> float | None:
    """Time until the next rejection is allowed, or None if it already is.

    Positive only when the budget is spent, so it serves as both the check and Retry-After.
    """
    if len(_recent_rejections) < MAX_FAILED_AUTH:
        return None
    elapsed = now - _recent_rejections[0]
    return FAILED_AUTH_WINDOW_SECONDS - elapsed if elapsed < FAILED_AUTH_WINDOW_SECONDS else None


def _digest(value: str) -> bytes:
    """A fixed-width, constant-time-comparable stand-in for the key.

    Encoded because Starlette decodes headers as latin-1, so this can be any str. Hashed so both
    sides are 32 bytes: compare_digest returns early on a length mismatch, leaking key length.
    """
    return hashlib.sha256(value.encode()).digest()


async def require_api_key(
    settings: ZepEnvDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    # Before the budget, so the right key is never rate limited — see MAX_FAILED_AUTH.
    # compare_digest so timing can't leak the key; it needs both sides, hence the guard.
    if credentials is not None and secrets.compare_digest(
        _digest(credentials.credentials), _digest(settings.graphiti_api_key)
    ):
        return

    now = time.monotonic()
    retry_after = _seconds_until_budget_frees(now)
    if retry_after is not None:
        # Not recorded: counting refused requests would hold the window open under sustained
        # traffic, so a wrong key would never get its 401 back.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail='Too many failed authentication attempts. Try again shortly.',
            headers={'Retry-After': str(int(retry_after) + 1)},  # ceil: never advertise 0s
        )

    _recent_rejections.append(now)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Invalid or missing API key. Send it as: Authorization: Bearer <GRAPHITI_API_KEY>',
        headers={'WWW-Authenticate': 'Bearer'},  # required on a 401 by RFC 9110
    )
