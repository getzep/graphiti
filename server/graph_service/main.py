import logging
from contextlib import asynccontextmanager

import openai
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from graph_service.auth import require_api_key
from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve
from graph_service.zep_graphiti import initialize_graphiti, shutdown_graphiti

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    await initialize_graphiti(settings)
    yield
    await shutdown_graphiti()


# /docs, /redoc and /openapi.json stay public, against the usual advice: this template's source is
# public, so the schema maps nothing routers/ doesn't, and a browser can't put an Authorization
# header on a plain navigation. Swagger's Authorize button picks the scheme up from the dependency
# below. Revisit if this ever ships as a private service.
app = FastAPI(title='Graphiti API', lifespan=lifespan)


# At the router, not per-route, so a new endpoint is closed by default and anything public is an
# explicit declaration below. tests/test_auth.py holds the whole route table to that.
app.include_router(retrieve.router, dependencies=[Depends(require_api_key)])
app.include_router(ingest.router, dependencies=[Depends(require_api_key)])


# Public: Render polls this to decide whether the service is live.
@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(content={'status': 'healthy'}, status_code=200)


def _upstream_failure(exc: openai.APIError) -> tuple[int, str]:
    """Map an OpenAI failure to a status and a message that names what to go and fix.

    Every retrieval endpoint embeds its query before it can search, so an unusable
    OPENAI_API_KEY surfaces here rather than at ingestion time. Unmapped, it reached the client
    as a bare 500 `Internal Server Error`, which reads as a bug in the graph query — the actual
    cause was only in the service logs, and only as a traceback.
    """
    if isinstance(exc, openai.RateLimitError):
        # 429 either way, because OpenAI returns 429 for a real rate limit and for
        # insufficient_quota alike, and only the caller can tell whether retrying is worth it.
        # The distinction that matters to the operator is in the message, so pass it through.
        return 429, (
            'OpenAI rejected the request: either a rate limit, or the quota on the account '
            f'behind OPENAI_API_KEY is exhausted. Upstream said: {exc}'
        )

    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        # 502, not 401/403: the caller's GRAPHITI_API_KEY was accepted, and echoing OpenAI's
        # status would tell them to go fix a credential they do not hold.
        return 502, (
            "OpenAI rejected this service's credentials. Check that OPENAI_API_KEY is set to a "
            f'valid key with access to the configured model. Upstream said: {exc}'
        )

    if isinstance(exc, openai.APIStatusError):
        return 502, f'OpenAI returned an error. Upstream said: {exc}'

    # Everything else is a connection failure or a timeout: no response ever arrived.
    return 504, f'Could not reach OpenAI. Upstream said: {exc}'


# Registered on the base class, so a subclass the SDK adds later is still mapped rather than
# falling through to a 500. Starlette resolves handlers along the MRO.
@app.exception_handler(openai.APIError)
async def handle_openai_error(_: Request, exc: openai.APIError) -> JSONResponse:
    status_code, detail = _upstream_failure(exc)
    # Logged as well as returned: the response goes to whoever made the request, and the operator
    # reading logs is usually someone else.
    logger.error('OpenAI call failed, returning %s: %s', status_code, exc)
    return JSONResponse(content={'detail': detail}, status_code=status_code)
