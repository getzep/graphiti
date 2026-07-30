import logging
from contextlib import asynccontextmanager

import openai
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from graph_service.auth import require_api_key
from graph_service.config import get_settings
from graph_service.openai_errors import describe_failure
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


# Registered on the base class, so a subclass the SDK adds later is still mapped rather than
# falling through to a 500. Starlette resolves handlers along the MRO; openai_errors.py decides
# which status each failure becomes.
@app.exception_handler(openai.APIError)
async def handle_openai_error(_: Request, exc: openai.APIError) -> JSONResponse:
    status_code, detail = describe_failure(exc)
    # Logged as well as returned: the response goes to whoever made the request, and the operator
    # reading logs is usually someone else.
    logger.error('OpenAI call failed, returning %s: %s', status_code, exc)
    return JSONResponse(content={'detail': detail}, status_code=status_code)
