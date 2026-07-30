from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from graph_service.auth import require_api_key
from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve
from graph_service.zep_graphiti import initialize_graphiti, shutdown_graphiti


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    await initialize_graphiti(settings)
    yield
    await shutdown_graphiti()


# /docs, /redoc and /openapi.json stay public, departing from the usual advice to gate them: this
# template's source is public, so the schema is the same map routers/ are, and a browser can't put
# an Authorization header on a plain navigation. Swagger's Authorize button picks the scheme up
# from the dependency below. Revisit if this ever ships as a private service.
app = FastAPI(title='Graphiti API', lifespan=lifespan)


# At the router, not per-route, so a new endpoint is closed by default and anything public is an
# explicit declaration below. tests/test_auth.py holds the whole route table to that.
app.include_router(retrieve.router, dependencies=[Depends(require_api_key)])
app.include_router(ingest.router, dependencies=[Depends(require_api_key)])


# Public: Render polls this to decide whether the service is live.
@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(content={'status': 'healthy'}, status_code=200)
