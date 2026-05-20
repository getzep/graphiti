import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve
from graph_service.zep_graphiti import initialize_graphiti


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Configure graphiti_core logging after uvicorn has set up its handlers.
    uvicorn_handlers = logging.getLogger('uvicorn.error').handlers
    graphiti_logger = logging.getLogger('graphiti_core')
    if uvicorn_handlers:
        graphiti_logger.handlers = uvicorn_handlers
    else:
        graphiti_logger.addHandler(logging.StreamHandler())
    graphiti_logger.setLevel(logging.WARNING)

    settings = get_settings()
    await initialize_graphiti(settings)
    yield
    # Shutdown
    # No need to close Graphiti here, as it's handled per-request


app = FastAPI(lifespan=lifespan)


app.include_router(retrieve.router)
app.include_router(ingest.router)


@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(content={'status': 'healthy'}, status_code=200)
