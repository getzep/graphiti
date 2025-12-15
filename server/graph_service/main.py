from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve
from graph_service.zep_graphiti import create_graphiti


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    graphiti = create_graphiti(settings)
    await graphiti.build_indices_and_constraints()
    app.state.graphiti = graphiti
    await ingest.async_worker.start()
    yield
    # Shutdown
    await ingest.async_worker.stop()
    await graphiti.close()


app = FastAPI(lifespan=lifespan)


app.include_router(retrieve.router)
app.include_router(ingest.router)


@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(content={'status': 'healthy'}, status_code=200)
