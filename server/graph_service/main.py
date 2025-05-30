from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from server.graph_service.config import get_settings
from server.graph_service.routers import ingest, retrieve, browser_agent
from server.graph_service.zep_graphiti import initialize_graphiti


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Skip Neo4j initialization for testing
    # settings = get_settings()
    # await initialize_graphiti(settings)
    yield
    # Shutdown
    # No need to close Graphiti here, as it's handled per-request


app = FastAPI(lifespan=lifespan)




# Add CORS middleware to allow requests from the browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your extension's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(retrieve.router)
app.include_router(ingest.router)
app.include_router(browser_agent.router)


@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(content={'status': 'healthy'}, status_code=200)
