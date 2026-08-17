import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from graph_service.config import get_settings
from graph_service.connections.manager import ConnectionManager, load_or_create_encryption_key
from graph_service.connections.store import ConnectionStore
from graph_service.routers import connections, ingest, retrieve, sources
from graph_service.sources.store import SourceStore
from graph_service.sources.sync import SyncManager
from graph_service.zep_graphiti import initialize_graphiti

logger = logging.getLogger(__name__)
STATIC_ROOT = Path(__file__).parent / 'static'


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = SourceStore(settings.source_state_path)
    connection_store = ConnectionStore(
        settings.source_state_path,
        load_or_create_encryption_key(settings),
    )
    connection_manager = ConnectionManager(settings, connection_store)
    manager = SyncManager(settings, store, connection_manager=connection_manager)
    app.state.source_store = store
    app.state.connection_store = connection_store
    app.state.connection_manager = connection_manager
    app.state.sync_manager = manager
    app.state.database_ready = False
    app.state.database_error = None
    try:
        await initialize_graphiti(settings)
        app.state.database_ready = True
    except Exception as exc:
        app.state.database_error = str(exc)[:500]
        logger.warning(
            'Graph database initialization failed; dashboard is in degraded mode: %s', exc
        )
    yield
    await manager.shutdown()


app = FastAPI(
    title='Graphiti Studio',
    description='Incremental knowledge graph ingestion from local files, Feishu, and MeeGo.',
    lifespan=lifespan,
)


app.include_router(retrieve.router)
app.include_router(ingest.router)
app.include_router(connections.router)
app.include_router(sources.router)
app.mount('/static', StaticFiles(directory=STATIC_ROOT), name='static')


@app.get('/', include_in_schema=False)
async def dashboard():
    return FileResponse(STATIC_ROOT / 'index.html')


@app.get('/healthcheck')
async def healthcheck():
    status_value = 'healthy' if app.state.database_ready else 'degraded'
    return JSONResponse(
        content={
            'status': status_value,
            'database_ready': app.state.database_ready,
            'database_error': app.state.database_error,
        },
        status_code=200,
    )
