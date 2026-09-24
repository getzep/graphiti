"""Standalone ASGI app for the Zep-compatible layer.

Deliberately separate from graph_service.main: that app's Settings require
OPENAI_API_KEY and it builds its own Graphiti instance. Keeping this entrypoint
independent means upstream Graphiti changes to main.py never conflict.

Serve with:
    uvicorn graph_service.zep_compat.app:app --host 0.0.0.0 --port 8088
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .router import router
from .runtime import Runtime

logging.basicConfig(
    level=os.environ.get('ZEP_COMPAT_LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger('zep_compat')

# MiroFish pins the SDK base_url to <host>/api/v2, so every route lives there.
API_PREFIX = os.environ.get('ZEP_COMPAT_API_PREFIX', '/api/v2')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Graphiti's telemetry ships to PostHog on init. On an offline box that is a
    # hang risk, not just a privacy leak, so default it off unless overridden.
    os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')
    logger.info('starting Zep compatibility layer on prefix %s', API_PREFIX)
    runtime = await Runtime.create()
    app.state.zep_runtime = runtime
    logger.info('ready')
    try:
        yield
    finally:
        await runtime.close()
        logger.info('shut down')


app = FastAPI(
    title='Zep-compatible Graph API (Graphiti-backed)',
    description='Local drop-in for the Zep Cloud v2 endpoints MiroFish calls.',
    lifespan=lifespan,
)

app.include_router(router, prefix=API_PREFIX)


@app.get('/healthcheck')
async def healthcheck():
    ready = getattr(app.state, 'zep_runtime', None) is not None
    return JSONResponse(
        content={'status': 'healthy' if ready else 'starting'},
        status_code=200 if ready else 503,
    )
