"""What the service does when OpenAI refuses the request.

The failure that motivated these: an exhausted OpenAI quota turned `POST /search` into a bare
500 `Internal Server Error` and killed the ingestion worker silently, so `POST /messages` kept
answering 202 into a queue nobody drained. Both looked like bugs in the graph query.

Need no OpenAI key and no database: the Graphiti client is a stand-in that raises.
"""

import asyncio

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from graph_service.config import get_settings
from graph_service.main import app
from graph_service.routers.ingest import AsyncWorker
from graph_service.zep_graphiti import get_graphiti

API_KEY = 'test-api-key-6Yp2Qk'
AUTH = {'Authorization': f'Bearer {API_KEY}'}


def _openai_error(cls: type[openai.APIStatusError], status: int, message: str):
    """Build a real openai error, since the handler dispatches on the exception type."""
    request = httpx.Request('POST', 'https://api.openai.com/v1/embeddings')
    response = httpx.Response(status, request=request)
    return cls(message, response=response, body={'error': {'message': message}})


QUOTA_ERROR = _openai_error(
    openai.RateLimitError,
    429,
    'You exceeded your current quota, please check your plan and billing details.',
)
BAD_KEY_ERROR = _openai_error(
    openai.AuthenticationError, 401, 'Incorrect API key provided: sk-xxx.'
)


class RaisingGraphiti:
    """Stands in for ZepGraphiti, failing the way an unusable OPENAI_API_KEY makes it fail."""

    def __init__(self, error: Exception):
        self.error = error

    async def search(self, **_kwargs):
        raise self.error


@pytest.fixture
def client(monkeypatch):
    """A TestClient over the real app, with auth satisfied and no lifespan.

    No lifespan, so no Graphiti client is built and no index build runs; the dependency override
    below is what the routers get instead.
    """
    monkeypatch.setenv('GRAPHITI_API_KEY', API_KEY)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-not-used')
    get_settings.cache_clear()

    def _client(error: Exception):
        app.dependency_overrides[get_graphiti] = lambda: RaisingGraphiti(error)
        return TestClient(app)

    yield _client
    app.dependency_overrides.clear()
    get_settings.cache_clear()


SEARCH_BODY = {'group_ids': ['demo'], 'query': 'who leads the payments team?', 'max_facts': 10}


def test_exhausted_quota_is_not_an_internal_server_error(client):
    """A refused OpenAI call is upstream's fault, and the response should say which upstream."""
    response = client(QUOTA_ERROR).post('/search', json=SEARCH_BODY, headers=AUTH)

    assert response.status_code == 429
    detail = response.json()['detail']
    assert 'OpenAI' in detail
    assert 'quota' in detail.lower()


def test_rejected_openai_key_is_reported_as_a_bad_gateway(client):
    """The caller's key was fine; OPENAI_API_KEY is the one the operator has to go fix."""
    response = client(BAD_KEY_ERROR).post('/search', json=SEARCH_BODY, headers=AUTH)

    assert response.status_code == 502
    assert 'OPENAI_API_KEY' in response.json()['detail']


async def _drain(worker: AsyncWorker):
    """Run the worker until the queue is empty, then let the job it last took finish.

    Polls rather than joins: the worker never calls queue.task_done(), so queue.join() would
    block forever.
    """
    await worker.start()
    for _ in range(200):
        if worker.queue.empty():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    await worker.stop()


@pytest.mark.asyncio
async def test_worker_survives_a_failing_job():
    """A job that raises must not take the worker down with it.

    The worker outlives every request, so a job that kills it leaves /messages answering 202
    for the rest of the process's life while nothing is ingested.
    """
    worker = AsyncWorker()
    ran: list[str] = []

    async def failing():
        ran.append('failing')
        raise QUOTA_ERROR

    async def succeeding():
        ran.append('succeeding')

    await worker.queue.put(failing)
    await worker.queue.put(succeeding)
    await _drain(worker)

    assert ran == ['failing', 'succeeding'], 'the worker stopped after the failing job'


@pytest.mark.asyncio
async def test_failing_job_is_logged(caplog):
    """A dropped episode has to leave a trace, or ingestion fails where nobody can see it."""
    worker = AsyncWorker()

    async def failing():
        raise QUOTA_ERROR

    await worker.queue.put(failing)
    with caplog.at_level('ERROR'):
        await _drain(worker)

    assert any(r.exc_info for r in caplog.records), 'no traceback was logged'
