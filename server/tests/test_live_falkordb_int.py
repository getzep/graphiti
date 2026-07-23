"""Live end-to-end regression test for the graph_service REST API.

Spawns the FastAPI server (``graph_service.main:app``) as a uvicorn subprocess
backed by FalkorDB (the default test database) and a real OpenAI model, then
exercises the public API end to end:

    POST /messages (async ingest)
        -> poll GET /episodes/{group_id} until the episode is persisted
        -> POST /search and assert at least one fact was extracted
        -> DELETE /group/{group_id} to clean up.

This is the server counterpart to ``mcp_server``'s live test. It is skipped
automatically unless an OpenAI API key is available AND FalkorDB is reachable:

- Local: the project-root ``.env`` is loaded, so an ``OPENAI_API_KEY`` there is
  picked up. Start FalkorDB first, e.g.::

      docker run -d --name falkordb -p 6379:6379 falkordb/falkordb:latest
      cd server && uv run pytest tests/test_live_falkordb_int.py

- CI: ``OPENAI_API_KEY`` comes from the GitHub environment and FalkorDB runs as a
  container (see .github/workflows/server-tests.yml).

The model is taken from ``MODEL_NAME`` (CI pins a lighter, broadly-available
model for speed/reliability), falling back to the server's configured default.
"""

import contextlib
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

# Load the project-root .env for local runs (CI provides the env directly).
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / '.env')

SERVER_DIR = Path(__file__).resolve().parent.parent
FALKORDB_HOST = os.environ.get('FALKORDB_HOST', 'localhost')
FALKORDB_PORT = int(os.environ.get('FALKORDB_PORT', '6379'))

pytestmark = [pytest.mark.integration]


def _falkordb_reachable(host: str, port: int) -> bool:
    """Best-effort TCP check that a FalkorDB (Redis) endpoint is reachable."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


# Skip the whole module unless prerequisites are present, so the suite is a no-op
# locally without setup and on fork PRs (which have no secrets).
if not os.environ.get('OPENAI_API_KEY'):
    pytest.skip('OPENAI_API_KEY not set; skipping live server tests', allow_module_level=True)
if not _falkordb_reachable(FALKORDB_HOST, FALKORDB_PORT):
    pytest.skip(
        f'FalkorDB not reachable at {FALKORDB_HOST}:{FALKORDB_PORT}; skipping live server tests',
        allow_module_level=True,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _unique_token() -> str:
    # Alphanumeric only: safe as both a FalkorDB graph name and a group_id. A uuid
    # (not a timestamp+pid) guarantees a distinct graph per fixture even for tests
    # set up within the same second or run under pytest-xdist.
    return f'srvtest{uuid.uuid4().hex}'


def _wait_for_health(proc: subprocess.Popen, base_url: str, log, timeout: float = 90.0) -> None:
    """Block until the server answers /healthcheck (lifespan startup complete)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.seek(0)
            raise RuntimeError(
                f'server exited early (code {proc.returncode}):\n{log.read().decode(errors="replace")}'
            )
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(f'{base_url}/healthcheck', timeout=2).status_code == 200:
                return
        time.sleep(1)
    log.seek(0)
    raise RuntimeError(
        f'server did not become healthy within {timeout}s:\n{log.read().decode(errors="replace")}'
    )


@pytest.fixture
def live_server() -> Iterator[tuple[str, str]]:
    """Spawn the graph_service on a fresh FalkorDB graph; yield (base_url, group_id)."""
    port = _free_port()
    token = _unique_token()
    env = {
        **os.environ,
        'DB_BACKEND': 'falkordb',
        'FALKORDB_HOST': FALKORDB_HOST,
        'FALKORDB_PORT': str(FALKORDB_PORT),
        # A fresh logical graph per run isolates the test and keeps a long-lived
        # local FalkorDB from accumulating data across runs.
        'FALKORDB_DATABASE': token,
    }
    # Not a context manager: the handle must outlive setup (the subprocess writes
    # to it across the yield) and is closed explicitly in the finally block.
    log = tempfile.TemporaryFile()  # noqa: SIM115
    proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'uvicorn',
            'graph_service.main:app',
            '--host',
            '127.0.0.1',
            '--port',
            str(port),
        ],
        cwd=str(SERVER_DIR),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    base_url = f'http://127.0.0.1:{port}'
    try:
        _wait_for_health(proc, base_url, log)
        yield base_url, token
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()
        log.close()


def _wait_for_episodes(
    client: httpx.Client,
    group_id: str,
    expected: int = 1,
    timeout: float = 180.0,
    poll: float = 3.0,
) -> list:
    """Poll GET /episodes until ``expected`` episodes are persisted.

    /episodes is only called after /healthcheck has passed, so the server is
    fully up: a 5xx here is a real error (e.g. a FalkorDB query failure) and is
    surfaced immediately rather than masked as a slow generic timeout. Any other
    non-200 is remembered and reported if the wait ultimately times out.
    """
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        resp = client.get(f'/episodes/{group_id}', params={'last_n': 50})
        if resp.status_code == 200:
            episodes = resp.json()
            if isinstance(episodes, list) and len(episodes) >= expected:
                return episodes
        elif resp.status_code >= 500:
            raise AssertionError(f'/episodes returned {resp.status_code}: {resp.text[:500]}')
        else:
            last_error = f'{resp.status_code} {resp.text[:500]}'
        time.sleep(poll)
    detail = f'; last non-200 from /episodes: {last_error}' if last_error else ''
    raise AssertionError(f'episode was not processed within {timeout:.0f}s{detail}')


def _search_until(
    client: httpx.Client, group_id: str, query: str, attempts: int = 6, poll: float = 3.0
) -> list:
    """POST /search, retrying until facts appear (index/processing lag)."""
    facts: list = []
    for _ in range(attempts):
        resp = client.post(
            '/search', json={'group_ids': [group_id], 'query': query, 'max_facts': 10}
        )
        assert resp.status_code == 200, f'/search failed: {resp.status_code} {resp.text}'
        facts = resp.json().get('facts', [])
        if facts:
            return facts
        time.sleep(poll)
    return facts


def test_healthcheck(live_server: tuple[str, str]) -> None:
    base_url, _ = live_server
    with httpx.Client(base_url=base_url, timeout=10) as client:
        resp = client.get('/healthcheck')
        assert resp.status_code == 200
        assert resp.json().get('status') == 'healthy'


def test_ingest_search_delete_e2e(live_server: tuple[str, str]) -> None:
    base_url, group_id = live_server
    with httpx.Client(base_url=base_url, timeout=30) as client:
        try:
            # 1. Ingest a message with clearly-extractable entities and a relationship.
            add = client.post(
                '/messages',
                json={
                    'group_id': group_id,
                    'messages': [
                        {
                            'content': (
                                'Alice is a software engineer at Acme Corporation. '
                                'She works on the Graphiti project.'
                            ),
                            'role_type': 'user',
                            'role': 'alice',
                            'name': 'intro',
                        }
                    ],
                },
            )
            assert add.status_code == 202, f'add /messages failed: {add.status_code} {add.text}'

            # 2. Wait for the async worker to persist the episode (raises with
            # detail on a server error or timeout).
            episodes = _wait_for_episodes(client, group_id)
            assert any(e.get('group_id') == group_id for e in episodes)

            # 3. At least one fact (edge) was extracted and is searchable.
            facts = _search_until(client, group_id, 'Alice Acme Corporation Graphiti')
            assert facts, 'expected at least one extracted fact'
        finally:
            # Always clean up this run's data, even if an assertion above failed.
            with contextlib.suppress(httpx.HTTPError):
                client.delete(f'/group/{group_id}')
