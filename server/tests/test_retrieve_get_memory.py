"""Unit tests for the graph_service retrieval endpoint.

``/get-memory`` requires ``center_node_uuid`` and documents it as "the uuid of the
node to center the retrieval on", so the value must reach ``Graphiti.search``; the
node-distance reranker is the only consumer of it.
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Allow both `cd server && pytest tests/...` and running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_service.routers import retrieve  # noqa: E402
from graph_service.zep_graphiti import get_graphiti  # noqa: E402

CENTER_NODE_UUID = 'd4f3a9d0-8d3b-4a51-9b9c-6d0b5a1a3f11'


class RecordingGraphiti:
    """Records the kwargs the endpoint passes to ``Graphiti.search``."""

    def __init__(self) -> None:
        self.search_calls: list[dict] = []

    async def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return []


@pytest.fixture
def recording_graphiti() -> RecordingGraphiti:
    return RecordingGraphiti()


@pytest.fixture
def client(recording_graphiti: RecordingGraphiti) -> TestClient:
    app = FastAPI()
    app.include_router(retrieve.router)
    app.dependency_overrides[get_graphiti] = lambda: recording_graphiti
    return TestClient(app)


def _memory_request(**overrides) -> dict:
    body = {
        'group_id': 'group_a',
        'max_facts': 5,
        'center_node_uuid': CENTER_NODE_UUID,
        'messages': [{'content': 'Where does Ana work?', 'role_type': 'user', 'role': 'Ana'}],
    }
    body.update(overrides)
    return body


def test_get_memory_forwards_center_node_uuid(client, recording_graphiti):
    response = client.post('/get-memory', json=_memory_request())

    assert response.status_code == 200
    assert len(recording_graphiti.search_calls) == 1
    search_kwargs = recording_graphiti.search_calls[0]
    assert search_kwargs['center_node_uuid'] == CENTER_NODE_UUID
    assert search_kwargs['group_ids'] == ['group_a']
    assert search_kwargs['num_results'] == 5
    assert 'Where does Ana work?' in search_kwargs['query']


def test_get_memory_accepts_null_center_node_uuid(client, recording_graphiti):
    response = client.post('/get-memory', json=_memory_request(center_node_uuid=None))

    assert response.status_code == 200
    assert recording_graphiti.search_calls[0]['center_node_uuid'] is None
