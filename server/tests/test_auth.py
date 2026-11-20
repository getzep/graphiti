import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from graph_service.auth import require_api_key
from graph_service.config import Settings, get_settings
from graph_service.routers import ingest, retrieve
from graph_service.zep_graphiti import get_graphiti


class _FakeGraphiti:
    async def retrieve_episodes(self, group_ids, last_n, reference_time):
        return []


def _settings(api_key: str | None) -> Settings:
    return Settings(openai_api_key='test-openai-key', api_key=api_key)


def _client(api_key: str | None) -> TestClient:
    settings = _settings(api_key)
    app = FastAPI()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_graphiti] = lambda: _FakeGraphiti()
    app.include_router(retrieve.router)
    app.include_router(ingest.router)
    return TestClient(app)


@pytest.mark.asyncio
async def test_missing_credentials_rejected():
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(_settings('secret'), None)
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_wrong_key_rejected():
    credentials = HTTPAuthorizationCredentials(scheme='Bearer', credentials='wrong')
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(_settings('secret'), credentials)
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_matching_key_accepted():
    credentials = HTTPAuthorizationCredentials(scheme='Bearer', credentials='secret')
    await require_api_key(_settings('secret'), credentials)


@pytest.mark.asyncio
async def test_fail_closed_when_no_key_configured():
    credentials = HTTPAuthorizationCredentials(scheme='Bearer', credentials='anything')
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(_settings(None), credentials)
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


def test_clear_requires_authentication():
    response = _client('secret').post('/clear')
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_cross_group_delete_requires_authentication():
    response = _client('secret').delete('/group/some-other-tenant')
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_authorized_request_succeeds():
    response = _client('secret').get(
        '/episodes/tenant-a',
        params={'last_n': 1},
        headers={'Authorization': 'Bearer secret'},
    )
    assert response.status_code == status.HTTP_200_OK


def test_authorized_request_fails_closed_without_configured_key():
    response = _client(None).get(
        '/episodes/tenant-a',
        params={'last_n': 1},
        headers={'Authorization': 'Bearer secret'},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
