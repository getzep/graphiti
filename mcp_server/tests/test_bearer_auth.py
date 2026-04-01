"""Tests for BearerTokenMiddleware."""

from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient
from starlette.types import Receive, Scope, Send

from graphiti_mcp_server import BearerTokenMiddleware


async def _dummy_app(scope: Scope, receive: Receive, send: Send) -> None:
    """Minimal ASGI app that returns 200 OK."""
    response = PlainTextResponse('OK')
    await response(scope, receive, send)


def _make_client(api_key: str) -> TestClient:
    app = BearerTokenMiddleware(_dummy_app, api_key)
    return TestClient(app, raise_server_exceptions=False)


class TestBearerTokenMiddleware:
    def test_valid_token(self):
        client = _make_client('test-secret')
        resp = client.get('/', headers={'Authorization': 'Bearer test-secret'})
        assert resp.status_code == 200
        assert resp.text == 'OK'

    def test_missing_auth_header(self):
        client = _make_client('test-secret')
        resp = client.get('/')
        assert resp.status_code == 401

    def test_wrong_token(self):
        client = _make_client('test-secret')
        resp = client.get('/', headers={'Authorization': 'Bearer wrong-token'})
        assert resp.status_code == 401

    def test_non_bearer_scheme(self):
        client = _make_client('test-secret')
        resp = client.get('/', headers={'Authorization': 'Basic dXNlcjpwYXNz'})
        assert resp.status_code == 401

    def test_empty_bearer_value(self):
        client = _make_client('test-secret')
        resp = client.get('/', headers={'Authorization': 'Bearer '})
        assert resp.status_code == 401

    def test_health_endpoint_exempt(self):
        client = _make_client('test-secret')
        resp = client.get('/health')
        assert resp.status_code == 200

    def test_health_exempt_no_auth_header(self):
        client = _make_client('test-secret')
        resp = client.get('/health')
        assert resp.status_code == 200
        assert resp.text == 'OK'

    def test_other_paths_require_auth(self):
        client = _make_client('test-secret')
        for path in ['/mcp', '/sse', '/messages', '/api']:
            resp = client.get(path)
            assert resp.status_code == 401, f'{path} should require auth'
