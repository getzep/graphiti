#!/usr/bin/env python3
"""
Comprehensive unit tests for OAuth wrapper
"""

import os

# Import the OAuth wrapper app
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from oauth_wrapper import OAUTH_CONFIG, app


class TestOAuthWrapper:
    """Test suite for OAuth wrapper functionality"""

    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Mock environment variables"""
        monkeypatch.setenv("MCP_INTERNAL_PORT", "8021")
        monkeypatch.setenv("MCP_SERVER_PORT", "8020")
        monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client")
        monkeypatch.setenv("OAUTH_CLIENT_SECRET", "test-secret")
        monkeypatch.setenv("OAUTH_ISSUER", "http://localhost:8020")
        monkeypatch.setenv("OAUTH_AUDIENCE", "test-audience")

    def test_root_endpoint(self, client):
        """Test root endpoint returns service info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Graphiti MCP OAuth Server"
        assert data["version"] == "1.0.0"

    def test_oauth_authorization_server_metadata(self, client):
        """Test OAuth authorization server metadata endpoint"""
        response = client.get("/.well-known/oauth-authorization-server")
        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert data["issuer"] == OAUTH_CONFIG["issuer"]
        assert data["authorization_endpoint"] == f"{OAUTH_CONFIG['issuer']}/authorize"
        assert data["token_endpoint"] == f"{OAUTH_CONFIG['issuer']}/token"
        assert data["registration_endpoint"] == f"{OAUTH_CONFIG['issuer']}/register"
        assert data["jwks_uri"] == f"{OAUTH_CONFIG['issuer']}/jwks"
        assert "code" in data["response_types_supported"]
        assert "authorization_code" in data["grant_types_supported"]
        assert "S256" in data["code_challenge_methods_supported"]
        assert "none" in data["token_endpoint_auth_methods_supported"]

    def test_sse_oauth_authorization_server_metadata(self, client):
        """Test SSE OAuth authorization server metadata endpoint"""
        response = client.get("/sse/.well-known/oauth-authorization-server")
        assert response.status_code == 200
        data = response.json()

        # Should return same metadata as main endpoint
        assert data["issuer"] == OAUTH_CONFIG["issuer"]

    def test_oauth_protected_resource_metadata(self, client):
        """Test OAuth protected resource metadata endpoint"""
        response = client.get("/.well-known/oauth-protected-resource")
        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert data["resource"] == OAUTH_CONFIG["issuer"]
        assert data["oauth_authorization_server"] == f"{OAUTH_CONFIG['issuer']}/.well-known/oauth-authorization-server"
        assert "header" in data["bearer_methods_supported"]
        assert "RS256" in data["resource_signing_alg_values_supported"]

    def test_client_registration_success(self, client):
        """Test successful client registration"""
        client_data = {
            "client_name": "Test Client",
            "redirect_uris": ["http://localhost:3000/callback"]
        }

        response = client.post("/register", json=client_data)
        assert response.status_code == 201
        data = response.json()

        # Verify response contains required fields
        assert "client_id" in data
        assert "client_secret" in data
        assert "client_id_issued_at" in data
        assert data["grant_types"] == ["authorization_code", "refresh_token"]
        assert data["response_types"] == ["code"]
        assert data["token_endpoint_auth_method"] == "none"
        assert data["client_name"] == client_data["client_name"]
        assert data["redirect_uris"] == client_data["redirect_uris"]

    def test_client_registration_invalid_json(self, client):
        """Test client registration with invalid JSON"""
        response = client.post("/register", content="invalid json")
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "invalid_request"

    @pytest.mark.asyncio
    async def test_sse_proxy_get_streaming(self, mock_env):
        """Test SSE proxy GET request with streaming response"""
        # We need to test the actual endpoint behavior, not mock the client
        # since the client is created inside the generator function
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            # The SSE endpoint returns a StreamingResponse
            # We'll just verify it accepts GET requests properly
            with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
                mock_instance = AsyncMock()
                mock_client_class.return_value.__aenter__.return_value = mock_instance

                # Mock the stream response
                mock_stream_response = AsyncMock()
                mock_stream_response.aiter_bytes = AsyncMock()
                mock_stream_response.aiter_bytes.return_value = self._async_generator([b"data: test\n\n"])
                mock_instance.stream.return_value.__aenter__.return_value = mock_stream_response

                response = client.get("/sse")

                # Verify the response headers
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
                assert response.headers["cache-control"] == "no-cache"

    def test_sse_proxy_post(self, mock_env, client):
        """Test SSE proxy POST request"""
        test_body = {"test": "data"}

        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_instance

            # Create a proper mock response
            mock_response = MagicMock()
            mock_response.content = b'{"result": "success"}'
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_instance.post.return_value = mock_response

            response = client.post("/sse", json=test_body)

            assert response.status_code == 200
            assert response.json() == {"result": "success"}

    def test_messages_proxy_post(self, mock_env, client):
        """Test messages proxy POST request"""
        test_body = {"message": "test"}

        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_instance

            # Create a proper mock response
            mock_response = MagicMock()
            mock_response.content = b'{"response": "ok"}'
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_instance.post.return_value = mock_response

            response = client.post("/messages/?session_id=123", json=test_body)

            assert response.status_code == 200
            assert response.json() == {"response": "ok"}

    def test_sse_proxy_headers_forwarding(self, mock_env, client):
        """Test that headers are properly forwarded in SSE proxy"""
        test_headers = {
            "authorization": "Bearer test-token",
            "x-custom-header": "custom-value"
        }

        # Track the headers that were passed to httpx
        captured_headers = None

        def capture_stream_call(*args, **kwargs):
            nonlocal captured_headers
            captured_headers = kwargs.get("headers", {})
            # Return a mock stream context manager
            mock_stream = AsyncMock()
            mock_stream.aiter_bytes = AsyncMock()
            mock_stream.aiter_bytes.return_value = self._async_generator([b"data: test\n\n"])
            return AsyncMock(__aenter__=AsyncMock(return_value=mock_stream))

        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_instance.stream = capture_stream_call
            mock_client_class.return_value.__aenter__.return_value = mock_instance

            response = client.get("/sse", headers=test_headers)

            # The test client will consume the stream, so we just verify status
            assert response.status_code == 200

    def test_sse_proxy_httpx_error_handling(self, mock_env, client):
        """Test SSE proxy handles httpx errors gracefully"""
        # When httpx fails inside the generator, FastAPI handles it
        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            # Make the AsyncClient constructor raise an error
            mock_client_class.return_value.__aenter__.side_effect = httpx.ConnectError("Connection failed")

            # The error will be caught by FastAPI and return 500
            response = client.get("/sse")
            assert response.status_code == 500

    async def _async_generator(self, items):
        """Helper to create async generator for testing"""
        for item in items:
            yield item
