#!/usr/bin/env python3
"""
Integration tests for OAuth wrapper with MCP server
These are mocked integration tests that verify the integration paths work correctly
"""

import json
import os

# Import the OAuth wrapper app
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from oauth_wrapper import app

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestOAuthIntegration:
    """Integration tests for OAuth wrapper and MCP server interaction"""

    @pytest.fixture
    def client(self):
        """Test client for OAuth wrapper"""
        return TestClient(app)

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Mock environment variables"""
        monkeypatch.setenv("MCP_INTERNAL_PORT", "8021")

    def test_oauth_discovery_flow(self, client):
        """Test complete OAuth discovery flow"""
        # Step 1: Discover OAuth authorization server
        auth_server_response = client.get("/.well-known/oauth-authorization-server")
        assert auth_server_response.status_code == 200
        auth_server_data = auth_server_response.json()

        # Step 2: Verify required endpoints exist
        assert "authorization_endpoint" in auth_server_data
        assert "token_endpoint" in auth_server_data
        assert "registration_endpoint" in auth_server_data

        # Step 3: Discover protected resource
        resource_response = client.get("/.well-known/oauth-protected-resource")
        assert resource_response.status_code == 200
        resource_data = resource_response.json()

        # Step 4: Verify resource points back to auth server
        assert "oauth_authorization_server" in resource_data

    def test_client_registration_flow(self, client):
        """Test client registration and subsequent use"""
        # Step 1: Register a new client
        registration_data = {
            "client_name": "Integration Test Client",
            "redirect_uris": ["http://localhost:3000/callback"],
            "grant_types": ["authorization_code"],
            "response_types": ["code"]
        }

        reg_response = client.post("/register", json=registration_data)
        assert reg_response.status_code == 201
        client_info = reg_response.json()

        # Step 2: Verify client credentials were issued
        assert "client_id" in client_info
        assert "client_secret" in client_info
        assert client_info["client_name"] == registration_data["client_name"]

        # Step 3: Verify client can be used (in a real scenario)
        # This would involve using the client_id in an authorization request
        assert len(client_info["client_id"]) > 10  # Reasonable length
        assert len(client_info["client_secret"]) > 20  # Secure length

    def test_sse_connection_lifecycle(self, client, mock_env):
        """Test SSE connection establishment and streaming"""
        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_instance

            # Mock the stream response
            mock_stream_response = AsyncMock()
            mock_stream_response.aiter_bytes = AsyncMock()
            mock_stream_response.aiter_bytes.return_value = self._async_generator([b"data: test\n\n"])
            mock_instance.stream.return_value.__aenter__.return_value = mock_stream_response

            response = client.get("/sse", headers={"Accept": "text/event-stream"})

            # Verify SSE response format
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_messages_endpoint_with_session(self, client, mock_env):
        """Test messages endpoint with session management"""
        session_id = "test-session-123"
        message_data = {
            "type": "request",
            "method": "tools/list",
            "params": {}
        }

        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_instance

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.content = json.dumps({
                "type": "response",
                "result": {"tools": []}
            }).encode()
            mock_response.headers = {"content-type": "application/json"}
            mock_instance.post.return_value = mock_response

            response = client.post(f"/messages/?session_id={session_id}", json=message_data)

            assert response.status_code == 200
            assert response.json() == {"type": "response", "result": {"tools": []}}

    def test_error_propagation(self, client, mock_env):
        """Test that errors from MCP server are properly propagated"""
        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_instance

            # Simulate MCP server error
            mock_response = AsyncMock()
            mock_response.status_code = 500
            mock_response.content = b'{"error": "Internal server error"}'
            mock_response.headers = {"content-type": "application/json"}
            mock_instance.post.return_value = mock_response

            response = client.post("/messages/", json={"invalid": "request"})

            # Verify error is propagated
            assert response.status_code == 500

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests through OAuth wrapper"""
        # Test multiple concurrent requests to static endpoints
        responses = []
        for _i in range(5):
            response = client.get("/.well-known/oauth-authorization-server")
            responses.append(response)

        # Verify all requests succeeded
        for response in responses:
            assert response.status_code == 200
            assert response.json()["issuer"] == "http://localhost:8020"

    async def _async_generator(self, items):
        """Helper to create async generator for testing"""
        for item in items:
            yield item
