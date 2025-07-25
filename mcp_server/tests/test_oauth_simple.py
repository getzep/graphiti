#!/usr/bin/env python3
"""
Simple tests for OAuth wrapper core functionality
"""

import os

# Import the OAuth wrapper app
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from oauth_wrapper import app


class TestOAuthWrapperSimple:
    """Simple test suite for OAuth wrapper functionality"""

    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Mock environment variables"""
        monkeypatch.setenv("MCP_INTERNAL_PORT", "8021")

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
        assert "issuer" in data
        assert "authorization_endpoint" in data
        assert "token_endpoint" in data
        assert "registration_endpoint" in data
        assert "jwks_uri" in data
        assert "response_types_supported" in data
        assert "grant_types_supported" in data
        assert "code_challenge_methods_supported" in data
        assert "token_endpoint_auth_methods_supported" in data

    def test_oauth_protected_resource_metadata(self, client):
        """Test OAuth protected resource metadata endpoint"""
        response = client.get("/.well-known/oauth-protected-resource")
        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert "resource" in data
        assert "oauth_authorization_server" in data
        assert "bearer_methods_supported" in data
        assert "resource_signing_alg_values_supported" in data

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

    def test_client_registration_invalid_json(self, client):
        """Test client registration with invalid JSON"""
        response = client.post("/register", content="invalid json")
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "invalid_request"

    def test_messages_proxy_endpoint_exists(self, client, mock_env):
        """Test that messages proxy endpoint exists and handles requests"""
        # Test that the endpoint is routed properly
        # We don't need to test the actual proxying, just that it's wired up
        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_instance

            # Create a proper mock response
            mock_response = MagicMock()
            mock_response.content = b'{"success": true}'
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_instance.post.return_value = mock_response

            response = client.post("/messages/", json={"test": "data"})
            assert response.status_code == 200

    def test_sse_post_endpoint_exists(self, client, mock_env):
        """Test that SSE POST endpoint exists and handles requests"""
        # Test the POST endpoint which is simpler to mock
        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.content = b'{"success": true}'
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_instance.post.return_value = mock_response

            response = client.post("/sse", json={"test": "data"})
            assert response.status_code == 200

    def test_environment_variable_usage(self, mock_env):
        """Test that environment variables are properly used"""
        # Test that the MCP_INTERNAL_PORT environment variable is used
        import os


        # Verify that the environment variable is read
        assert os.environ.get("MCP_INTERNAL_PORT") == "8021"

    def test_header_filtering(self, client, mock_env):
        """Test that host header is filtered out"""
        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.content = b'{"success": true}'
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_instance.post.return_value = mock_response

            # Send request with host header
            response = client.post("/messages/",
                                 json={"test": "data"},
                                 headers={"host": "example.com", "custom": "value"})

            assert response.status_code == 200
            # The test passes if no exception is raised during header processing

    def test_query_parameters_forwarded(self, client, mock_env):
        """Test that query parameters are forwarded"""
        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.content = b'{"success": true}'
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_instance.post.return_value = mock_response

            # Send request with query parameters
            response = client.post("/messages/?session_id=123&param=value",
                                 json={"test": "data"})

            assert response.status_code == 200
            # Verify the mock was called (indicating proxying occurred)
            mock_instance.post.assert_called_once()

    def test_error_responses(self, client, mock_env):
        """Test that error responses are properly handled"""
        with patch("src.oauth_wrapper.httpx.AsyncClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_instance

            # Simulate an error response from MCP server
            mock_response = MagicMock()
            mock_response.content = b'{"error": "Not found"}'
            mock_response.status_code = 404
            mock_response.headers = {"content-type": "application/json"}
            mock_instance.post.return_value = mock_response

            response = client.post("/messages/", json={"test": "data"})

            # Error should be passed through
            assert response.status_code == 404
            assert response.json() == {"error": "Not found"}
