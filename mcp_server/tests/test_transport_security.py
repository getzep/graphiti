"""Unit tests for MCP_HOSTNAMES transport-security allowlist.

Covers the _build_transport_security helper and its behavior when driven
through the real MCP SDK TransportSecurityMiddleware (DNS rebinding protection).
"""

from mcp.server.transport_security import (
    TransportSecurityMiddleware,
)
from starlette.requests import Request

from graphiti_mcp_server import _build_transport_security

# Each host is allowlisted in both forms: the bare host (for the SDK's exact-match
# path, which is what a portless Host header hits) and 'host:*' (for explicit ports).
LOCALHOST_TRIO = ['127.0.0.1', '127.0.0.1:*', 'localhost', 'localhost:*', '[::1]', '[::1]:*']


def _both(host: str) -> list[str]:
    return [host, f'{host}:*']


def _request_with_host(host: str) -> Request:
    """Build a minimal Starlette HTTP request carrying a Host header."""
    scope = {
        'type': 'http',
        'method': 'POST',
        'path': '/mcp/',
        'headers': [(b'host', host.encode())],
    }
    return Request(scope)


def test_unset_returns_none():
    assert _build_transport_security('') is None


def test_single_hostname():
    settings = _build_transport_security('mem.example.com')
    assert settings is not None
    assert settings.enable_dns_rebinding_protection is True
    assert settings.allowed_hosts == _both('mem.example.com') + LOCALHOST_TRIO


def test_csv_trims_and_drops_empties():
    settings = _build_transport_security('a.local, b.example.com ,,')
    assert settings is not None
    assert settings.allowed_hosts == _both('a.local') + _both('b.example.com') + LOCALHOST_TRIO


async def test_middleware_allows_configured_hostname():
    middleware = TransportSecurityMiddleware(_build_transport_security('mem.example.com'))
    result = await middleware.validate_request(_request_with_host('mem.example.com:8000'))
    assert result is None


async def test_middleware_allows_localhost():
    middleware = TransportSecurityMiddleware(_build_transport_security('mem.example.com'))
    result = await middleware.validate_request(_request_with_host('127.0.0.1:8000'))
    assert result is None


async def test_middleware_rejects_unknown_host():
    middleware = TransportSecurityMiddleware(_build_transport_security('mem.example.com'))
    result = await middleware.validate_request(_request_with_host('evil.com:8000'))
    assert result is not None
    assert result.status_code == 421


async def test_middleware_allows_configured_hostname_without_port():
    """A client reaching the server on port 80/443 sends a portless Host header.

    The MCP SDK's wildcard matcher requires a port (`host.startswith(base + ':')`),
    so a `host:*` pattern alone rejects the bare hostname — the exact case
    MCP_HOSTNAMES exists to enable. The allowlist must carry both forms.
    """
    middleware = TransportSecurityMiddleware(_build_transport_security('mem.example.com'))
    result = await middleware.validate_request(_request_with_host('mem.example.com'))
    assert result is None


async def test_middleware_allows_localhost_without_port():
    middleware = TransportSecurityMiddleware(_build_transport_security('mem.example.com'))
    for host in ('127.0.0.1', 'localhost', '[::1]'):
        result = await middleware.validate_request(_request_with_host(host))
        assert result is None, f'{host} (portless) should be allowed'


async def test_middleware_still_rejects_unknown_host_without_port():
    middleware = TransportSecurityMiddleware(_build_transport_security('mem.example.com'))
    result = await middleware.validate_request(_request_with_host('evil.com'))
    assert result is not None
    assert result.status_code == 421


async def test_middleware_none_settings_disables_protection():
    # None settings -> FastMCP would apply its own localhost-only defaults, but the
    # middleware's own None path disables protection (any host allowed).
    middleware = TransportSecurityMiddleware(_build_transport_security(''))
    result = await middleware.validate_request(_request_with_host('anything.example:8000'))
    assert result is None
