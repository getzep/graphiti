import os

from mcp.server.transport_security import TransportSecuritySettings

from utils.transport_security import build_transport_security_settings


def test_build_transport_security_settings_defaults_local_hosts(monkeypatch):
    monkeypatch.delenv('MCP_ALLOWED_HOSTS', raising=False)
    monkeypatch.delenv('MCP_ALLOWED_ORIGINS', raising=False)
    monkeypatch.delenv('MCP_ENABLE_DNS_REBINDING_PROTECTION', raising=False)

    settings = build_transport_security_settings(server_host='0.0.0.0')

    assert isinstance(settings, TransportSecuritySettings)
    assert settings.enable_dns_rebinding_protection is True
    assert 'localhost:*' in settings.allowed_hosts
    assert '127.0.0.1:*' in settings.allowed_hosts
    assert 'http://localhost:*' in settings.allowed_origins
    assert 'https://localhost:*' in settings.allowed_origins


def test_build_transport_security_settings_accepts_extra_hosts_from_env(monkeypatch):
    monkeypatch.setenv(
        'MCP_ALLOWED_HOSTS',
        'localhost:*,127.0.0.1:*,192.168.123.104:*,mcp.example.internal:*',
    )
    monkeypatch.delenv('MCP_ALLOWED_ORIGINS', raising=False)
    monkeypatch.delenv('MCP_ENABLE_DNS_REBINDING_PROTECTION', raising=False)

    settings = build_transport_security_settings(server_host='0.0.0.0')

    assert '192.168.123.104:*' in settings.allowed_hosts
    assert 'mcp.example.internal:*' in settings.allowed_hosts
    assert 'http://192.168.123.104:*' in settings.allowed_origins
    assert 'https://mcp.example.internal:*' in settings.allowed_origins


def test_build_transport_security_settings_can_be_disabled(monkeypatch):
    monkeypatch.setenv('MCP_ENABLE_DNS_REBINDING_PROTECTION', 'false')
    monkeypatch.delenv('MCP_ALLOWED_HOSTS', raising=False)
    monkeypatch.delenv('MCP_ALLOWED_ORIGINS', raising=False)

    settings = build_transport_security_settings(server_host='0.0.0.0')

    assert settings.enable_dns_rebinding_protection is False
