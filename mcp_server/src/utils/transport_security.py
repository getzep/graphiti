import os

from mcp.server.transport_security import TransportSecuritySettings


LOCAL_ALLOWED_HOSTS = ['localhost:*', '127.0.0.1:*']


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def _split_csv_env(name: str) -> list[str]:
    raw = os.getenv(name, '')
    return [item.strip() for item in raw.split(',') if item.strip()]


def _normalize_host_pattern(host: str) -> str:
    host = host.strip()
    if not host:
        return host
    if host.endswith(':*'):
        return host
    if ':' in host:
        return host
    return f'{host}:*'


def _origin_patterns_from_hosts(hosts: list[str]) -> list[str]:
    origins: list[str] = []
    for host in hosts:
        if host.endswith(':*'):
            base = host[:-2]
            origins.extend([f'http://{base}:*', f'https://{base}:*'])
        else:
            origins.extend([f'http://{host}', f'https://{host}'])
    return origins


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def build_transport_security_settings(server_host: str) -> TransportSecuritySettings:
    enable_dns_rebinding_protection = _env_flag(
        'MCP_ENABLE_DNS_REBINDING_PROTECTION',
        True,
    )

    allowed_hosts = list(LOCAL_ALLOWED_HOSTS)
    if server_host not in {'0.0.0.0', '127.0.0.1', 'localhost'}:
        allowed_hosts.append(_normalize_host_pattern(server_host))

    allowed_hosts.extend(_normalize_host_pattern(host) for host in _split_csv_env('MCP_ALLOWED_HOSTS'))
    allowed_hosts = _dedupe(allowed_hosts)

    env_origins = _split_csv_env('MCP_ALLOWED_ORIGINS')
    allowed_origins = _dedupe(env_origins or _origin_patterns_from_hosts(allowed_hosts))

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=enable_dns_rebinding_protection,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )
