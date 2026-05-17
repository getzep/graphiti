# Plan — MCP 2026 spec compliance

## Approach

Three workstreams sharing a single config surface. **Statelessness**: configure `FastMCP(stateless_http=True)` and remove `sse` from supported transports; per-request state is reconstituted from `Mcp-Session-Id` headers. **Authorization**: a new `mcp_server/src/auth/` package implements bearer validation, JWKS fetch + cache, and RFC 8707 audience/resource enforcement; wired in via FastMCP middleware. **Discovery**: a startup hook builds a `ServerCard` model from the `@mcp.tool` registry and serves it at the well-known URL. Feature flags ship default-off in 0.30 so existing operators can opt in.

## Steps

1. Add `mcp_server/src/auth/` with `jwks.py` (httpx-backed JWKS client with TTL cache), `oauth_resource.py` (RFC 8707 aud/resource validation + scope mapping), `bearer_middleware.py` (FastMCP middleware that enforces validation per request).
2. Wire the middleware into the FastMCP app construction in `mcp_server/src/graphiti_mcp_server.py`, gated by `MCP_OAUTH_ENABLED`.
3. Add `mcp_server/src/discovery/server_card.py` that introspects the live `@mcp.tool` registry at startup and serves `/.well-known/mcp-server-card.json`. Generate the JSON schema per the MCP 2026 spec; assert at startup that every tool has a description.
4. Set `FastMCP(..., stateless_http=True)` when `MCP_STATELESS_HTTP=true` (default in 0.30). Remove `sse` from the supported transport list; emit a deprecation warning when configured.
5. Replace the hardcoded DNS-rebind allowlist with `MCP_ALLOWED_HOSTS` (comma-separated). Default keeps current strict behavior. Resolves #1470.
6. Bearer-token auth path (#1488): the auth middleware short-circuits requests without `Authorization`. Update `mcp_server/docker/docker-compose.yml` with sample env vars referencing a public test issuer for dev (e.g., a Keycloak container in the compose file).
7. Audit log: wrap every `@mcp.tool` invocation with a decorator that emits one JSON line via the existing logger — fields `{ts, request_id, tool, group_id, sub, duration_ms, status}`. Redact `episode_body` from `add_memory` calls. Add a `tests/test_audit_log_redaction.py`.
8. Update `mcp_server/config/config.yaml` schema and `mcp_server/README.md` with the new `auth:` block, the stateless flag, and the audit-log fields.
9. Add tests under `mcp_server/tests/auth/`, `mcp_server/tests/discovery/`, `mcp_server/tests/transport/`, `mcp_server/tests/audit/`.

## Dependencies / order

Step 1 before 2 and 7 (audit decorator wants the auth subject). Step 3 independent. Steps 4 and 5 independent. Step 8 and 9 last.
