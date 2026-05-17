# Validation — MCP 2026 spec compliance

## Automated tests

- `mcp_server/tests/auth/test_bearer_middleware.py` — accepts a valid JWT with matching `aud`/`resource`; rejects missing `Authorization`; rejects wrong audience; rejects expired; rejects bad signature.
- `mcp_server/tests/auth/test_jwks_cache.py` — TTL behavior, key rotation, fail-closed on JWKS endpoint unavailability.
- `mcp_server/tests/discovery/test_server_card.py` — `/.well-known/mcp-server-card.json` returns the full tool list; JSON schema matches the MCP 2026 spec; every tool has a non-empty description.
- `mcp_server/tests/transport/test_stateless_http.py` — two FastMCP instances behind an in-test reverse proxy serve the same client without sticky session; `Mcp-Session-Id` round-trips correctly.
- `mcp_server/tests/transport/test_sse_deprecated.py` — configuring `transport: sse` emits a `DeprecationWarning` and the server still starts in 0.30 (will fail in 0.31).
- `mcp_server/tests/audit/test_audit_log_redaction.py` — every `@mcp.tool` invocation emits exactly one JSON line; `episode_body` field is absent from `add_memory` log lines; `group_id` and `sub` present.
- `mcp_server/tests/test_allowed_hosts.py` — `MCP_ALLOWED_HOSTS=foo.local,bar.local` accepts those Host headers, rejects others; default keeps current strict allowlist behavior.

## Smoke checks

```bash
# Server Card discovery
curl -s http://localhost:8000/.well-known/mcp-server-card.json | jq '.tools | length'
# Should match `len(mcp.list_tools())`

# Bearer-token auth
curl -s -H 'Authorization: Bearer <valid jwt>' http://localhost:8000/mcp/  # → 200
curl -s http://localhost:8000/mcp/                                          # → 401

# Stateless across replicas
docker compose -f mcp_server/docker/docker-compose-ha.yml up
# Run a Claude Desktop session; interleaving requests across both replicas should succeed
```

## Manual criteria

- Audit log line is greppable, contains no episode content, and is one line per tool call.
- README's "Production deployment" section explains the OAuth env vars cleanly enough that a new operator can configure it without reading the source.

## Risks & rollback

- **Failure modes**: stateless mode breaks long-running tools that silently relied on per-process state; JWKS fetch failures hard-fail tool calls; some MCP clients don't yet send `Mcp-Session-Id`; the MCP 2026 spec finalizes differently than today's draft; SSO providers vary in claim shape and our defaults don't fit one.
- **Rollback**: feature flags (`MCP_OAUTH_ENABLED`, `MCP_STATELESS_HTTP`) default off in 0.30 to allow operators to opt in. Revert PR removes the new packages cleanly; the existing single-instance stateful path is untouched.

## Open questions

- Bearer alternative: do we want SPIFFE/mTLS as a Phase 2? Defer until a user asks.
- Server Card URL: well-known path vs config-discoverable? The 2026 spec final wording is pending; track and adapt before tagging 0.30.
- Audit log schema: stable internal format vs OpenTelemetry log event? Decide whether to align with the existing OTel tracing integration in `graphiti_core/tracer.py`.
