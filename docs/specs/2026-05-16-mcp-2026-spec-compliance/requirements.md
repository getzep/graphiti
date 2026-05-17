# Requirements — MCP 2026 spec compliance

## Outcome

`mcp_server/` aligns with the MCP 2026 spec evolution — stateless Streamable HTTP across instances, OAuth 2.1 Resource Server semantics with RFC 8707 Resource Indicators, a published MCP Server Card for discovery, configurable DNS-rebind allowlist, and structured per-tool-call audit logs — so enterprise operators can put it behind a load balancer + gateway without bespoke shims.

## Users affected

Self-hosted Graphiti operators deploying the MCP server behind a load balancer, behind SSO, or in front of multiple AI clients (Claude Desktop, Cursor, AI Studio); SREs reading audit logs; security teams running compliance reviews. Open issues #1441, #1470, #1488 all map here.

## In scope

- Switch `FastMCP` to **stateless** Streamable HTTP so multiple replicas can serve the same client behind a round-robin LB without sticky sessions.
- OAuth 2.1 Resource Server: bearer token validation (JWT-only default; opt-in token introspection), JWKS fetch + cache, `aud` / `resource` enforcement per RFC 8707, scope claim → tool allowlist mapping.
- Static MCP Server Card at `GET /.well-known/mcp-server-card.json`, generated at startup from the live `@mcp.tool` registry.
- Configurable DNS-rebind allowlist via `MCP_ALLOWED_HOSTS` (resolves #1470).
- Bearer-token auth (resolves #1488).
- Structured audit-log line per tool call: tool name, group_id, subject from token, request id, duration, status; `episode_body` redacted.
- New config block under `mcp_server/config/config.yaml`:
  ```yaml
  auth:
    issuer: ${MCP_OAUTH_ISSUER}
    jwks_url: ${MCP_OAUTH_JWKS_URL}
    audience: ${MCP_OAUTH_AUDIENCE}
    introspection_url: ${MCP_OAUTH_INTROSPECTION_URL:}
  ```

## Out of scope

- Building an identity provider. Assume an upstream OAuth issuer (Auth0, Okta, Cognito, Keycloak, etc.).
- Multi-tenancy isolation beyond `group_id` — that is Zep's concern.
- mTLS / SPIFFE — defer until a user asks.
- Touching the REST `server/` package; this PR is MCP-only.

## Decisions

- Stateless mode is the default in 0.30 with a feature flag `MCP_STATELESS_HTTP=false` to revert. Reason: the 2026 roadmap is explicit; horizontal scale is the bottleneck most enterprises hit first.
- JWT-only validation is the default; token introspection is opt-in (`MCP_OAUTH_INTROSPECTION_URL`). Reason: lower-latency default, opt-in for stricter posture.
- MCP Server Card is generated from the live tool registry, not hand-maintained. Reason: drift-proof.
- SSE transport is dropped from supported list in 0.30 with a one-cycle deprecation warning. Reason: upstream MCP deprecated SSE; carrying it forward is a maintenance tax.
- Audit log writes via existing logger as a single JSON line per call. Reason: composes with any log shipper without changing infra.
