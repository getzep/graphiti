# Per-User Graph Isolation via HTTP Headers - Technical Analysis

**Date**: November 8, 2025
**Status**: INVESTIGATION PHASE - Not Yet Implemented
**Priority**: Medium - Enhancement for Multi-User LibreChat Deployments

---

## Executive Summary

This document analyzes the feasibility of implementing per-user graph isolation in the Graphiti MCP server using HTTP headers from LibreChat, allowing each user to have their own isolated knowledge graph without modifying tool signatures.

**Verdict**: **FEASIBLE BUT COMPLEX** - Technology supports this approach, but several critical issues must be addressed before implementation.

---

## Table of Contents

1. [Background](#background)
2. [Proposed Solution](#proposed-solution)
3. [Critical Analysis](#critical-analysis)
4. [LibreChat Capabilities](#librechat-capabilities)
5. [FastMCP Middleware Support](#fastmcp-middleware-support)
6. [Implementation Approaches](#implementation-approaches)
7. [Known Issues & Risks](#known-issues--risks)
8. [Requirements for Implementation](#requirements-for-implementation)
9. [Alternative Approaches](#alternative-approaches)
10. [Next Steps](#next-steps)

---

## Background

### Current State

The Graphiti MCP server currently uses a single `group_id` for all users, meaning:
- All users share the same knowledge graph
- No data isolation between users
- Configured via `config.graphiti.group_id` or CLI argument

### Desired State

Enable per-user graph isolation where:
- Each LibreChat user has their own knowledge graph
- Isolation happens automatically via HTTP headers
- LLMs don't need to know about multi-user architecture
- Tools work identically for single and multi-user deployments

### Use Case

Multi-user LibreChat deployment where:
- User A's preferences/conversations → Graph A
- User B's preferences/conversations → Graph B
- No data leakage between users

---

## Proposed Solution

### High-Level Architecture

```
LibreChat (per-user session)
    ↓
    Headers: { X-User-ID: "user_12345" }
    ↓
FastMCP Middleware
    ↓
    Extracts user_id from headers
    ↓
    Stores in request context
    ↓
MCP Tools (add_memory, search_nodes, etc.)
    ↓
    Uses: group_id = explicit_param OR context_user_id OR config_default
    ↓
Graphiti Core (with user-specific group_id)
```

### Technical Approach

**Option A: Direct Header Access in Tools**
```python
@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    ...
):
    from fastmcp.server.dependencies import get_http_headers

    headers = get_http_headers()
    user_id = headers.get("x-user-id")

    effective_group_id = group_id or user_id or config.graphiti.group_id
    # ... rest of implementation
```

**Option B: Middleware + Context State** (Recommended)
```python
class UserContextMiddleware(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        headers = get_http_headers()
        user_id = headers.get("x-user-id")

        if context.fastmcp_context and user_id:
            context.fastmcp_context.set_state("user_id", user_id)
            logger.info(f"Request from user_id: {user_id}")

        return await call_next(context)

mcp.add_middleware(UserContextMiddleware())

# In tools:
@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    ctx: Context | None = None,
    ...
):
    user_id = ctx.get_state("user_id") if ctx else None
    effective_group_id = group_id or user_id or config.graphiti.group_id
```

---

## Critical Analysis

A comprehensive architectural review identified several critical concerns:

### ✅ Valid Points

1. **LibreChat Officially Supports This Pattern**
   - Headers with user context are a documented, core feature
   - `{{LIBRECHAT_USER_ID}}` placeholder designed for this use case
   - Per-user connection management built into LibreChat

2. **FastMCP Middleware Exists**
   - Added in FastMCP v2.9.0
   - Supports request interception and context injection
   - `get_http_headers()` dependency function available

3. **Context State Management Available**
   - Request-scoped state via `ctx.set_state()` / `ctx.get_state()`
   - Async-safe context handling

### ⚠️ Critical Concerns

#### 1. **MCP Protocol Transport Coupling**

**Issue**: Using HTTP headers creates transport dependency
- MCP is designed to be transport-agnostic (stdio, sse, http)
- Header-based isolation only works with HTTP transports
- **Impact**: stdio/sse transports won't have per-user isolation

**Mitigation**:
- Document HTTP transport requirement
- Detect transport type at runtime
- Provide graceful fallback for non-HTTP transports

#### 2. **Queue Service Context Loss**

**Issue**: Background task spawning may lose context
```python
# From queue_service.py:45
asyncio.create_task(self._process_episode_queue(group_id))
```

Context state is request-scoped only:
> "Context is scoped to a single request; state set in one request will not be available in subsequent requests"

**Risk**: Episodes processed in background queues may use wrong/missing group_id

**Solution**: Pass user_id explicitly to queue service
```python
await queue_service.add_episode(
    group_id=effective_group_id,
    user_id=user_id,  # Pass explicitly for background processing
    ...
)
```

#### 3. **Neo4j Driver Thread Pool Context Loss**

**Issue**: Neo4j async driver uses thread pools internally
- Python ContextVars may not propagate across thread boundaries
- Could cause context loss during database operations

**Testing Required**: Verify context preservation with concurrent Neo4j operations

#### 4. **Stale Context Bug in StreamableHTTP** ⚠️ BLOCKER

**Known FastMCP Issue**:
> "When using StreamableHTTP transport with multiple requests in the same session, MCP tool execution consistently receives stale HTTP request context from the first request"

**Impact**: User A's request might receive User B's context
**Severity**: CRITICAL - Data isolation violation

**Mitigation**:
- Test thoroughly with concurrent requests
- Add request ID logging to detect stale context
- Monitor FastMCP issue tracker for fix
- Consider request ID validation in middleware

#### 5. **Security & Fallback Behavior**

**Missing Header Scenario**:
```
Request with no X-User-ID
    ↓
user_id = None
    ↓
Falls back to config.graphiti.group_id = "main"
    ↓
SECURITY ISSUE: User writes to shared graph
```

**Header Injection Attack**:
```
Attacker sends: X-User-ID: admin
    ↓
Gains access to admin's graph
    ↓
PRIVILEGE ESCALATION
```

**Required Mitigations**:
- Validate header presence in multi-user mode
- Reject requests missing X-User-ID (401/403)
- Validate user_id format (alphanumeric, max length)
- Consider verifying user_id against auth token
- Add defense-in-depth even if LibreChat validates

#### 6. **Debugging & Observability**

**Problem**: Implicit state makes debugging difficult

**Required Logging**:
```python
logger.info(
    f"Tool: add_memory | episode: {name} | "
    f"group_id={effective_group_id} | "
    f"source=(explicit={group_id}, context={user_id}, default={config.graphiti.group_id}) | "
    f"request_id={ctx.request_id if ctx else 'N/A'}"
)
```

**Metrics Needed**:
- `tool_calls_by_user_id`
- `context_fallback_count` (when header missing)
- `stale_context_detected` (request ID mismatches)

#### 7. **LLM Override Behavior**

**Design Question**: What happens if LLM explicitly passes `group_id`?

```python
# LLM calls:
add_memory(name="...", episode_body="...", group_id="other_user")
```

**Options**:
1. **Explicit param wins** (flexible but risky)
   ```python
   effective_group_id = group_id or user_id or config_default
   ```

2. **Header always wins** (strict isolation)
   ```python
   effective_group_id = user_id or group_id or config_default
   ```

3. **Reject mismatch** (paranoid)
   ```python
   if group_id and user_id and group_id != user_id:
       raise PermissionError("Cannot access other user's graph")
   ```

**Decision Required**: Choose based on security requirements

---

## LibreChat Capabilities

### Headers Configuration

LibreChat supports dynamic header substitution in `librechat.yaml`:

```yaml
mcpServers:
  graphiti-memory:
    url: "http://graphiti-mcp:8000/mcp/"
    headers:
      X-User-ID: "{{LIBRECHAT_USER_ID}}"
      X-User-Email: "{{LIBRECHAT_USER_EMAIL}}"
```

### Available User Placeholders

- `{{LIBRECHAT_USER_ID}}` - Unique user identifier
- `{{LIBRECHAT_USER_NAME}}` - User display name
- `{{LIBRECHAT_USER_EMAIL}}` - User email address
- `{{LIBRECHAT_USER_ROLE}}` - User role
- `{{LIBRECHAT_USER_PROVIDER}}` - Auth provider
- Social auth IDs (Google, GitHub, etc.)

### Multi-User Features

LibreChat provides:
- **Per-user connection management**: Separate MCP connections per user
- **User idle management**: Disconnects after 15 minutes inactivity
- **Connection lifecycle**: Proper setup/teardown per user session
- **Custom user variables**: Per-user credentials storage

### Transport Requirements

Headers only work with:
- `sse` (Server-Sent Events)
- `streamable-http` (HTTP with streaming)

Not supported:
- `stdio` (standard input/output)

---

## FastMCP Middleware Support

### Middleware System (v2.9.0+)

FastMCP provides a pipeline-based middleware system:

```python
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers

class UserContextMiddleware(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        # Extract headers
        headers = get_http_headers()
        user_id = headers.get("x-user-id")

        # Validate and store
        if user_id:
            if not self._validate_user_id(user_id):
                raise ValueError(f"Invalid user_id format: {user_id}")

            if context.fastmcp_context:
                context.fastmcp_context.set_state("user_id", user_id)
                logger.info(f"User context set: {user_id}")
        else:
            logger.warning("Missing X-User-ID header")

        return await call_next(context)

    def _validate_user_id(self, user_id: str) -> bool:
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]{1,64}$', user_id))

# Add to server
mcp.add_middleware(UserContextMiddleware())
```

### Context Access in Tools

**Via Dependency Injection**:
```python
@mcp.tool()
async def my_tool(ctx: Context) -> str:
    user_id = ctx.get_state("user_id")
    return f"Processing for user: {user_id}"
```

**Direct Header Access**:
```python
from fastmcp.server.dependencies import get_http_headers

@mcp.tool()
async def my_tool() -> str:
    headers = get_http_headers()
    user_id = headers.get("x-user-id")
    return f"User: {user_id}"
```

### Middleware Execution Order

```python
mcp.add_middleware(AuthMiddleware())      # Runs first
mcp.add_middleware(UserContextMiddleware()) # Runs second
mcp.add_middleware(LoggingMiddleware())    # Runs third
```

Order matters: First added runs first on request, last on response.

### Context Scope & Limitations

**Request-Scoped Only**:
- Each MCP request gets new context
- State doesn't persist between requests
- Background tasks may lose context

**Transport Compatibility**:
> "Middleware inspecting HTTP headers won't work with stdio transport"

**Known Breaking Changes**:
> "MCP middleware is a brand new concept and may be subject to breaking changes in future versions"

---

## Implementation Approaches

### Approach 1: Direct Header Access (Simple)

**Implementation**:
```python
@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    ...
) -> SuccessResponse | ErrorResponse:
    from fastmcp.server.dependencies import get_http_headers

    headers = get_http_headers()
    user_id = headers.get("x-user-id")

    effective_group_id = group_id or user_id or config.graphiti.group_id

    logger.info(f"add_memory: group_id={effective_group_id} (explicit={group_id}, header={user_id})")

    # ... rest of implementation
```

**Pros**:
- Simple, no middleware needed
- Direct, explicit header access
- Easy to debug

**Cons**:
- Code duplication across 8-10 tools
- No centralized logging
- Harder to add validation

**Tools to Modify**: ~8-10 tools
- `add_memory`
- `search_nodes`
- `get_entities_by_type`
- `search_memory_facts`
- `compare_facts_over_time`
- `delete_entity_edge`
- `delete_episode`
- `get_entity_edge`
- `get_episodes`
- `clear_graph`

### Approach 2: Middleware + Context State (Recommended)

**Implementation**:

**Step 1: Add Middleware**
```python
# mcp_server/src/middleware/user_context.py

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
import logging
import re

logger = logging.getLogger(__name__)

class UserContextMiddleware(Middleware):
    """Extract user_id from X-User-ID header and store in context."""

    def __init__(self, require_user_id: bool = False):
        self.require_user_id = require_user_id

    async def on_request(self, context: MiddlewareContext, call_next):
        headers = get_http_headers()
        user_id = headers.get("x-user-id")

        if user_id:
            # Validate format
            if not self._validate_user_id(user_id):
                logger.error(f"Invalid user_id format: {user_id}")
                raise ValueError(f"Invalid X-User-ID format")

            # Store in context
            if context.fastmcp_context:
                context.fastmcp_context.set_state("user_id", user_id)
                logger.debug(f"User context established: {user_id}")

        elif self.require_user_id:
            logger.error("Missing required X-User-ID header")
            raise ValueError("X-User-ID header is required for multi-user mode")

        else:
            logger.warning("X-User-ID header not provided, using default group_id")

        # Log request with user context
        method = context.method
        logger.info(f"Request: {method} | user_id={user_id or 'default'}")

        result = await call_next(context)
        return result

    @staticmethod
    def _validate_user_id(user_id: str) -> bool:
        """Validate user_id format: alphanumeric, dash, underscore, 1-64 chars."""
        return bool(re.match(r'^[a-zA-Z0-9_-]{1,64}$', user_id))
```

**Step 2: Register Middleware**
```python
# mcp_server/src/graphiti_mcp_server.py

from middleware.user_context import UserContextMiddleware

# After mcp = FastMCP(...) initialization
# Set require_user_id=True for strict multi-user mode
mcp.add_middleware(UserContextMiddleware(require_user_id=False))
```

**Step 3: Modify Tools**
```python
@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    ctx: Context | None = None,
    ...
) -> SuccessResponse | ErrorResponse:
    # Extract user_id from context
    user_id = ctx.get_state("user_id") if ctx else None

    # Priority: explicit param > context user_id > config default
    effective_group_id = group_id or user_id or config.graphiti.group_id

    # Detailed logging
    logger.info(
        f"add_memory: episode={name} | "
        f"group_id={effective_group_id} | "
        f"source=(explicit={group_id}, context={user_id}, default={config.graphiti.group_id})"
    )

    # ... rest of implementation
```

**Pros**:
- Centralized user extraction and validation
- DRY (Don't Repeat Yourself)
- Consistent logging across all tools
- Easy to add security checks
- Single point for header validation

**Cons**:
- Requires adding `ctx: Context | None` to all tool signatures
- More complex initial setup
- Context state is request-scoped only

### Approach 3: Hybrid Approach

Combine both approaches:
- Use middleware for logging, validation, metrics
- Use direct header access in tools (simpler signatures)

**Implementation**:
```python
class UserContextMiddleware(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        headers = get_http_headers()
        user_id = headers.get("x-user-id")

        # Validate and log, but don't store
        if user_id:
            if not self._validate_user_id(user_id):
                raise ValueError("Invalid user_id format")
            logger.info(f"Request from user: {user_id}")

        return await call_next(context)

# In tools: still access headers directly
@mcp.tool()
async def add_memory(...):
    headers = get_http_headers()
    user_id = headers.get("x-user-id")
    effective_group_id = group_id or user_id or config.graphiti.group_id
```

**Pros**:
- No need to modify tool signatures
- Centralized validation and logging
- Simpler tool implementation

**Cons**:
- Still some duplication in tools
- Two places reading the same header

---

## Known Issues & Risks

### 🚨 Critical Blockers

1. **Stale Context in StreamableHTTP** - FastMCP bug
   - **Severity**: CRITICAL
   - **Impact**: User A receives User B's context
   - **Status**: Known issue in FastMCP
   - **Mitigation**: Thorough testing, request ID logging
   - **Link**: GitHub issue tracking required

2. **Queue Service Context Loss**
   - **Severity**: HIGH
   - **Impact**: Background episodes use wrong group_id
   - **Status**: Architectural limitation
   - **Mitigation**: Pass user_id explicitly to queue
   - **Testing**: Integration tests with concurrent episodes

3. **Neo4j Thread Pool Context**
   - **Severity**: MEDIUM-HIGH
   - **Impact**: Database operations may lose user context
   - **Status**: Needs verification
   - **Mitigation**: Test with concurrent database operations
   - **Testing**: Load testing with multiple users

### ⚠️ Major Concerns

4. **Security - Missing Header Fallback**
   - **Risk**: Users write to shared graph when header missing
   - **Mitigation**: Require header in multi-user mode
   - **Config**: Add `REQUIRE_USER_ID` environment variable

5. **Security - Header Injection**
   - **Risk**: Attacker spoofs user_id to access other graphs
   - **Mitigation**: Validate header format, trust LibreChat validation
   - **Enhancement**: Add header signature verification

6. **Debugging Complexity**
   - **Risk**: Difficult to troubleshoot which group_id was used
   - **Mitigation**: Comprehensive structured logging
   - **Metrics**: Expose per-user metrics

7. **LLM Override Ambiguity**
   - **Risk**: Unclear behavior when LLM passes explicit group_id
   - **Mitigation**: Choose and document priority strategy
   - **Testing**: Test explicit param vs header precedence

### ⚙️ Minor Considerations

8. **Transport Limitation**
   - **Impact**: Only works with HTTP transports
   - **Mitigation**: Document requirement clearly
   - **Detection**: Add runtime transport detection

9. **FastMCP API Stability**
   - **Risk**: Breaking changes in middleware API
   - **Mitigation**: Pin FastMCP version, monitor changelog
   - **Version**: Test with FastMCP 2.9.0+

10. **Configuration Drift**
    - **Risk**: Logs show default group_id but actual varies
    - **Mitigation**: Log effective group_id per request
    - **UX**: Update startup logging for multi-user mode

---

## Requirements for Implementation

### Mandatory Testing

- [ ] **Verify stale context bug** with StreamableHTTP
  - Create integration test with concurrent requests
  - Validate each request receives correct user_id
  - Log request IDs to detect context reuse

- [ ] **Test queue service context propagation**
  - Add episodes concurrently from different users
  - Verify each episode uses correct group_id
  - Check background task context inheritance

- [ ] **Test Neo4j thread pool behavior**
  - Concurrent database operations from multiple users
  - Verify context doesn't bleed across threads
  - Load test with realistic concurrency

- [ ] **Test missing header scenarios**
  - Request without X-User-ID header
  - Verify fallback behavior (reject vs default)
  - Test error messages and logging

- [ ] **Test explicit group_id override**
  - LLM passes group_id parameter
  - Verify precedence (param vs header)
  - Test security implications

### Mandatory Implementation

- [ ] **Add UserContextMiddleware**
  - Extract X-User-ID header
  - Validate format (alphanumeric, 1-64 chars)
  - Store in context state or use directly
  - Add structured logging

- [ ] **Modify all tools** (8-10 tools)
  - Add context parameter or direct header access
  - Implement group_id priority logic
  - Add detailed logging per tool

- [ ] **Update queue service**
  - Pass user_id explicitly for background tasks
  - Verify context doesn't get lost
  - Add queue-specific logging

- [ ] **Add comprehensive logging**
  - Log effective group_id for every operation
  - Include source (explicit/context/default)
  - Add request ID correlation
  - Structured logging format

- [ ] **Add security validations**
  - User ID format validation
  - Header presence check (if required)
  - Rate limiting per user_id
  - Audit logging for security events

- [ ] **Update configuration**
  - Add `REQUIRE_USER_ID` environment variable
  - Add `MULTI_USER_MODE` flag
  - Document HTTP transport requirement
  - Update example configs

### Recommended Enhancements

- [ ] **Add observability**
  - Prometheus metrics: `tool_calls_by_user{user_id="X"}`
  - Context fallback counter
  - Stale context detection counter
  - Request duration per user

- [ ] **Add admin capabilities**
  - Admin override for debugging
  - View all users' graphs
  - Cross-user search (admin only)

- [ ] **Add documentation**
  - Update MCP server README
  - Update LibreChat setup guide
  - Add troubleshooting section
  - Document security model

- [ ] **Add migration path**
  - Script to split shared graph by user
  - Backup/restore per user
  - User data export

### Testing Checklist

**Unit Tests**:
- [ ] UserContextMiddleware header extraction
- [ ] User ID validation logic
- [ ] Group ID priority logic
- [ ] Error handling for missing headers

**Integration Tests**:
- [ ] Concurrent requests from different users
- [ ] Queue service with multiple users
- [ ] Database operations with user context
- [ ] All tools with user isolation

**Security Tests**:
- [ ] Header injection attempts
- [ ] Missing header handling
- [ ] Explicit group_id override attempts
- [ ] Rate limiting per user

**Performance Tests**:
- [ ] 10+ concurrent users
- [ ] 100+ episodes queued
- [ ] Context overhead measurement
- [ ] Database connection pooling

---

## Alternative Approaches

### Alternative 1: Explicit group_id Required

Make `group_id` a required parameter in all tools:

```python
@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str,  # REQUIRED - no default
    ...
):
    """LibreChat must provide group_id in every call."""
    # No fallback logic needed
```

**Pros**:
- Explicit, no hidden state
- Works with all transports (stdio, sse, http)
- Clear ownership in code
- Easy to debug and audit

**Cons**:
- Requires LibreChat plugin/modification
- More verbose tool calls
- LLM must know about multi-user architecture

**Implementation**: Requires LibreChat to inject group_id into tool parameters

### Alternative 2: LibreChat Proxy Layer

Create a thin proxy between LibreChat and Graphiti:

```
LibreChat → User-Aware Proxy → Graphiti MCP
            (injects group_id)
```

**Pros**:
- Keeps Graphiti MCP clean and transport-agnostic
- Separation of concerns
- Easy to swap LibreChat for other clients
- No changes to Graphiti MCP server

**Cons**:
- Additional component to maintain
- Extra network hop (minimal overhead)
- More complex deployment

**Implementation**: Python/Node.js proxy that intercepts requests

### Alternative 3: Per-User MCP Instances

Run separate Graphiti MCP server instances per user/tenant:

```
LibreChat routes to:
  - http://localhost:8000/mcp/ (User A)
  - http://localhost:8001/mcp/ (User B)
  - http://localhost:8002/mcp/ (User C)
```

**Pros**:
- Complete isolation (process boundaries)
- Simplest architecture
- No context management complexity
- Easy to scale horizontally

**Cons**:
- Resource intensive (N servers)
- Complex orchestration (start/stop/route)
- Overkill for most use cases
- Connection overhead

**Implementation**: Kubernetes/Docker Compose with dynamic routing

### Alternative 4: Database-Level Isolation

Use Neo4j multi-database feature (Enterprise only):

```python
# Each user gets their own database
graphiti_client_user_a = Graphiti(uri="bolt://neo4j:7687", database="user_a")
graphiti_client_user_b = Graphiti(uri="bolt://neo4j:7687", database="user_b")
```

**Pros**:
- Strong isolation at database level
- Better resource utilization than separate instances
- Leverages Neo4j native features

**Cons**:
- Requires Neo4j Enterprise
- Database management complexity
- Not supported with FalkorDB

**Implementation**: Dynamic database selection per request

---

## Next Steps

### Phase 1: Investigation (Current)

- [x] Document LibreChat capabilities
- [x] Verify FastMCP middleware support
- [x] Identify critical issues and risks
- [x] Document implementation approaches
- [ ] **Test for stale context bug** ← NEXT STEP
- [ ] Create proof-of-concept implementation
- [ ] Test context propagation in queue service

### Phase 2: Proof of Concept

1. **Implement minimal middleware**
   - Extract user_id from header
   - Log to verify correct user per request
   - Test with 2-3 concurrent users

2. **Test critical scenarios**
   - Concurrent requests (detect stale context)
   - Queue service background tasks
   - Neo4j thread pool behavior
   - Missing header handling

3. **Measure performance impact**
   - Context overhead
   - Logging overhead
   - Additional parameter cost

### Phase 3: Full Implementation (If POC Successful)

1. **Implement full middleware**
   - Validation, logging, metrics
   - Security checks
   - Error handling

2. **Modify all tools**
   - Add context parameter
   - Implement group_id priority
   - Add comprehensive logging

3. **Update queue service**
   - Explicit user_id passing
   - Context preservation

4. **Add tests**
   - Unit, integration, security, performance
   - Automated CI/CD tests

5. **Update documentation**
   - README, setup guide, troubleshooting
   - Security model documentation

### Phase 4: Production Deployment

1. **Staged rollout**
   - Deploy to test environment
   - Limited user beta testing
   - Monitor for issues

2. **Monitoring & metrics**
   - Set up dashboards
   - Configure alerts
   - Track user isolation

3. **Security audit**
   - Penetration testing
   - Header injection testing
   - Audit logging review

---

## Decision Log

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| 2025-11-08 | Document findings without implementation | Critical issues need investigation | ✅ Complete |
| TBD | Choose implementation approach | Pending POC testing | 🔄 Pending |
| TBD | Define group_id priority strategy | Pending security requirements | 🔄 Pending |
| TBD | Decide on REQUIRE_USER_ID default | Pending deployment model | 🔄 Pending |

---

## References

### LibreChat Documentation
- [MCP Servers Configuration](https://www.librechat.ai/docs/configuration/librechat_yaml/object_structure/mcp_servers)
- User placeholders: `{{LIBRECHAT_USER_ID}}`
- Headers support for SSE and streamable-http

### FastMCP Documentation
- [Middleware Guide](https://gofastmcp.com/servers/middleware)
- [Context & Dependencies](https://gofastmcp.com/servers/context)
- `get_http_headers()` function
- Middleware added in v2.9.0

### Known Issues
- FastMCP #1233: Stale context in StreamableHTTP
- FastMCP #817: Access headers in middleware
- FastMCP #1291: HTTP request header access

### Related Files
- `/mcp_server/src/graphiti_mcp_server.py` - Main MCP server
- `/mcp_server/src/services/queue_service.py` - Background processing
- `/DOCS/Librechat.setup.md` - LibreChat setup guide

---

## Contact & Support

For questions about this analysis or implementation:
- Create GitHub issue in fork repository
- Reference this document in discussions
- Tag issues with `enhancement`, `multi-user`, `security`

---

**Document Version**: 1.0
**Last Updated**: 2025-11-08
**Next Review**: After POC testing
