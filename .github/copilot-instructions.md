# Copilot instructions for Graphiti

Graphiti is a Python library for temporal knowledge graphs. The core package is
`graphiti_core/`. The REST service is `server/`, and the MCP server is
`mcp_server/`. `AGENTS.md` describes the layout, the commands, and the
conventions in detail. Read it before you review or change code.

## Code review

A maintainer requests a Copilot review on a pull request. Copilot does not
review a pull request without a request. Review the diff against the criteria
below, and comment only when a criterion is not met. Do not comment on style
that `ruff` already enforces.

### Required

- Every Python file starts with the copyright header
  (`Copyright 2024, Zep Software, Inc.` and the Apache 2.0 notice).
- Strings use single quotes, and lines are at most 100 characters
  (`pyproject.toml`).
- No secret, credential, API key, or customer data is in the code or in the
  tests.
- Logging does not include sensitive data: no prompts, no episode content,
  no credentials.
- New behavior has tests under `tests/`. A test file is `test_<feature>.py`,
  and a test function is `test_<behavior>`.
- A test that needs a database is in a file with the `_int` suffix and is
  marked `@pytest.mark.integration`.
- A change to a public interface updates the docs or the examples.

### Check for

- Side effects outside `graphiti_core/driver`, `graphiti_core/cross_encoder`,
  and `graphiti_core/utils`. Other modules are pure helpers.
- A change to a Cypher query that works on Neo4j only. The drivers for
  FalkorDB and Neptune must keep working.
- A change to the bi-temporal fields (`valid_at`, `invalid_at`, `created_at`,
  `expired_at`) that breaks the ordering rules.
- A Pydantic model in `graphiti_core/models` without explicit type hints.
- An async function that blocks the event loop.

### Do not flag

- Code that `make format` or `make lint` already accepts.
- Missing docstrings on private helpers.
- The use of AI assistance by the contributor. AI assistance is acceptable in
  this repository.
