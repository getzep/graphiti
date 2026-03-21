# Graphiti MCP Commit Plan 2026-03-21

## Purpose

这份文档只覆盖**当前工作区仍然是 dirty 的文件**，用于把本轮剩余改动整理成可审核、可提交的批次。

注意：

- 它不是对“本轮所有历史工作”的完整重述
- 它只针对当前 `git status` 仍然显示为 modified / untracked 的文件
- 其他已经不在当前 dirty worktree 里的运行时改动，不在这里重复列出

## Current Dirty Files In Scope

当前与本轮 Graphiti MCP 迁移直接相关、且仍然 dirty 的文件有：

- `QUICK_START.md`
- `docs/graphiti-mcp-guide.md`
- `docs/GRAPHITI_MCP_WORKFLOW.md`
- `docs/GRAPHITI_MCP_2026-03-21_CHANGESET_AUDIT.md`
- `mcp_server/README.md`
- `mcp_server/main.py`
- `mcp_server/tests/conftest.py`
- `mcp_server/tests/http_mcp_test_client.py`
- `mcp_server/tests/test_async_operations.py`
- `mcp_server/tests/test_comprehensive_integration.py`
- `mcp_server/tests/test_falkordb_integration.py`
- `mcp_server/tests/test_fixtures.py`
- `mcp_server/tests/test_graphiti_ingest_unit.py`
- `mcp_server/tests/test_http_integration.py`
- `mcp_server/tests/test_integration.py`
- `mcp_server/tests/test_load_and_async_compat_unit.py`
- `mcp_server/tests/test_mcp_integration.py`
- `mcp_server/tests/test_mcp_transports.py`
- `mcp_server/tests/test_queue_service_status.py`
- `mcp_server/tests/test_stdio_simple.py`
- `mcp_server/tests/test_stress_load.py`

## Suggested Commit Split

### Commit 1: Test Infrastructure and Client Transport

Goal:

- unify the MCP test transport on the stable HTTP-backed helper path
- make `graphiti_test_client()` reliable in the current NAS/Neo4j-backed environment

Files:

- `mcp_server/main.py`
- `mcp_server/tests/conftest.py`
- `mcp_server/tests/http_mcp_test_client.py`
- `mcp_server/tests/test_fixtures.py`
- `mcp_server/tests/test_http_integration.py`
- `mcp_server/tests/test_integration.py`
- `mcp_server/tests/test_mcp_integration.py`
- `mcp_server/tests/test_mcp_transports.py`
- `mcp_server/tests/test_stdio_simple.py`

Suggested message:

```text
stabilize mcp test client transport and env handling
```

Suggested command:

```bash
git add \
  mcp_server/main.py \
  mcp_server/tests/conftest.py \
  mcp_server/tests/http_mcp_test_client.py \
  mcp_server/tests/test_fixtures.py \
  mcp_server/tests/test_http_integration.py \
  mcp_server/tests/test_integration.py \
  mcp_server/tests/test_mcp_integration.py \
  mcp_server/tests/test_mcp_transports.py \
  mcp_server/tests/test_stdio_simple.py
```

### Commit 2: Async and Stress Regression Migration

Goal:

- migrate async/load regression tests to current MCP tool names, parameters, and ingest-aware waiting
- keep representative async and stress suites green under the current runtime model

Files:

- `mcp_server/tests/test_async_operations.py`
- `mcp_server/tests/test_stress_load.py`
- `mcp_server/tests/test_graphiti_ingest_unit.py`
- `mcp_server/tests/test_load_and_async_compat_unit.py`
- `mcp_server/tests/test_queue_service_status.py`

Suggested message:

```text
migrate async and stress regressions to current mcp semantics
```

Suggested command:

```bash
git add \
  mcp_server/tests/test_async_operations.py \
  mcp_server/tests/test_stress_load.py \
  mcp_server/tests/test_graphiti_ingest_unit.py \
  mcp_server/tests/test_load_and_async_compat_unit.py \
  mcp_server/tests/test_queue_service_status.py
```

### Commit 3: Comprehensive Integration Alignment

Goal:

- align the heavy integration suite with the stabilized test client semantics
- preserve coverage while removing old tool-name/parameter assumptions

Files:

- `mcp_server/tests/test_comprehensive_integration.py`

Suggested message:

```text
align comprehensive integration suite with current mcp api
```

Suggested command:

```bash
git add mcp_server/tests/test_comprehensive_integration.py
```

### Commit 4: Docs and Audit Trail

Goal:

- document the current MCP usage model
- capture the final audited state of the changeset

Files:

- `QUICK_START.md`
- `docs/graphiti-mcp-guide.md`
- `docs/GRAPHITI_MCP_WORKFLOW.md`
- `docs/GRAPHITI_MCP_2026-03-21_CHANGESET_AUDIT.md`
- `mcp_server/README.md`

Suggested message:

```text
document current graphiti mcp workflow and test audit
```

Suggested command:

```bash
git add \
  QUICK_START.md \
  docs/graphiti-mcp-guide.md \
  docs/GRAPHITI_MCP_WORKFLOW.md \
  docs/GRAPHITI_MCP_2026-03-21_CHANGESET_AUDIT.md \
  mcp_server/README.md
```

## Verification to Re-Run Before Commit

Before any actual commit, re-run:

```bash
timeout 900s pytest /opt/claude/graphiti/mcp_server/tests -q
```

Expected:

- `95 passed`

Also re-run:

```bash
ruff check /opt/claude/graphiti/mcp_server/tests \
  /opt/claude/graphiti/mcp_server/src \
  /opt/claude/graphiti/mcp_server/main.py \
  /opt/claude/graphiti/docs \
  /opt/claude/graphiti/QUICK_START.md \
  /opt/claude/graphiti/mcp_server/README.md
```

## Files Explicitly Excluded From This Plan

当前工作区里仍有其他 dirty / untracked 文件，例如：

- `.gitignore`
- `AGENTS.md`
- `CLAUDE.md`
- `Dockerfile`
- `README.md`
- `server/README.md`
- 以及其他 `.claude/`, `.omc/`, `docs/superpowers/` 类文件

这些不属于本提交计划，除非你明确要把它们一起纳入本批次审核。
