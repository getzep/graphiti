# Graphiti MCP Changeset Audit 2026-03-21

## Scope

本次 changeset 目标是把 Graphiti MCP 从“可用但需要猜测状态”推进到“可观测、可等待、可验证”，并把仓库内依赖旧接口的测试与脚本迁移到当前协议。

本次工作主要覆盖四个方面：

1. MCP 本体能力
2. Ingest 可观测性与 benchmark
3. 节点检索可靠性与更早可见性
4. 仓库内测试基础设施和回归脚本迁移

## Implemented Changes

### 1. MCP API and Queue Lifecycle

已实现：

- `add_memory` 返回：
  - `episode_uuid`
  - `group_id`
  - `queue_position`
- 新增 `get_ingest_status`
- `QueueService` 增加按 `episode_uuid` 的状态跟踪：
  - `queued`
  - `processing`
  - `completed`
  - `failed`
  - `last_error`
  - `queued_at`
  - `started_at`
  - `processed_at`

主要文件：

- `mcp_server/src/graphiti_mcp_server.py`
- `mcp_server/src/services/queue_service.py`
- `mcp_server/src/models/response_types.py`

### 2. Search Reliability

已实现：

- `search_nodes` 在 hybrid 为空时自动降级到 exact-name / contains fallback
- fallback 逻辑下沉到 repo-local helper

主要文件：

- `graphiti_core/search/node_name_lookup.py`
- `mcp_server/src/graphiti_mcp_server.py`

### 3. Earlier Node Visibility

已实现：

- `Graphiti.add_episode()` 在完整边/summary/后处理完成前，先把 resolved entity nodes 提前落库
- `search_nodes` 可在 `get_ingest_status.state == processing` 时先命中节点

主要文件：

- `graphiti_core/graphiti.py`

### 4. UUID Semantics Fix

已修复：

- `add_memory` 引入 `episode_uuid` 后，`Graphiti.add_episode(..., uuid=...)` 不再假定该 episode 必须已经存在
- 如果 `uuid` 不存在，会按该 `uuid` 新建 episode
- bulk 路径同步修复

主要文件：

- `graphiti_core/graphiti.py`

### 5. Status Introspection

已实现：

- `get_status.details.index_check`
- 当前对 Neo4j fulltext index 做只读体检：
  - `missing`
  - `stale`
  - `observed`

主要文件：

- `mcp_server/src/graphiti_mcp_server.py`

### 6. Benchmark and Live Probes

已实现：

- `mcp_server/benchmark_mcp.py` 优先使用 `get_ingest_status`
- summary 输出增加：
  - `ingest_state`
  - `success_at_seconds`

主要文件：

- `mcp_server/benchmark_mcp.py`

### 7. Test Infrastructure Modernization

已实现：

- 新增 raw MCP HTTP 测试客户端：
  - `mcp_server/tests/http_mcp_test_client.py`
- 新增 ingest 等待 helper：
  - `mcp_server/tests/ingest_wait_helpers.py`
- `graphiti_test_client()` 现在会：
  - 在当前环境下优先起临时 HTTP 子进程
  - 继承当前环境
  - 叠加 `.env.nas`
  - 显式传 `config-docker-neo4j-external.yaml`
  - 关闭 telemetry
- 修复并发请求固定 JSON-RPC id 导致的挂起

主要文件：

- `mcp_server/tests/test_fixtures.py`
- `mcp_server/tests/http_mcp_test_client.py`
- `mcp_server/tests/ingest_wait_helpers.py`

## Verification Evidence

### Unit / Focused Regression

已确认通过：

- `pytest mcp_server/tests/test_queue_service_status.py -q`
- `pytest mcp_server/tests/test_graphiti_mcp_server_unit.py -q`
- `pytest mcp_server/tests/test_node_name_lookup_unit.py -q`
- `pytest mcp_server/tests/test_benchmark_mcp.py -q`
- `pytest mcp_server/tests/test_graphiti_ingest_unit.py -q`
- `pytest mcp_server/tests/test_ingest_wait_helpers_unit.py -q`
- `pytest mcp_server/tests/test_load_and_async_compat_unit.py -q`

### Async / Stress Representative Runs

已确认通过：

- `timeout 120s pytest mcp_server/tests/test_async_operations.py -q -k 'test_sequential_queue_processing or test_mixed_operation_concurrency'`
- `timeout 180s pytest mcp_server/tests/test_async_operations.py -q -k 'test_large_response_streaming or test_incremental_processing'`
- `timeout 240s pytest mcp_server/tests/test_stress_load.py -q -k test_sustained_load`
- `timeout 240s pytest mcp_server/tests/test_stress_load.py -q -k test_spike_load`
- `timeout 240s pytest mcp_server/tests/test_stress_load.py -q -k test_connection_pool_exhaustion`
- `timeout 300s pytest mcp_server/tests/test_stress_load.py -q -k test_gradual_degradation`
- `timeout 300s pytest mcp_server/tests/test_stress_load.py -q -k test_memory_leak_detection`

### Full Async / Stress Suites

已确认通过：

- `timeout 420s pytest mcp_server/tests/test_async_operations.py -q`
  - `12 passed`
- `timeout 420s pytest mcp_server/tests/test_stress_load.py -q`
  - `7 passed`

### Integration / Live Scripts

已确认通过：

- `python mcp_server/tests/test_http_integration.py http 127.0.0.1 18011`
- `python mcp_server/tests/test_mcp_transports.py http 127.0.0.1 18011`
- `python mcp_server/tests/test_integration.py 127.0.0.1 18011`
- `python mcp_server/tests/test_mcp_integration.py`

### Full MCP Test Directory

已确认通过：

- `timeout 900s pytest mcp_server/tests -q`
  - `95 passed`

## Live Runtime Observations

在当前本地代码 + `.env.nas` + `config-docker-neo4j-external.yaml` 条件下，已经验证：

- `/health` 正常
- `get_status.details.index_check.status == "ok"`
- benchmark 成功
- `add_memory -> get_ingest_status -> search_nodes/search_memory_facts` 链路闭环

单独节点可见性探针结果：

- `first_node_visible_s = 15.9`
- `completed_s = 31.9`

说明：

- 节点可在 `processing` 阶段先可见
- 完整 ingest 完成稍后到达

## Known Boundaries

### 1. `queue_position` Semantics

当前 `queue_position` 表示“提交时待处理队列中的相对位置”，不是严格的全局单调提交序号。

这意味着：

- 连续快速提交时，队列 worker 可能已经弹走队首
- 第二次提交返回 `1` 不是错误

### 2. FalkorDB-Specific Integration

`test_falkordb_integration.py` 已调整为：

- 有本地 FalkorDB 时运行
- 当前环境无 FalkorDB 时明确跳过

它不再错误地退回默认 OpenAI/embedder 配置并卡在启动阶段。

### 3. Telemetry Noise

测试路径已经默认关闭 telemetry。

服务本体默认运行时并未把 telemetry 逻辑移除，只是在测试/fixture 路径里避免了噪音对 MCP 通信和回归结果的干扰。

## Suggested Commit Split

建议拆成 4 组提交：

### Commit A: MCP Runtime Semantics

- `mcp_server/src/graphiti_mcp_server.py`
- `mcp_server/src/services/queue_service.py`
- `mcp_server/src/models/response_types.py`
- `graphiti_core/search/node_name_lookup.py`
- `graphiti_core/graphiti.py`
- `mcp_server/benchmark_mcp.py`
- `mcp_server/main.py`

目标：

- 新 MCP 能力
- ingest 状态
- search fallback
- 早期节点可见
- uuid 语义修复
- benchmark 迁移

### Commit B: Test Infrastructure

- `mcp_server/tests/http_mcp_test_client.py`
- `mcp_server/tests/ingest_wait_helpers.py`
- `mcp_server/tests/test_fixtures.py`
- `mcp_server/tests/conftest.py`
- `mcp_server/tests/test_ingest_wait_helpers_unit.py`
- `mcp_server/tests/test_load_and_async_compat_unit.py`
- `mcp_server/tests/test_graphiti_ingest_unit.py`
- `mcp_server/tests/test_queue_service_status.py`
- `mcp_server/tests/test_graphiti_mcp_server_unit.py`

目标：

- 测试基座统一
- helper 和 fixture 稳定化
- 最小回归单测

### Commit C: Test Script Migration

- `mcp_server/tests/test_http_integration.py`
- `mcp_server/tests/test_mcp_transports.py`
- `mcp_server/tests/test_integration.py`
- `mcp_server/tests/test_mcp_integration.py`
- `mcp_server/tests/test_comprehensive_integration.py`
- `mcp_server/tests/test_stdio_simple.py`
- `mcp_server/tests/test_async_operations.py`
- `mcp_server/tests/test_stress_load.py`
- `mcp_server/tests/test_falkordb_integration.py`
- `mcp_server/tests/README.md`

目标：

- 所有旧脚本迁到当前 MCP 接口
- 当前等待语义与参数语义统一

### Commit D: Docs / Runbooks

- `docs/graphiti-mcp-guide.md`
- `docs/GRAPHITI_MCP_WORKFLOW.md`
- `docs/GRAPHITI_NAS_RUNBOOK.md`
- `docs/GRAPHITI_NAS_DEPLOYMENT_RECORD_2026-03-21.md`
- `mcp_server/docs/nas-mcp-deployment-playbook.md`
- `QUICK_START.md`
- `mcp_server/README.md`

目标：

- 对外文档、运行手册、部署手册一致

## Review Notes

当前工作区仍然是脏的，包含与本批次无关的其他修改。

因此：

- `git status`
- `git diff --stat`
- `gitnexus_detect_changes(scope="all")`

只能作为“整个工作区现状”参考，不能直接等价为“本次 changeset 独有内容”。

本审计文档列出的文件集合，才是本批次建议优先审核和提交的范围。
