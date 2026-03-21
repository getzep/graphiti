# Graphiti NAS Deployment Record 2026-03-21

## Summary

2026-03-21 已确认并修复 NAS `graphiti-mcp` 部署版本漂移问题。

根因不是 `mystocks_spec` 调用侧，也不是 repo-local adapter 解析，而是 NAS 上运行中的 `graphiti-mcp` 实例落后于 `/opt/claude/graphiti` 当前源码。

## Root Cause

live `add_memory` 原始响应曾经只返回：

- `message`

缺少：

- `episode_uuid`
- `group_id`
- `queue_position`

这会让 `mystocks_spec` 的 preflight 逻辑退化为：

- `ingest_status: best_effort`

同时，standalone 镜像链路还存在两个部署级问题：

1. 镜像默认只安装 PyPI `graphiti-core`
2. standalone build context 没有把 repo-local `graphiti_core/` 一起打进容器

因此容器里实际运行逻辑与工作树源码不一致。

## Fix

本次修复包含：

- `mcp_server` transport 兼容：
  - 优先解析 `structuredContent.result`
- `mcp_server` CLI 配置覆盖修复：
  - `--host` / `--port` 真正覆盖 YAML
- `mcp_server` standalone build 修复：
  - build context 改为 repo root
  - 镜像显式复制 `mcp_server/*`
  - 镜像显式复制 `graphiti_core/`
- standalone 运行时依赖收敛：
  - `search_nodes` fallback helper 下沉到 `mcp_server/src/utils/node_name_lookup.py`
  - 避免容器直接依赖未发布的 `graphiti_core.search.node_name_lookup`
- Docker pin 对齐：
  - standalone 默认 `graphiti-core` 版本对齐到 `0.28.2`

## Verification

### NAS MCP Benchmark

命令：

```bash
python benchmark_mcp.py \
  --url http://192.168.123.104:8011/mcp \
  --group-id-prefix nasready \
  --sleep-seconds 5 \
  --max-attempts 6 \
  --keep-data
```

结果：

- `episode_uuid` 已返回
- `INGEST_STATUS`: `processing -> completed`
- `nodes_found: true`
- `facts_found: true`
- `ingest_state: completed`
- `success_at_seconds: 15.2`

### MyStocks Real Preflight

命令：

```bash
GRAPHITI_MCP_URL=http://192.168.123.104:8011/mcp \
python /opt/claude/mystocks_spec/scripts/runtime/maestro_collab.py \
  --mongo-uri 'mongodb://mongo:c790414J@localhost:27017/?authSource=admin' \
  --mongo-db mystocks_coord \
  work preflight 2026-03-14-active-tree-legacy-cleanup-mystocks-spec2 \
  --actor-cli main \
  --write-memory \
  --max-wait-seconds 30 \
  --output json
```

结果：

- `server_status: ok`
- `ingest_status: completed`
- `search_outcome: hit`
- `errors: []`

## Final State

2026-03-21 起，真实 NAS Graphiti + MyStocks Mongo 写链路已闭环：

- `add_memory`
- `get_ingest_status`
- `search_nodes`
- `search_memory_facts`
- `mystocks_spec work preflight --write-memory`

全部通过。
