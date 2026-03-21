# Graphiti NAS Runbook

## Purpose

本手册定义 MyStocks 当前对 Graphiti 的唯一运行方式：

- 只保留 NAS `192.168.123.104` 作为 Graphiti 运行地
- 不在本地和 NAS 同时保留 Graphiti 运行实例
- 只使用 NAS 暴露地址，不再使用 `localhost`

## Runtime Topology

NAS 上与 Graphiti 相关的运行容器只有 3 个：

- `neo4j-graph-db`
- `graphiti-api`
- `graphiti-mcp`

当前统一入口：

- Graphiti MCP
  - `http://192.168.123.104:8011/mcp`
- Graphiti MCP health
  - `http://192.168.123.104:8011/health`
- Graphiti API
  - `http://192.168.123.104:8010`
- Neo4j Bolt
  - `bolt://192.168.123.104:7687`

## Working Configuration

当前已验证通过的组合如下：

- LLM provider
  - `anthropic`
- LLM model
  - `glm-5`
- Anthropic-compatible endpoint
  - `https://open.bigmodel.cn/api/anthropic`
- Embedder provider
  - `openai`
- Embedder model
  - `text_embedding`
- Embedder dimensions
  - `1024`
- OpenAI-compatible embeddings endpoint
  - `https://open.bigmodel.cn/api/paas/v4`
- Graphiti group
  - `nas_mcp`

## Required Files

NAS 上 `graphiti-mcp` 成功版本依赖以下文件：

- `graphiti_core/graphiti.py`
- `mcp_server/.env.nas`
- `mcp_server/benchmark_mcp.py`
- `mcp_server/pyproject.toml`
- `mcp_server/uv.lock`
- `mcp_server/config/config-docker-neo4j-external.yaml`
- `mcp_server/docker/Dockerfile.standalone`
- `mcp_server/docker/docker-compose-neo4j-external.yml`
- `mcp_server/src/graphiti_mcp_server.py`
- `mcp_server/src/utils/node_name_lookup.py`

说明：

- `docker-compose-neo4j-external.yml` 必须使用本地镜像标签 `graphiti-mcp-neo4j-external:local`
- 不应继续依赖 registry 中的旧 `latest` 镜像作为最终运行版本
- standalone 镜像必须从 repo root build context 构建，这样容器内才能带上 repo-local `graphiti_core/`

## Sync

从 WSL2 同步到 NAS：

```bash
cd /opt/claude/graphiti

rsync -rv --relative \
  graphiti_core/ \
  mcp_server/main.py \
  mcp_server/benchmark_mcp.py \
  mcp_server/pyproject.toml \
  mcp_server/uv.lock \
  mcp_server/config/ \
  mcp_server/src/ \
  mcp_server/docker/Dockerfile.standalone \
  mcp_server/docker/docker-compose-neo4j-external.yml \
  john@192.168.123.104:/volume5/docker5/graphiti/
```

注意：

- 不要把 standalone 镜像只当作 `mcp_server/` 子树构建
- 如果这次漏掉 `graphiti_core/`，容器内会重新回退到 PyPI 发布版 `graphiti-core`
- 如果漏掉 `benchmark_mcp.py`，NAS 上 probe 输出会缺少 `INGEST_STATUS[...]` 诊断信息

## Deploy

在 NAS 上执行：

```bash
cd /volume5/docker5/graphiti/mcp_server

docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml down

docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml build graphiti-mcp

docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml up -d --force-recreate
```

部署后检查：

```bash
docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml ps
```

预期：

- `IMAGE` 为 `graphiti-mcp-neo4j-external:local`
- `STATUS` 为 `healthy`

## Verify

健康检查：

```bash
curl http://192.168.123.104:8011/health
```

预期返回：

```json
{"status":"healthy","service":"graphiti-mcp"}
```

端到端验证：

```bash
cd /volume5/docker5/graphiti/mcp_server

python benchmark_mcp.py \
  --url http://192.168.123.104:8011/mcp \
  --group-id-prefix perfprobe \
  --sleep-seconds 20 \
  --max-attempts 5
```

当前成功基线：

- `nodes_found: true`
- `facts_found: true`
- `ingest_state: completed`
- `success_at_seconds: 15.2`

## 2026-03-21 Verified Result

2026-03-21 在 NAS `192.168.123.104` 上实测通过：

- `add_memory` raw 返回带有 `episode_uuid` / `group_id` / `queue_position`
- `benchmark_mcp.py` 观测到 `processing -> completed`
- `search_nodes` 命中
- `search_memory_facts` 命中
- `mystocks_spec work preflight --write-memory` 返回：
  - `server_status: ok`
  - `ingest_status: completed`
  - `search_outcome: hit`
  - `errors: []`

这说明真实 NAS Graphiti + MyStocks Mongo 的写链路已经闭环。

## Backup

备份当前成功版本：

```bash
mkdir -p /volume5/docker5/graphiti_backups/2026-03-20-mcp-working

mkdir -p /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/config
mkdir -p /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/docker
mkdir -p /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/src/utils
mkdir -p /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/graphiti_core

cp -av \
  /volume5/docker5/graphiti/mcp_server/.env.nas \
  /volume5/docker5/graphiti/mcp_server/benchmark_mcp.py \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/

cp -av \
  /volume5/docker5/graphiti/mcp_server/pyproject.toml \
  /volume5/docker5/graphiti/mcp_server/uv.lock \
  /volume5/docker5/graphiti/mcp_server/src/graphiti_mcp_server.py \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/

cp -av \
  /volume5/docker5/graphiti/mcp_server/config/config-docker-neo4j-external.yaml \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/config/

cp -av \
  /volume5/docker5/graphiti/mcp_server/docker/Dockerfile.standalone \
  /volume5/docker5/graphiti/mcp_server/docker/docker-compose-neo4j-external.yml \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/docker/

cp -av \
  /volume5/docker5/graphiti/mcp_server/src/utils/node_name_lookup.py \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/src/utils/

cp -av \
  /volume5/docker5/graphiti/graphiti_core/graphiti.py \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/graphiti_core/
```

## Restore

恢复到当前成功版本：

```bash
cp -av \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/.env.nas \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/benchmark_mcp.py \
  /volume5/docker5/graphiti/mcp_server/

cp -av \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/pyproject.toml \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/uv.lock \
  /volume5/docker5/graphiti/mcp_server/

cp -av \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/graphiti_mcp_server.py \
  /volume5/docker5/graphiti/mcp_server/src/graphiti_mcp_server.py

cp -av \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/config/config-docker-neo4j-external.yaml \
  /volume5/docker5/graphiti/mcp_server/config/config-docker-neo4j-external.yaml

cp -av \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/docker/Dockerfile.standalone \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/docker/docker-compose-neo4j-external.yml \
  /volume5/docker5/graphiti/mcp_server/docker/

cp -av \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/mcp_server/src/utils/node_name_lookup.py \
  /volume5/docker5/graphiti/mcp_server/src/utils/node_name_lookup.py

cp -av \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/graphiti_core/graphiti.py \
  /volume5/docker5/graphiti/graphiti_core/graphiti.py
```

恢复后重新部署：

```bash
cd /volume5/docker5/graphiti/mcp_server

docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml down

docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml build graphiti-mcp

docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml up -d --force-recreate
```

## MyStocks Integration Rule

如果调用方不是在 NAS 本机上运行，Graphiti MCP 配置必须使用：

- `http://192.168.123.104:8011/mcp`

不要再使用：

- `http://localhost:8011/mcp`

## Status

当前状态：

- MCP health 已通过
- `get_status` 已通过
- `add_memory` 已通过
- `get_ingest_status` 已提供显式摄入状态
- `search_nodes` 已通过
- `search_memory_facts` 已通过
- NAS-only 部署已验证通过
