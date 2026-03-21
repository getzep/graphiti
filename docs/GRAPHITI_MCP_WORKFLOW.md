# Graphiti MCP Workflow Guide

## Purpose

本指南定义 MyStocks 当前项目内对 `graphiti-mcp` 与 `graphiti-api` 的使用边界。

当前结论：

- `graphiti-mcp` 是 AI CLI 的长期记忆层
- `graphiti-api` 当前只作为已部署能力保留，不接入本仓库 runtime 代码
- Mongo control plane 仍是 main/worker 协同状态的唯一真相源

配套运行手册：

- 参见 [GRAPHITI_NAS_RUNBOOK.md](./GRAPHITI_NAS_RUNBOOK.md)

## Verified Endpoints

以下地址已于 `2026-03-20` 在当前环境中实际探活，并统一按 NAS 暴露地址使用：

- Graphiti MCP health:
  - `http://192.168.123.104:8011/health`
- Graphiti MCP endpoint:
  - `http://192.168.123.104:8011/mcp`
- Graphiti API healthcheck:
  - `http://192.168.123.104:8010/healthcheck`
- Graphiti API docs:
  - `http://192.168.123.104:8010/docs`

说明：

- `http://192.168.123.104:8011/mcp/` 会重定向到 `/mcp`
- 项目 MCP 配置统一直接使用 `/mcp`
- 除非调用方本身就在 NAS 本机上，否则不要再使用 `localhost` 作为 Graphiti 入口

## Source-of-Truth Boundary

### Mongo Owns Coordination State

以下信息必须以 Mongo control plane 为准：

- work item creation
- assignment / dispatch
- claim / plan / submit
- ready_for_review / verified / merged
- worker progress percentage and review lifecycle

Graphiti 不得充当以下信息的权威来源：

- `work_item.status`
- 审批结果
- merge 状态
- task ownership

### Graphiti Owns Agent Memory

以下信息适合沉淀到 Graphiti：

- main CLI handoff 摘要
- worker 开工回执的自然语言说明
- 任务分解后的语义摘要
- 关键排障结论
- 文档审核结论
- 架构决策与历史事实

简化判断：

- 问“现在任务状态是什么” -> 查 Mongo
- 问“之前这个问题为什么这么改” -> 查 Graphiti

## How Graphiti Works

要真正理解 `graphiti-mcp`，最好的起点仍然是 Graphiti 仓库中的 `examples/quickstart/`。

这组示例展示的是 Graphiti 本体的直接 Python API，不经过 MCP 包装，因此很适合作为心智模型：

1. 连接到 Neo4j 数据库
   - 直接创建 `Graphiti(neo4j_uri, neo4j_user, neo4j_password)`
   - 这一步对应 `graphiti-mcp` 在服务启动时创建底层 Graphiti client
2. 初始化 Graphiti 索引和约束
   - Quickstart 文档明确把 `build_indices_and_constraints()` 视为图数据库第一次使用时应完成的初始化动作
   - 在 `graphiti-mcp` 中，这一步通常在服务启动阶段自动完成
3. 向图谱中添加 Episodes
   - 示例同时展示了纯文本和结构化 JSON 两种 episode
   - JSON 进入 Graphiti 前仍然会先序列化为字符串
   - 底层会抽实体、关系、摘要和 embedding
4. 使用混合搜索查找关系（边）
   - `graphiti.search(...)` 返回的是事实关系，而不是普通文档片段
   - 它组合了语义、关键词和图结构
   - 在 MCP 中对应 `search_memory_facts`
5. 使用图距离重排搜索结果
   - Quickstart 会先找到一条事实，再取其 `source_node_uuid` 作为中心节点做二次重排
   - 在 MCP 中对应 `search_memory_facts(..., center_node_uuid=...)`
6. 使用预定义配方搜索节点
   - Quickstart 使用 `NODE_HYBRID_SEARCH_RRF` 这个内置 recipe 做节点检索
   - 在 MCP 中对应 `search_nodes`

Quickstart 还给出了一些重要的底层事实：

- Graphiti 的核心数据单元是 `episode -> nodes -> facts`
- `reference_time` 不只是写入时间，它会影响事实的时间语义
- 节点搜索和事实搜索是两条不同路径，不应混用
- 节点检索更适合回答“有哪些对象/实体”
- 事实检索更适合回答“对象之间的关系是什么”

对 MyStocks 来说，这一层理解很重要，因为它决定了使用姿势：

- 当我们调用 `add_memory` 时，本质上是在把自然语言摘要、handoff、架构事实转成 Graphiti 的 episode
- 当我们调用 `search_nodes` 时，本质上是在搜“实体”
- 当我们调用 `search_memory_facts` 时，本质上是在搜“关系”

因此，MyStocks 当前对 Graphiti 的定位不是“状态数据库”，而是：

- 面向 AI CLI 的长期语义记忆层
- 面向历史原因、架构决策、handoff 事实的检索层
- 不承担任务状态真相源职责

## Recommended `group_id` Layout

建议至少按职责隔离：

- `mystocks_spec_main`
  - main CLI 的长期记忆、派单结论、合并判断
- `mystocks_spec_workers`
  - worker 执行中的交付摘要、handoff 事实
- `mystocks_spec_docs`
  - 文档审核、术语收敛、规范决议
- `mystocks_spec_review`
  - review finding、follow-up 结论、验收判断

约束：

- 不要长期把所有内容都写进 `main`
- 不要把任务状态字段复制写入 Graphiti 当作状态库
- 同一类记忆尽量使用稳定 `group_id`，不要每次会话新建随机组

## Main CLI Usage

main CLI 适合在这些时机调用 Graphiti MCP：

1. 派单前
   - 搜索该任务域是否已有历史结论
2. worker 上报后
   - 写入一条 handoff / review memory
3. 合并前
   - 记录最终验收结论和 residual risks
4. 长链路中断恢复时
   - 通过 facts / episodes 恢复上下文

推荐调用顺序：

1. `get_status`
2. `search_nodes`
3. `search_memory_facts`
4. `add_memory`
5. `get_episodes`

## Worker CLI Usage

worker CLI 适合在这些时机调用 Graphiti MCP：

1. 接单开工
   - 读取与本任务相关的历史事实
2. 完成一个批次后
   - 写入批次摘要、验证结果、风险
3. 需要上报 main CLI 之前
   - 写入可复用的 handoff 说明

worker 不应把以下内容只写入 Graphiti 而不写 Mongo：

- 开工回执状态
- plan item 完成状态
- ready_for_review 申请

这些仍必须走 Mongo control plane。

## Graphiti API Status

`graphiti-api` 当前已部署并可用，但本次切换不把它接入仓库 runtime。

当前定位：

- 仅作为后续扩展能力保留
- 适用于未来的程序化写入/检索
- 若要把它接入 `coordctl`、`maestro`、后端服务或自动任务，必须单独立项

推荐后续场景：

- 自动沉淀交付总结到 Graphiti
- 由脚本批量导入架构说明、运维事件
- 做一个本仓库专用 Graphiti SDK

## First-Use Checklist

首次在本项目里用 Graphiti MCP 时，建议：

1. 先确认 `get_status`
2. 再用 `search_nodes`
3. 再用 `search_memory_facts`
4. 最后再调用 `add_memory`
5. 写入后不要立刻假定可检索，必要时用 `get_episodes` 观察落图进度

## Non-Goals For This Change

本次变更明确不做：

- 不把 Mongo 任务状态迁到 Graphiti
- 不在业务代码里新增 `graphiti-api` 调用
- 不自动导出 `TASK.md` / `TASK-REPORT.md` 到 Graphiti
- 不把 Graphiti 当 review gate 或审批系统

---

## MCP Tools Quick Reference

### 可用工具列表

| 工具 | 用途 | 关键参数 |
|------|------|----------|
| `get_status` | 检查服务状态和 Neo4j 连接 | 无 |
| `add_memory` | 添加记忆（异步处理） | `name`, `episode_body`, `group_id`, `source` |
| `get_ingest_status` | 查看一条写入的摄入状态 | `episode_uuid`, `group_id` |
| `search_nodes` | 搜索实体节点 | `query`, `group_ids`, `max_nodes` |
| `search_memory_facts` | 搜索关系事实 | `query`, `group_ids`, `max_facts` |
| `get_episodes` | 获取历史记录 | `group_ids`, `max_episodes` |
| `get_entity_edge` | 获取特定关系 | `uuid` |
| `delete_episode` | 删除记录 | `uuid` |
| `clear_graph` | 清空图数据 | `group_ids` |

### 调用示例

```python
# 1. 检查服务状态
mcp__graphiti-memory__get_status()
# 返回: {"status":"ok","message":"Graphiti MCP server is running..."}

# 2. 添加记忆（异步处理，需等待落图）
mcp__graphiti-memory__add_memory(
    name="架构决策标题",
    episode_body="详细描述决策背景、原因、方案、影响...",
    group_id="mystocks_spec_main",
    source="text",
    source_description="决策记录"
)
# 返回: {"result":{"message":"Episode '...' queued for processing...","episode_uuid":"...","group_id":"mystocks_spec_main","queue_position":1}}

# 3. 等待摄入完成
mcp__graphiti-memory__get_ingest_status(
    episode_uuid="...",
    group_id="mystocks_spec_main"
)
# 返回: {"result":{"state":"queued|processing|completed|failed", ...}}

# 4. 搜索节点（实体）
mcp__graphiti-memory__search_nodes(
    query="自然语言查询，如：为什么选择 TDengine",
    group_ids=["mystocks_spec_main"],
    max_nodes=10
)
# 返回: {"result":{"message":"Nodes retrieved successfully","nodes":[...]}}

# 5. 搜索事实（关系）
mcp__graphiti-memory__search_memory_facts(
    query="Graphiti MCP 地址",
    group_ids=["mystocks_spec_main"],
    max_facts=10
)
# 返回: {"result":{"message":"Facts retrieved successfully","facts":[...]}}

# 6. 获取历史记录
mcp__graphiti-memory__get_episodes(
    group_ids=["mystocks_spec_main"],
    max_episodes=20
)
# 返回: {"result":{"message":"Episodes retrieved","episodes":[...]}}
```

### 返回数据结构

**Node（节点/实体）**:
```json
{
  "uuid": "bf65618a-4318-4562-8733-22438d666d41",
  "name": "Graphiti MCP",
  "labels": ["Entity"],
  "summary": "实体摘要...",
  "group_id": "mystocks_spec_main",
  "created_at": "2026-03-19T20:07:00.320914+00:00"
}
```

**Fact（关系/事实）**:
```json
{
  "uuid": "b2b834e0-7cb4-4082-a376-3820e90c480d",
  "name": "LOCATED_AT",
  "fact": "Graphiti MCP 的地址是 http://192.168.123.104:8011/mcp",
  "source_node_uuid": "...",
  "target_node_uuid": "...",
  "group_id": "mystocks_spec_main",
  "created_at": "2026-03-19T20:07:25.212271Z",
  "valid_at": "2026-03-19T20:06:55.150179Z",
  "invalid_at": null
}
```

---

## Verification Status

**验证日期**: 2026-03-20

| 检查项 | 状态 | 详情 |
|--------|------|------|
| MCP 配置 | ✅ | `.mcp.json` 指向 NAS 地址 |
| 服务状态 | ✅ | `get_status` 返回 `status: ok` |
| 添加记忆 | ✅ | `add_memory` 成功排队处理 |
| 搜索节点 | ✅ | `search_nodes` 返回实体 |
| 搜索事实 | ✅ | `search_memory_facts` 返回关系 |
| 知识图谱 | ✅ | 实体和关系正确提取 |

**已验证的知识图谱示例**:

| 实体 | 关系 | 目标 |
|------|------|------|
| MyStocks | INTEGRATED_AS | Graphiti MCP |
| Graphiti MCP | SERVES_AS | AI CLI 长期记忆层 |
| Graphiti MCP | LOCATED_AT | http://192.168.123.104:8011/mcp |
| Graphiti MCP | DEPLOYED_ON | NAS |
| Graphiti | STORES | Agent Memory |

---

## Usage Tips

### 写入最佳实践

1. **命名清晰**: `name` 应简洁明确，便于后续搜索
2. **内容详尽**: `episode_body` 应包含完整上下文、原因、影响
3. **分组隔离**: 使用正确的 `group_id` 按职责隔离
4. **等待完成**: 优先轮询 `get_ingest_status`，等 `state == completed` 再搜索验证

### 搜索最佳实践

1. **自然语言**: 使用自然语言查询，Graphiti 会做语义匹配
2. **指定分组**: 使用 `group_ids` 限定搜索范围，提高准确性
3. **组合查询**: 先搜节点定位实体，再搜事实了解关系
4. **空结果兜底**: `search_nodes` 现在会在 hybrid 为空时自动做 exact-name / contains fallback

### 判断规则速查

| 问题类型 | 查询目标 | 工具 |
|----------|----------|------|
| "现在任务状态是什么" | Mongo | 不用 Graphiti |
| "之前为什么这么改" | Graphiti | `search_memory_facts` |
| "架构决策历史" | Graphiti | `search_nodes` + `search_memory_facts` |
| "审批结果" | Mongo | 不用 Graphiti |
| "排障结论" | Graphiti | `search_memory_facts` |
| "规范决议" | Graphiti | `search_nodes` |

---

## AI Command Patterns

下面这些模板面向 AI CLI 使用，目的是让 Graphiti MCP 的调用顺序、写入格式和检索范围尽量标准化。

### Pattern 1: 开工前读取历史上下文

适用：

- main CLI 派单前
- worker CLI 接单开工前
- 接手一个中断任务时

推荐顺序：

```python
# 1. 检查 Graphiti 服务状态
mcp__graphiti-memory__get_status()

# 2. 搜索相关实体
mcp__graphiti-memory__search_nodes(
    query="任务主题、模块名、架构名、数据源名",
    group_ids=["mystocks_spec_main", "mystocks_spec_workers"],
    max_nodes=10
)

# 3. 搜索相关事实
mcp__graphiti-memory__search_memory_facts(
    query="为什么这样设计 / 历史上怎么处理 / 已知风险",
    group_ids=["mystocks_spec_main", "mystocks_spec_workers"],
    max_facts=10
)
```

### Pattern 2: worker 完成一个批次后写入摘要

适用：

- 完成一个 patch
- 完成一轮排障
- 完成一个 review 批次

推荐写法：

```python
mcp__graphiti-memory__add_memory(
    name="worker batch summary",
    group_id="mystocks_spec_workers",
    source="text",
    source_description="worker handoff summary",
    episode_body=(
        "任务: xxx\\n"
        "变更: xxx\\n"
        "验证: xxx\\n"
        "风险: xxx\\n"
        "后续建议: xxx"
    )
)
```

最低要求：

- 不要只写“已完成”
- 必须至少写：
  - 任务主题
  - 变更内容
  - 验证结果
  - 风险或待办

### Pattern 3: main CLI 在验收前写入最终结论

适用：

- review 结束后
- 合并前
- 任务关闭前

推荐写法：

```python
mcp__graphiti-memory__add_memory(
    name="main review decision",
    group_id="mystocks_spec_review",
    source="text",
    source_description="main cli review conclusion",
    episode_body=(
        "主题: xxx\\n"
        "验收结论: accepted / needs follow-up\\n"
        "依据: xxx\\n"
        "残余风险: xxx\\n"
        "是否已同步 Mongo: yes"
    )
)
```

### Pattern 4: 按 group 精确检索

当同一主题存在多类记忆时，优先按 group 限定范围：

```python
# 查 main 的架构决策
mcp__graphiti-memory__search_memory_facts(
    query="Graphiti MCP 地址和部署方式",
    group_ids=["mystocks_spec_main"],
    max_facts=10
)

# 查 worker handoff
mcp__graphiti-memory__search_memory_facts(
    query="graphiti nas mcp handoff",
    group_ids=["mystocks_spec_workers"],
    max_facts=10
)

# 查文档规范
mcp__graphiti-memory__search_nodes(
    query="Graphiti workflow guide",
    group_ids=["mystocks_spec_docs"],
    max_nodes=10
)
```

### Pattern 5: 写入后观察落图

`add_memory` 是异步处理，不等于立刻可检索。

建议顺序：

```python
# 1. add_memory
# 2. get_episodes
mcp__graphiti-memory__get_episodes(
    group_ids=["mystocks_spec_workers"],
    max_episodes=10
)

# 3. 再搜索 nodes / facts
```

### Pattern 6: 围绕某个实体扩展检索

如果 `search_memory_facts` 返回了一条明显正确的事实，可以把其中的 `source_node_uuid` 或 `target_node_uuid`
拿来继续做中心节点搜索：

```python
mcp__graphiti-memory__search_memory_facts(
    query="Graphiti MCP 部署位置",
    group_ids=["mystocks_spec_main"],
    max_facts=10,
    center_node_uuid="<uuid from previous fact>"
)
```

这个模式适合：

- 先锁定正确实体
- 再围绕该实体扩展更多事实

---

## NAS Command Recipes

下面这些命令面向人类运维或排障使用，但 AI 在给出 runbook、handoff 或排障建议时，也应优先复用这些标准命令。

### Recipe 1: 健康检查

```bash
curl http://192.168.123.104:8011/health
```

预期：

```json
{"status":"healthy","service":"graphiti-mcp"}
```

### Recipe 2: 查看容器状态

```bash
cd /volume5/docker5/graphiti/mcp_server
docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml ps
```

重点检查：

- `IMAGE` 为 `graphiti-mcp-neo4j-external:local`
- `STATUS` 为 `healthy`
- 端口映射为 `8011->8000`

### Recipe 3: 查看最近日志

```bash
cd /volume5/docker5/graphiti/mcp_server
docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml logs --tail=120 graphiti-mcp
```

重点看：

- `LLM: anthropic / glm-5`
- `Embedder: openai / text_embedding`
- `dimensions: 1024`
- `connected to neo4j database`

### Recipe 4: 标准重建

```bash
cd /volume5/docker5/graphiti/mcp_server

docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml down

docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml build graphiti-mcp

docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml up -d --force-recreate
```

### Recipe 5: 端到端回环

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
- `success_at_seconds: 20.0`

### Recipe 6: 恢复到当前成功版本

```bash
cp -av \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/.env.nas \
  /volume5/docker5/graphiti/mcp_server/.env.nas

cp -av \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/config-docker-neo4j-external.yaml \
  /volume5/docker5/graphiti/mcp_server/config/config-docker-neo4j-external.yaml

cp -av \
  /volume5/docker5/graphiti_backups/2026-03-20-mcp-working/docker-compose-neo4j-external.yml \
  /volume5/docker5/graphiti/mcp_server/docker/docker-compose-neo4j-external.yml
```

然后重新执行 `Recipe 4`。

---

## Common Failure Modes

下面这些问题都不是理论清单，而是本轮 NAS Graphiti 排障中已经实际出现过的故障模式。AI 在生成排障建议时，应优先复用这里的判断和修复动作。

### Case 1: `/health` 正常，但 MCP 写入/检索不正常

现象：

- `curl /health` 返回 `healthy`
- `get_status` 也可能返回 `ok`
- 但 `search_nodes` / `search_memory_facts` 行为异常

判断：

- 这说明“服务进程活着”
- 不代表“当前镜像版本、LLM、embedder、配置组合正确”

优先检查：

1. 当前运行镜像是否为 `graphiti-mcp-neo4j-external:local`
2. 日志中是否出现：
   - `LLM: anthropic / glm-5`
   - `Embedder: openai / text_embedding`
   - `dimensions: 1024`
3. 配置文件是否真的同步到了 NAS 上的正确路径

### Case 2: `search_nodes` 为空，或 `search_nodes` / `search_memory_facts` 都空

现象：

- `add_memory` 返回已入队
- 随后多轮查询都返回：
  - `No relevant nodes found`
  - `No relevant facts found`

判断：

- 先不要怀疑 Neo4j
- 先检查 episode 处理链是否真正完成
- `add_memory` 是异步处理，不是同步落图

优先动作：

1. 先看 `get_episodes`
2. 再看日志里是否出现：
   - `Completed add_episode`
   - `Successfully processed episode`
3. 如果没有，则说明写入链还没完成或已失败

### Case 3: `benchmark_mcp.py` 报 307 或找不到 SSE data

现象：

- 脚本报：
  - `No SSE data line found`
  - 或出现 `307 Temporary Redirect`

判断：

- URL 末尾斜杠问题

正确写法：

```bash
python benchmark_mcp.py \
  --url http://192.168.123.104:8011/mcp
```

不要写：

```text
http://192.168.123.104:8011/mcp/
```

### Case 4: 容器还在使用 registry 旧镜像

现象：

- `docker compose ps` 里 `IMAGE` 仍然显示：
  - `docker.1ms.run/zepai/knowledge-graph-mcp:latest`

判断：

- 当前运行的不是已修复源码构建出的镜像
- 很容易继续复现旧行为

正确状态：

```text
graphiti-mcp-neo4j-external:local
```

修复动作：

```bash
docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml down
docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml build graphiti-mcp
docker compose --env-file .env.nas -f docker/docker-compose-neo4j-external.yml up -d --force-recreate
```

### Case 5: `.env.nas` 看起来“差不多”，但其实还是旧配置

现象：

- 文件里仍然是：
  - `OPENAI_API_URL=https://open.bigmodel.cn/api/anthropic`
  - `EMBEDDER_MODEL=text-embedding-3-small`
  - `EMBEDDER_DIMENSIONS=1536`

判断：

- 这是旧配置
- 会导致 embedder 路径错误或维度不匹配

当前已验证通过的关键配置必须是：

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_URL=https://open.bigmodel.cn/api/anthropic
OPENAI_API_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDER_MODEL=text_embedding
EMBEDDER_DIMENSIONS=1024
```

### Case 6: 使用 `localhost` 导致跨主机调用失败

现象：

- 在非 NAS 机器上访问：
  - `http://localhost:8011/mcp`
  - 失败

判断：

- `localhost` 只指向当前机器自己
- 只有调用方本身就在 NAS 本机上时才成立

项目统一规则：

- 非 NAS 本机调用，一律使用：
  - `http://192.168.123.104:8011/mcp`

### Case 7: 文件同步后仍然没生效

现象：

- 已经执行过拷贝
- 但运行结果像是没更新

高概率原因：

- 文件被同步到了错误目录
- 例如扁平复制到了 `mcp_server/` 根，而不是：
  - `mcp_server/config/`
  - `mcp_server/docker/`
  - `mcp_server/src/services/`

排查方式：

```bash
grep -n 'graphiti-mcp-neo4j-external:local' docker/docker-compose-neo4j-external.yml
grep -n 'provider: ${LLM_PROVIDER:openai}' config/config-docker-neo4j-external.yaml
grep -n 'AsyncAnthropic' src/services/factories.py
grep -n 'LLM_PROVIDER=anthropic' .env.nas
```

### Case 8: embedding 正常，但整条 Graphiti 写入链仍失败

现象：

- embedder API 单独测试通过
- 但 Graphiti 实际写入链仍失败或结果为空

判断：

- 不能据此直接断定“Graphiti 没问题”
- embedding 只是链路中的一段
- 还必须继续看：
  - LLM 端点协议
  - 结构化输出格式
  - Graphiti prompt 阶段日志

本轮结论：

- BigModel 的 `text_embedding` 是可用的
- 但 `glm-5` 在 Graphiti 当前链路里更适合走 Anthropic-compatible 端点
- 因此当前成功组合是：
  - LLM: Anthropic-compatible
  - Embedder: OpenAI-compatible embeddings

---

## Changelog

| 日期 | 变更 |
|------|------|
| 2026-03-20 | 增加 Common Failure Modes 区块 |
| 2026-03-20 | 增加 AI Command Patterns 与 NAS Command Recipes |
| 2026-03-20 | 添加 MCP Tools Quick Reference、Verification Status、Usage Tips |
| 2026-03-20 | 初始版本：定义职责边界、group_id 布局、Main/Worker 使用时机 |
