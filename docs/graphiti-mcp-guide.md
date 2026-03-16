# Graphiti MCP 使用指南

> 本文把仓库中 `mcp_server` 提供的服务称为 `graphiti-mcp`。
> 它不是给业务系统直接调 HTTP 用的，而是给 Claude、Cursor、Codex 等 MCP 客户端接入 Graphiti 记忆能力用的。

## 1. `graphiti-mcp` 是什么

`graphiti-mcp` 是 Graphiti 的 MCP 封装层。它把 Graphiti 的知识图能力暴露成一组 MCP tools，让 AI 客户端能够直接：

- 写入长期记忆
- 搜索实体节点
- 搜索事实关系
- 查看最近摄入的 episode
- 删除错误数据或清理指定知识域

它更像“给 AI 助手用的记忆服务器”，而不是普通 REST API。

## 2. 它能做什么

### 2.1 写入记忆

核心工具：

- `add_memory`

支持三类 source：

- `text`
- `message`
- `json`

适合：

- 把聊天、说明、笔记、运行记录沉淀为长期记忆
- 把结构化 JSON 交给 Graphiti 自动抽实体和关系

要注意两件事：

- 调用会很快返回，因为实际处理走后台队列
- 同一 `group_id` 下的任务会串行处理

### 2.2 搜索实体

核心工具：

- `search_nodes`

适合回答：

- 当前记忆里有哪些组织、文档、地点、事件
- 某个主题相关的实体有哪些
- 某个实体是否已经进入图谱

### 2.3 搜索事实

核心工具：

- `search_memory_facts`

适合回答：

- 某条规则在哪里实现
- 某个对象和另一个对象之间有什么关系
- 某个需求、服务、文档之间的语义连接是什么

### 2.4 回看和维护

回看类工具：

- `get_episodes`
- `get_entity_edge`
- `get_status`

维护类工具：

- `delete_episode`
- `delete_entity_edge`
- `clear_graph`

基础设施探活：

- `GET /health`

## 3. 什么时候选 MCP，而不是 API

优先选 `graphiti-mcp` 的场景：

- 你要把 Graphiti 直接接给 MCP 客户端
- 你希望 AI 助手自己调 `add_memory`、`search_nodes`、`search_memory_facts`
- 你不想自己再封装一层 REST 客户端

优先选 `graphiti-api` 的场景：

- 你的调用方是后端服务、ETL、任务流、内部系统
- 你要显式控制 HTTP 写入和检索

一句话区分：

- 给 AI 客户端接入，用 MCP
- 给程序系统接入，用 API

## 4. 如何把它用到你的项目里

### 4.1 作为 AI 客户端的长期记忆服务

最常见的接法：

1. Claude / Cursor / Codex 通过 MCP 连到 `graphiti-mcp`
2. 在对话或工作流里调用 `add_memory`
3. 在需要上下文时调用 `search_nodes` / `search_memory_facts`

适合：

- 编程助手
- 项目知识助手
- 客服或运营 copilot
- 面向团队的内部 AI 助手

### 4.2 与 `graphiti-api` 并存

推荐的实际部署方式通常是：

- `graphiti-api`
  - 给你的业务系统用
- `graphiti-mcp`
  - 给 AI 客户端用

两者可以共用同一套 Neo4j：

- 想共享记忆：共用 `group_id`
- 想隔离记忆：拆开 `group_id`

### 4.3 通过 `.env.nas` 切换模型和端点

这是这次实际验证后最推荐的做法：

- 不直接改 compose YAML
- 只改 `.env.nas`
- 然后 `docker compose up -d --build`

你可以通过 `.env.nas` 切这些项：

- `LLM_MODEL`
- `OPENAI_API_URL`
- `EMBEDDER_MODEL`
- `EMBEDDER_OPENAI_API_URL`
- `SEMAPHORE_LIMIT`

这意味着你可以很方便地做 A/B 测试：

- 只换 LLM 端点
- 保持 embedder 不变
- 再比较写入到可检索的时延

## 5. 核心使用约定

### 5.1 `group_id` 决定记忆空间

建议：

- 一个项目或业务域一个稳定 `group_id`
- 测试、开发、生产分开
- 多租户按租户隔离

不建议长期把所有内容都塞进：

- `main`

### 5.2 `add_memory` 不等于“立刻可搜索”

`add_memory` 成功只表示：

- 请求已入队
- 后台会继续跑实体抽取、去重、关系抽取、摘要、embedding、入图

它不表示：

- 下一秒就一定能搜到

这条链路在实践里可能是：

- 十几秒
- 几十秒
- 甚至分钟级

主要取决于：

- LLM 端点速度
- LLM 返回格式稳定性
- 当前 prompt 阶段数量

### 5.3 节点搜索和事实搜索不要混用

经验上：

- 想找“有哪些对象”，用 `search_nodes`
- 想找“对象之间的关系”，用 `search_memory_facts`

### 5.4 `text`、`message`、`json` 该怎么选

- `text`
  - 最通用，适合笔记、说明、文档片段
- `message`
  - 适合对话
- `json`
  - 适合结构化对象

如果你用 `json`，传的是 JSON 字符串，不是客户端语言里的对象字面量。

## 6. 最小可用部署

### 6.1 方案 A：本地快速体验，FalkorDB 组合容器

适合第一次跑通：

```bash
cd mcp_server
cp .env.example .env
docker compose -f docker/docker-compose.yml up -d
```

默认会得到：

- MCP endpoint：`http://localhost:8000/mcp/`
- Health：`http://localhost:8000/health`

### 6.2 方案 B：Neo4j + `graphiti-mcp`

适合你已经有 Neo4j：

```bash
cd mcp_server
cp .env.example .env
docker compose -f docker/docker-compose-neo4j.yml up -d
```

### 6.3 方案 C：外部 Neo4j / NAS / 自定义模型端点

这次实际验证后，外部 Neo4j 路径推荐用：

- Compose：`mcp_server/docker/docker-compose-neo4j-external.yml`
- 配置：`mcp_server/config/config-docker-neo4j-external.yaml`
- 环境变量模板：`mcp_server/.env.nas.example`

启动步骤：

```bash
cd mcp_server
cp .env.nas.example .env.nas
docker compose -f docker/docker-compose-neo4j-external.yml up -d --build
```

### 6.4 方案 D：`stdio`

适合只支持标准输入输出的客户端：

```bash
cd mcp_server
uv sync
uv run main.py --transport stdio
```

## 7. 推荐配置方式

当前最推荐的配置思路是：

1. YAML 只负责稳定结构
2. `.env.nas` 负责数据库、模型和端点
3. compose 通过 `env_file` 读取 `.env.nas`

覆盖顺序仍然是：

```text
CLI 参数 > 环境变量 > YAML 配置 > 默认值
```

### 7.1 `.env.nas` 推荐起点

对 NAS / 外部 Neo4j / GLM + Ollama 这条链路，推荐从下面开始：

```bash
MCP_PORT=8011

NEO4J_URI=bolt://192.168.123.104:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

GRAPHITI_GROUP_ID=nas_mcp
MCP_ALLOWED_HOSTS=0.0.0.0:*,localhost:*,127.0.0.1:*,192.168.123.104:*
SEMAPHORE_LIMIT=10

LLM_MODEL=glm-5
OPENAI_API_KEY=your_api_key
OPENAI_API_URL=https://open.bigmodel.cn/api/paas/v4/

EMBEDDER_PROVIDER=openai
EMBEDDER_MODEL=qwen3-embedding:0.6b
EMBEDDER_DIMENSIONS=1024
EMBEDDER_OPENAI_API_KEY=ollama
EMBEDDER_OPENAI_API_URL=http://192.168.123.74:11434/v1
```

### 7.2 切换端点时只改 `.env.nas`

例如你要换 LLM：

- 改 `LLM_MODEL`
- 改 `OPENAI_API_URL`

例如你要换 embedder：

- 改 `EMBEDDER_MODEL`
- 改 `EMBEDDER_OPENAI_API_URL`

然后统一：

```bash
docker compose -f docker/docker-compose-neo4j-external.yml up -d --build
```

不要再直接改 compose YAML。

## 8. 启动后怎么验证

### 8.1 健康检查

```bash
curl http://localhost:8000/health
```

或 NAS 上：

```bash
curl http://127.0.0.1:8011/health
```

### 8.2 MCP endpoint

HTTP transport 下客户端入口是：

```text
http://<host>:<port>/mcp/
```

### 8.3 查看启动日志

建议确认：

- LLM provider / model / base_url
- embedder provider / model / base_url / dimensions
- `Embedder connectivity check passed`

## 9. 第一次接入与使用

### 9.1 如果客户端支持 HTTP MCP

通用配置大致类似：

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "http",
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

### 9.2 如果客户端只支持 `stdio`

通用思路是让客户端执行：

```text
uv run main.py --transport stdio
```

并通过环境变量把数据库和模型配置传进去。

### 9.3 第一次工具调用顺序

建议按这个顺序试：

1. `get_status`
2. `add_memory`
3. `search_nodes`
4. `search_memory_facts`
5. `get_episodes`

## 10. 性能与运行预期

### 10.1 `add_memory` 是后台任务

不要用“请求返回快”来判断写入是否完成。

真正完成路径通常是：

1. `add_memory` 入队
2. Graphiti 后台处理 episode
3. 抽实体、去重、抽关系、摘要、embedding
4. 最后才变成可检索

### 10.2 当前慢点通常在 LLM，不在 embedder

基于这次实际排查经验：

- Ollama embedding 往往是毫秒到一秒级
- 真正的大头通常在 LLM prompt，例如：
  - `extract_nodes.extract_text`
  - `extract_edges.edge`
  - `extract_nodes.extract_summaries_batch`

### 10.3 你可以直接看性能日志

如果你同步了当前仓库的最新 `mcp_server` 代码，容器日志里会出现：

- `LLM timing prompt=... elapsed_ms=...`
- `Embedder timing model=... elapsed_ms=...`

这比继续猜“是不是 embedder 慢”更有效。

### 10.4 做 A/B 测试的正确方式

推荐：

1. 固定 embedder
2. 只切换 `.env.nas` 里的 LLM 端点或模型
3. 重建容器
4. 用同一条 `add_memory` 探针比较：
   - `LLM timing`
   - `Successfully processed episode`
   - 实际可检索时间

## 11. 常见问题

### 11.1 `/health` 正常，但 tools 不可用

优先检查：

- URL 是否指向 `/mcp/`
- transport 是否一致

### 11.2 写入成功，但搜索不到

先查：

1. `get_status`
2. `get_episodes`
3. 容器日志

不要只盯着检索接口。

### 11.3 为什么 OpenAI 兼容端点看起来能调，但写入还是慢或失败

因为 Graphiti 不只是简单聊天：

- 它会要求结构化 JSON
- 会有多轮 prompt
- 兼容端点如果返回 fenced JSON、`answer` 包装、列表包装，都会增加重试成本

### 11.4 为什么我改了端点却没生效

优先检查：

- MCP 侧是不是改了 `OPENAI_API_URL`
- 而不是误改成 API 侧的 `OPENAI_BASE_URL`
- compose 是否真的重新 build

### 11.5 如果我要给业务系统接入，不是给 AI 客户端接入

那就不要优先用 MCP，直接看 [graphiti-api-guide.md](./graphiti-api-guide.md)。
