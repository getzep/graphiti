# Graphiti API 使用指南

> 本文把仓库中 `server/graph_service` 提供的 FastAPI 服务称为 `graphiti-api`。
> 它适合被你的应用、后端服务、工作流或脚本直接调用，是“程序访问 Graphiti”的 HTTP 入口。

## 1. `graphiti-api` 是什么

`graphiti-api` 是 Graphiti 的 REST API 封装层。它把 Graphiti 的核心能力暴露为普通 HTTP 接口，让你的项目可以像调用任何内部服务一样调用它，而不需要先理解 MCP 协议或 AI 客户端的 tool 调用方式。

如果你的调用方是下面这些，优先考虑 `graphiti-api`：

- Web 后端
- 任务队列或 ETL
- 定时同步脚本
- 内部工具平台
- 业务服务之间的 HTTP 集成

一句话理解：

- 程序调用 Graphiti，用 `graphiti-api`
- AI 客户端调用 Graphiti，用 `graphiti-mcp`

## 2. 它能做什么

### 2.1 持续写入知识

`graphiti-api` 最常用的写入口是：

- `POST /messages`

它适合把下面这些内容持续写入 Graphiti：

- 对话记录
- 架构说明
- 产品规则
- 工单和运维事件
- 人工整理的事实说明

除此之外，它还支持：

- `POST /entity-node`

这个接口适合你已经在服务外部完成了结构化处理，想直接把一个实体节点写进图谱。

### 2.2 检索事实和上下文

检索相关接口主要有：

- `POST /search`
  - 用自然语言查 fact
- `POST /get-memory`
  - 用对话消息作为上下文查 fact
- `GET /entity-edge/{uuid}`
  - 直接取一条具体事实

这里要注意：Graphiti 的检索重点是“图里的事实关系”，不是简单返回原文片段。

### 2.3 调试写入是否真的完成

排查写入问题时，最有用的接口通常不是继续查 `/search`，而是：

- `GET /episodes/{group_id}?last_n=N`

它更适合回答：

- 这批数据有没有进入 episodic layer
- 后台处理有没有真正跑完
- 问题是写入没进队列，还是入图后没命中检索

### 2.4 做维护和清理

维护类接口包括：

- `DELETE /entity-edge/{uuid}`
- `DELETE /episode/{uuid}`
- `DELETE /group/{group_id}`
- `POST /clear`

推荐使用边界：

- 清一个知识域，优先 `DELETE /group/{group_id}`
- 清整张图，只在测试/重建场景用 `POST /clear`

## 3. 什么时候选 API，而不是 MCP

优先选 `graphiti-api` 的场景：

- 你要在自己的服务里显式控制写入和检索
- 你希望通过普通 HTTP 接口接入 Graphiti
- 你要把 Graphiti 藏在自己的业务后端后面
- 你不希望把 Graphiti tools 直接暴露给 AI 客户端

优先选 `graphiti-mcp` 的场景：

- 你要把 Graphiti 直接挂给 Claude、Cursor、Codex 等 MCP 客户端
- 你希望 AI 助手自己调 `add_memory`、`search_nodes`、`search_memory_facts`

一个常见做法是两者并存：

- `graphiti-api` 给你的业务系统用
- `graphiti-mcp` 给 AI 客户端用

两者可以共用同一套图数据库：

- 想共享记忆：用同一个 `group_id`
- 想隔离记忆：用不同 `group_id`

## 4. 如何接入你的项目

### 4.1 作为“长期记忆后端”

最常见的接法是：

1. 你的应用把消息、笔记、事件通过 `/messages` 写入
2. 在需要回答问题时，通过 `/search` 或 `/get-memory` 取回相关 facts
3. 把 facts 注入你自己的提示词或业务逻辑

适合：

- 聊天应用
- AI 助手后端
- 企业知识平台
- 工单助手

### 4.2 作为“结构化知识沉淀服务”

如果你的项目已经有结构化对象，比如：

- 客户档案
- 项目模块
- 流程定义
- 资产清单

你可以：

- 在服务外部做预处理
- 再通过 `/entity-node` 或 `/messages` 写入

推荐原则：

- 自己已经很确定的数据，用 `/entity-node`
- 仍以文本/上下文为主的数据，用 `/messages`

### 4.3 作为“调试和回放入口”

如果你的项目需要排查“为什么这次没搜到”，建议保留一套轻量的调试能力：

- 查 `/episodes/{group_id}`
- 查 `/search`
- 查具体 edge UUID

这样你能把问题区分成：

- 写入没进队列
- 队列没处理完
- 已入图但检索没命中

## 5. 核心使用约定

### 5.1 `group_id` 是最重要的隔离字段

建议把 `group_id` 当成稳定知识域，而不是临时会话 ID。

推荐示例：

- `product_docs`
- `repo_backend`
- `customer_acme_prod`
- `ops_runbook`

不建议：

- 所有内容都塞进 `main`
- 同一个项目频繁切换 `group_id`
- 把测试和生产数据混进同一 `group_id`

### 5.2 `/messages` 是异步入队，不是同步入图

`POST /messages` 返回 `202 Accepted` 只表示：

- 请求已接收
- 已进入后台处理队列

它不表示：

- 实体抽取已完成
- 关系已落库
- 下一秒搜索一定命中

实践上要接受最终一致性：

- 写入后立刻查，可能为空
- 处理时间可能是几十秒，也可能到分钟级，取决于底层 LLM

### 5.3 `name`、`role_type`、`role`、`source_description` 不只是装饰

当前实现会把消息组装成更结构化的 episode 文本，因此这些字段会影响：

- 后续检索命中率
- 调试体验
- 多来源数据的可区分性

推荐：

- `name` 用业务上有意义的标识
- `source_description` 写清来源，比如 `architecture_note`、`ops_event`、`product_requirement`

### 5.4 写入后优先看 `episodes`

建议排查顺序：

1. `POST /messages`
2. `GET /episodes/{group_id}`
3. `POST /search`

如果 `episodes` 都没有，问题更可能在：

- 队列尚未处理到
- 写入请求本身有问题
- 模型或数据库连接异常

## 6. 最小可用部署

### 6.1 前置条件

至少准备：

- Docker 和 Docker Compose
- Neo4j
- 一个可用的模型 API Key

当前 `graphiti-api` 这条代码路径推荐按 Neo4j 走，因为它直接读取：

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

### 6.2 环境变量

在仓库根目录准备 `.env`，至少包括：

```bash
OPENAI_API_KEY=your_api_key
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_PORT=7687
```

如果你要接 OpenAI 兼容端点，而不是官方 OpenAI，再补：

```bash
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
MODEL_NAME=gpt-4o-mini
```

关键区别必须记住：

- `graphiti-api` 读的是 `OPENAI_BASE_URL`
- 不是 `OPENAI_API_URL`

另外，确保你配置的兼容端点真的支持 Graphiti 用到的 OpenAI 风格结构化调用，而不是只“名字上兼容”。

### 6.3 启动服务

在仓库根目录执行：

```bash
docker compose up -d
```

这会启动：

- `graph`
  - 对外 `8000`
  - 即本文说的 `graphiti-api`
- `neo4j`
  - Browser 默认 `7474`
  - Bolt 默认 `7687`

### 6.4 启动后验证

健康检查：

```bash
curl http://localhost:8000/healthcheck
```

期望：

```json
{"status":"healthy"}
```

Swagger：

```text
http://localhost:8000/docs
```

## 7. 第一次写入和检索

### 7.1 写入一条消息

```bash
curl -X POST http://localhost:8000/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "group_id": "demo_repo",
    "messages": [
      {
        "content": "JWT scope validation is implemented in AuthMiddleware before requests reach controllers.",
        "name": "auth_rule_001",
        "role_type": "system",
        "role": "backend",
        "source_description": "architecture_note"
      }
    ]
  }'
```

典型响应：

```json
{
  "message": "Messages added to processing queue",
  "success": true
}
```

### 7.2 先查 episodes

```bash
curl 'http://localhost:8000/episodes/demo_repo?last_n=10'
```

### 7.3 再查 facts

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "group_ids": ["demo_repo"],
    "query": "Where is JWT scope validation implemented?",
    "max_facts": 5
  }'
```

### 7.4 对话型场景用 `/get-memory`

```bash
curl -X POST http://localhost:8000/get-memory \
  -H 'Content-Type: application/json' \
  -d '{
    "group_id": "demo_repo",
    "max_facts": 5,
    "messages": [
      {
        "role_type": "user",
        "role": "developer",
        "content": "Where is JWT scope validation implemented?"
      }
    ]
  }'
```

## 8. 常见问题

### 8.1 写入成功，但搜索为空

最常见的原因：

- 队列还没处理完
- `group_id` 不一致
- 模型或 Neo4j 配置有问题

优先查：

- `/episodes/{group_id}`
- `docker compose logs graph`

### 8.2 OpenAI 兼容端点为什么没生效

因为 `graphiti-api` 读的是：

- `OPENAI_BASE_URL`

如果你写成：

- `OPENAI_API_URL`

这边不会生效。

### 8.3 `graphiti-api` 和 `graphiti-mcp` 能不能一起用

可以，而且通常值得一起用：

- 程序系统走 API
- AI 客户端走 MCP

要不要共享记忆，只取决于你是否共用同一个 `group_id`。

### 8.4 什么时候不建议一上来就用 API

如果你的目标是让 Claude、Cursor、Codex 直接拿 Graphiti 当工具用，优先看 [graphiti-mcp-guide.md](./graphiti-mcp-guide.md)。
