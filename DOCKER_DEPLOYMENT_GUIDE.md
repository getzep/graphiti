# Graphiti Docker 部署指南

> 本文档基于当前仓库里的实际 Docker、Compose、FastAPI 和 MCP 代码整理。
> 重点覆盖两条部署路径：
>
> 1. `mcp_server/`：给 Claude、Cursor、Codex 等编程助手提供 MCP 服务
> 2. 根目录 `docker-compose.yml`：部署 `server/graph_service/` 这个 REST API

---

## 一、先选部署形态

### 方案 A：Neo4j + Graphiti MCP Server

推荐给这些场景：

- 你要给 AI 编程助手接入长期记忆
- 你希望客户端直接通过 MCP 使用 Graphiti
- 你关心的是 `/mcp/` 入口，而不是自己封装 REST API

仓库对应文件：

- Compose：`mcp_server/docker/docker-compose-neo4j.yml`
- 配置：`mcp_server/config/config-docker-neo4j.yaml`
- Dockerfile：`mcp_server/docker/Dockerfile.standalone`

### 方案 B：Neo4j + Graphiti FastAPI 服务

推荐给这些场景：

- 你要让其他服务通过 HTTP 调用 Graphiti
- 你更想用 REST API，而不是 MCP
- 你准备自己实现上层编排或网关

仓库对应文件：

- Compose：`docker-compose.yml`
- Dockerfile：`Dockerfile`
- 服务入口：`server/graph_service/main.py`

### 一句话建议

如果你的目标是“搭建 Neo4j + Graphiti 服务，辅助编程”，优先部署：

```text
Neo4j + mcp_server
```

`server/graph_service` 更适合做你自己系统里的 API 组件。

---

## 二、共同前置条件

- Docker Engine
- Docker Compose
- 可用的 LLM API Key
- 至少 2 核 CPU、4 GB 内存的宿主机

Neo4j 版本建议：

- `graphiti-core` 当前依赖 `neo4j>=5.26.0`
- 仓库里的 compose 使用了 `5.26.0` 或 `5.26.2`

如果你要持久化数据，务必准备：

- Neo4j 数据目录或 named volume
- `.env` 文件
- 备份策略

---

## 三、推荐方案：部署 Neo4j + MCP Server

这一套最贴合“Graphiti 辅助编程”的目标。

### 3.1 目录与关键文件

```text
mcp_server/
├── docker/
│   ├── docker-compose-neo4j.yml
│   └── Dockerfile.standalone
├── config/
│   └── config-docker-neo4j.yaml
├── .env.example
└── main.py
```

### 3.2 环境变量

在 `mcp_server/` 目录下创建 `.env`：

```bash
cd /opt/claude/graphiti/mcp_server
cp .env.example .env
```

建议最少配置：

```bash
OPENAI_API_KEY=your_api_key

NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=demodemo
NEO4J_DATABASE=neo4j

GRAPHITI_GROUP_ID=graphiti_repo
SEMAPHORE_LIMIT=10
```

几个要点：

- `docker-compose-neo4j.yml` 里 Neo4j 的默认账号密码就是 `neo4j / demodemo`
- `GRAPHITI_GROUP_ID` 默认是 `main`，建议在真实项目里显式改掉
- `SEMAPHORE_LIMIT` 默认为 `10`，这是 `mcp_server/src/graphiti_mcp_server.py` 里的默认值

### 3.3 启动命令

```bash
docker compose -f docker/docker-compose-neo4j.yml up -d --build
```

为什么建议 `--build`：

- compose 文件虽然写了 `image: zepai/knowledge-graph-mcp:standalone`
- 但同一个文件也配置了本地 `build`
- 仓库注释已经明确说明：Docker Hub 镜像可能落后于最新 `graphiti-core`

如果你是跟随当前仓库代码部署，本地构建更一致。

### 3.4 这套 compose 实际做了什么

根据 `mcp_server/docker/docker-compose-neo4j.yml`：

- 启动一个 `neo4j` 容器
- 启动一个 `graphiti-mcp` 容器
- 把 `config/config-docker-neo4j.yaml` 挂载到容器内 `/app/mcp/config/config.yaml`
- 设置 `CONFIG_PATH=/app/mcp/config/config.yaml`
- 对外暴露：
  - Neo4j Browser：`7474`
  - Neo4j Bolt：`7687`
  - MCP HTTP：`8000`

### 3.5 验证方式

```bash
# 查看容器状态
docker compose -f docker/docker-compose-neo4j.yml ps

# 查看 MCP 健康检查
curl http://localhost:8000/health

# 查看日志
docker compose -f docker/docker-compose-neo4j.yml logs -f graphiti-mcp
docker compose -f docker/docker-compose-neo4j.yml logs -f neo4j
```

预期：

- `graphiti-mcp` 对 `http://localhost:8000/health` 返回 healthy
- Neo4j Browser 可访问：`http://localhost:7474`
- MCP 客户端接入地址是：`http://localhost:8000/mcp/`

### 3.6 配置层级

`mcp_server/src/config/schema.py` 的实现决定了配置优先级：

```text
CLI 参数 > 环境变量 > YAML 配置 > 默认值
```

这很适合 Docker 部署：

- 稳定默认值放 `config-docker-neo4j.yaml`
- 敏感信息放 `.env`
- 临时覆盖用 `docker compose run ... -- ...`

### 3.7 默认模型与数据库设置

`config/config-docker-neo4j.yaml` 当前默认是：

- transport：`http`
- host：`0.0.0.0`
- port：`8000`
- LLM provider：`openai`
- LLM model：`gpt-4o-mini`
- embedder：`text-embedding-3-small`
- database provider：`neo4j`

如果你使用 OpenAI 兼容端点，变量名是：

```bash
OPENAI_API_URL=https://your-endpoint/v1
```

这里不是 `OPENAI_BASE_URL`。那是 `server/graph_service` 侧的变量名。

### 3.8 生产建议

- 不要把 MCP 端口直接裸露到公网，前面至少加一层反向代理和鉴权
- 不要一直依赖 `latest`，固定镜像标签或固定仓库 commit 构建
- 为 Neo4j 的 `neo4j_data`、`neo4j_logs` 做备份
- 给 `GRAPHITI_GROUP_ID` 做项目级隔离，不要所有仓库共用一个 `main`

---

## 四、备选方案：部署 Neo4j + FastAPI Graph Service

这套来自仓库根目录的 `docker-compose.yml`，适合你自己对接 HTTP API。

### 4.1 目录与关键文件

```text
.
├── docker-compose.yml
├── Dockerfile
└── server/
    └── graph_service/
        ├── main.py
        ├── routers/
        └── config.py
```

### 4.2 环境变量

根目录和 `server/` 下都有示例文件，至少要准备这些变量：

```bash
OPENAI_API_KEY=your_api_key
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_PORT=7687
```

如果 Neo4j 不跟 Graph 服务放在同一个 compose 里，还需要：

```bash
NEO4J_URI=bolt://your-neo4j-host:7687
```

`server/graph_service/config.py` 实际读取的是：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `MODEL_NAME`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

注意：

- `OPENAI_BASE_URL` 只对 `graph_service` 生效
- `embedding_model_name` 虽然定义在 settings 里，但当前 `zep_graphiti.py` 没有使用它

### 4.3 启动命令

在仓库根目录运行：

```bash
docker compose up -d --build
```

这会启动：

- `neo4j`
- `graph`

其中：

- `graph` 使用根目录 `Dockerfile` 构建
- 启动命令固定为 `uv run uvicorn graph_service.main:app --host 0.0.0.0 --port 8000`

### 4.4 端口设计的真实情况

这里有一个很容易写错的点。

虽然 compose 里传了：

```bash
PORT=8000
```

但根目录 `Dockerfile` 的 `CMD` 已经把 uvicorn 端口写死成：

```text
--port 8000
```

也就是说：

- 容器内部端口就是 `8000`
- 如果你要改宿主机端口，改的是 compose 的 `ports`
- 不要指望只改 `PORT` 环境变量就让服务改端口

例如改成 NAS 上常见的 `8010`：

```yaml
ports:
  - "8010:8000"
```

### 4.5 健康检查与文档入口

Graph service 当前暴露的是：

- 健康检查：`GET /healthcheck`
- Swagger：`/docs`
- ReDoc：`/redoc`

验证命令：

```bash
curl http://localhost:8000/healthcheck
docker compose logs -f graph
docker compose logs -f neo4j
```

### 4.6 这个 REST 服务当前有哪些接口

根据 `server/graph_service/routers/` 的实现，现有接口主要是：

| 方法 | 路径 | 作用 |
|------|------|------|
| `POST` | `/messages` | 异步写入消息 |
| `POST` | `/entity-node` | 手工创建实体节点 |
| `DELETE` | `/entity-edge/{uuid}` | 删除事实边 |
| `DELETE` | `/group/{group_id}` | 删除某个分组 |
| `DELETE` | `/episode/{uuid}` | 删除单个 episode |
| `POST` | `/clear` | 清空整图并重建索引 |
| `POST` | `/search` | 按 query 和 `group_ids` 搜索事实 |
| `GET` | `/entity-edge/{uuid}` | 获取单条事实 |
| `GET` | `/episodes/{group_id}?last_n=N` | 获取最近 episodes |
| `POST` | `/get-memory` | 基于消息拼接查询文本再搜索 |

这里要特别纠正一个常见误写：

- 当前服务没有 `/ingest/episodes`
- 当前服务也没有 `/retrieve/search`

如果文档里写了这两个路径，那是和仓库实现不一致的。

### 4.7 REST 调用示例

写入消息：

```bash
curl -X POST http://localhost:8000/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "group_id": "graphiti_repo",
    "messages": [
      {
        "content": "JWT scope validation lives in middleware.",
        "name": "auth_note",
        "role_type": "system",
        "role": "architect",
        "source_description": "design note"
      }
    ]
  }'
```

搜索事实：

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "group_ids": ["graphiti_repo"],
    "query": "Where is JWT scope validation implemented?",
    "max_facts": 5
  }'
```

几点说明：

- `/messages` 返回 `202 Accepted`，因为它和 MCP 的 `add_memory` 一样走后台队列
- `/search` 使用的是 `group_ids` 和 `max_facts`
- 不是旧文档里那种 `group_id + limit` 结构

### 4.8 当前已验证的 NAS 部署示例

你当前已经在线上跑通的就是这一类 REST 部署，关键特征是：

- Neo4j 容器：`neo4j-graph-db`
- Graphiti 容器：`graphiti-api`
- Graphiti 外部端口：`8010`
- Graphiti 内部端口：`8000`
- 健康检查：`/healthcheck`
- Swagger：`/docs`
- Neo4j Browser：`http://192.168.123.104:7474`
- Graphiti REST 基址：`http://192.168.123.104:8010`

按你当前配置，实际部署要点如下：

```yaml
services:
  neo4j:
    image: docker.1ms.run/neo4j:latest
    container_name: neo4j-graph-db
    ports:
      - "7474:7474"
      - "7687:7687"

  graph:
    image: docker.1ms.run/zepai/graphiti:latest
    container_name: graphiti-api
    ports:
      - "${PORT:-8010}:8000"
```

当前 `.env` 里与服务行为直接相关的配置特征是：

- `OPENAI_BASE_URL` 使用了自定义推理端点
- `MODEL_NAME=glm-5`
- `NEO4J_URI=bolt://192.168.123.104:7687`
- `PORT=8010`

文档里不要写入真实密钥；保留成下面这种形式即可：

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
MODEL_NAME=glm-5

NEO4J_URI=bolt://192.168.123.104:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

PORT=8010
```

### 4.9 这套当前部署说明了什么

从你给出的实际运行配置，可以明确得出：

- 你跑的是 `graph_service` 风格的 REST 服务
- 不是 `mcp_server` 风格的 MCP 服务
- 其他项目应该通过 HTTP 调用它，而不是通过 `/mcp/` 接入
- 如果未来想让 Claude / Cursor 直接把 Graphiti 当 MCP 服务使用，需要额外再部署 `mcp_server`

这也意味着：

- 其他项目不需要再配置自己的 `OPENAI_API_KEY`
- 其他项目也不需要直接连接 Neo4j
- 它们只要访问 `graphiti-api` 的 REST 地址即可

### 4.10 在当前 NAS 上继续容器化部署 `mcp_server`

如果你希望：

- 保留现有 `graphiti-api` 给业务项目走 REST
- 再给 Claude / Cursor / Codex 等客户端提供 MCP

那么最合理的方式是：

```text
同一台 NAS 上同时运行 graphiti-api + graphiti-mcp
```

仓库里已经补了一套适合你当前环境的模板文件：

- Compose：`mcp_server/docker/docker-compose-neo4j-external.yml`
- 配置：`mcp_server/config/config-docker-neo4j-external.yaml`
- 环境变量模板：`mcp_server/.env.nas.example`

推荐端口规划：

- `graphiti-api`：`8010 -> 8000`
- `graphiti-mcp`：`8011 -> 8000`

如果你是通过 NAS IP 直接访问 MCP，还需要在 `.env.nas` 里放行 Host：

```bash
MCP_ALLOWED_HOSTS=localhost:*,127.0.0.1:*,192.168.123.104:*
```

否则 `/mcp` 可能返回：

```text
421 Invalid Host header
```

启动方式：

```bash
cd /opt/claude/graphiti/mcp_server
cp .env.nas.example .env.nas
docker compose -f docker/docker-compose-neo4j-external.yml up -d --build
```

启动后：

- MCP 健康检查：`http://192.168.123.104:8011/health`
- MCP Endpoint：`http://192.168.123.104:8011/mcp/`

这套方案的特点是：

- `graphiti-mcp` 不再自己启动 Neo4j
- 它直接复用你现有的 `neo4j-graph-db`
- 和 `graphiti-api` 共用同一套图数据库
- 通过不同端口并存，不互相冲突
- `.env.nas` 是切换 LLM / embedder 端点与模型的主入口，优先改这里，不要直接改 compose YAML

### 4.11 MCP 与 REST 共用 Neo4j 时的建议

最重要的是 `group_id` 策略。

你有两种选择：

1. 共用 `group_id`
   适合你希望 REST 写入的知识，也能立刻被 MCP 检索到
2. 分离 `group_id`
   适合你希望业务系统知识和 AI 助手知识各自隔离

推荐初始做法：

- REST：继续使用业务项目自己的 `group_id`
- MCP：先用 `nas_mcp` 或 `repo_assistant`

如果后面你确认要共享知识，再逐步对齐。

### 4.12 一个关键兼容性提醒：MCP 需要可用 embedder

这点比 REST 部署更需要明确。

当前你 REST 服务里使用的是：

- `OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/`
- `MODEL_NAME=glm-5`

而 `mcp_server` 默认会同时创建：

- LLM client
- Embedder client

如果你给 MCP 也复用同一个 OpenAI 兼容端点，需要确认这个端点：

1. 支持聊天/推理
2. 也支持 embeddings

如果它不支持 embeddings，MCP 侧常见症状会是：

- 写入时报 embedding 相关错误
- 搜索无法返回结果

仓库里新增的 `mcp_server/.env.nas.example` 已经把这个风险写出来了。出现这种情况时，优先做法是：

- 保留 `LLM_PROVIDER=openai` + 你的兼容端点
- 把 `EMBEDDER_PROVIDER` 切到支持 embeddings 的 provider
- 当前已经验证通过的组合是：
  - `OPENAI_API_URL=https://open.bigmodel.cn/api/paas/v4/`
  - `EMBEDDER_PROVIDER=openai`
  - `EMBEDDER_MODEL=qwen3-embedding:0.6b`
  - `EMBEDDER_OPENAI_API_URL=http://192.168.123.74:11434/v1`

### 4.13 当前实测说明：MCP 的主要慢点通常在 LLM，不在 embedder

基于当前实际部署日志，已经验证过：

- MCP 链路可以跑通
- `qwen3-embedding:0.6b` 通常只占毫秒到一秒级
- 主要耗时在 Graphiti 的几个 LLM prompt，例如：
  - `extract_nodes.extract_text`
  - `extract_edges.edge`
  - `extract_nodes.extract_summaries_batch`

因此如果你要继续优化这条部署：

- 优先通过 `.env.nas` 切换 `LLM_MODEL` / `OPENAI_API_URL`
- 不要先把问题归咎于 embedder

---

## 五、Neo4j 部署细节与建议

无论你选 MCP 还是 REST，Neo4j 都是核心依赖。

### 5.1 当前仓库里的版本

- 根目录 compose：`neo4j:5.26.2`
- MCP compose：`neo4j:5.26.0`

两者都满足 `graphiti-core` 的版本要求。

### 5.2 数据持久化

仓库里的 compose 使用的是 named volume：

- 根目录：`neo4j_data`
- `mcp_server/docker/docker-compose-neo4j.yml`：`neo4j_data`、`neo4j_logs`

生产环境建议：

- 保留 named volume 或改绑宿主机目录
- 定期备份 `/data` 和 `/logs`

### 5.3 连接地址怎么写

最常见错误是把容器内连接地址写成 `localhost`。

经验规则：

- 容器访问同 compose 内 Neo4j：`bolt://neo4j:7687`
- 宿主机访问 Neo4j：`bolt://localhost:7687`
- 另一个宿主机访问：`bolt://<host-ip>:7687`

### 5.4 启动等待时间

Neo4j 启动经常比 Graph 服务慢。

仓库里的两个 compose 都已经做了健康检查和 `depends_on`，但如果你拆成多个 compose 或多台机器部署，仍然要自己保证：

1. Neo4j 已经 ready
2. 再启动 Graphiti 容器

---

## 六、部署时最容易踩的坑

### 6.1 把 `mcp_server` 和 `graph_service` 混为一谈

它们不是一个东西：

- `mcp_server`：对外是 `/mcp/` 和 `/health`
- `graph_service`：对外是 `/healthcheck`、`/docs` 以及各类 REST 路径

不要把两个服务的环境变量、端口和路径混写。

### 6.2 改了 `PORT` 但服务没换端口

这在 `graph_service` 部署里最常见。

原因不是 Docker，而是根目录 `Dockerfile` 的 uvicorn 启动命令把端口固定成了 `8000`。

### 6.3 把 `NEO4J_PORT` 当成随便可改的外部端口

根目录 `docker-compose.yml` 里：

- `NEO4J_URI=bolt://neo4j:${NEO4J_PORT:-7687}`
- 端口映射也是 `${NEO4J_PORT:-7687}:${NEO4J_PORT:-7687}`

这意味着它默认假设 Neo4j 容器内部监听也是这个端口。

更稳妥的做法是：

- 让 `NEO4J_PORT` 保持默认 `7687`
- 如果只是要换宿主机端口，显式改映射配置，不要只改变量名

### 6.4 以为写入后立刻可搜索

无论 `mcp_server` 的 `add_memory`，还是 `graph_service` 的 `/messages`，都走后台队列。

如果你刚写完数据就马上查不到，优先看日志，而不是先怀疑 Neo4j。

### 6.5 用错 OpenAI 兼容端点变量名

- `mcp_server`：`OPENAI_API_URL`
- `graph_service`：`OPENAI_BASE_URL`

---

## 七、让其他项目复用你当前的 REST 服务

你现在这套 `graphiti-api` 部署，最适合当成“内网共享记忆服务”来复用。

推荐架构：

```text
项目 A / 项目 B / 项目 C
        │
        ├── HTTP 调用 /messages 写入项目知识
        ├── HTTP 调用 /search 检索项目事实
        └── 各自使用独立 group_id
                │
                ▼
        graphiti-api (8010)
                │
                ▼
           neo4j-graph-db
```

### 7.1 推荐的接入原则

- 一个项目一个 `group_id`
- 所有项目共用一个 Graphiti 服务实例
- 写入统一走 `/messages`
- 检索统一走 `/search`
- 把调用封装成内部 SDK 或基础服务，而不是散落在业务代码里

示例 `group_id`：

- `crm_prod`
- `ops_knowledge`
- `repo_backend`
- `repo_frontend`

### 7.2 当前可供其他项目直接使用的接口

| 能力 | 方法 | 路径 | 备注 |
|------|------|------|------|
| 健康检查 | `GET` | `/healthcheck` | 探活 |
| 写入消息 | `POST` | `/messages` | 异步入队 |
| 搜索事实 | `POST` | `/search` | 最常用 |
| 查看最近记录 | `GET` | `/episodes/{group_id}?last_n=N` | 调试写入 |
| 删除分组 | `DELETE` | `/group/{group_id}` | 重建项目记忆 |
| 查看单条事实 | `GET` | `/entity-edge/{uuid}` | 调试 |
| 删除单条事实 | `DELETE` | `/entity-edge/{uuid}` | 修正错误关系 |

### 7.3 适合写进 Graphiti 的项目信息

最有价值的通常不是原始聊天，而是工程事实：

- 仓库结构说明
- 服务依赖关系
- 部署步骤
- 环境变量约定
- 认证与权限规则
- 故障复盘
- code review 结论
- 发布限制和回滚策略

当前 REST 服务只有 `/messages` 这个写入口时，推荐把这些信息包装成：

- `role_type=system`
- `role=app`、项目名或服务名
- `source_description=architecture_note` / `deploy_note` / `runbook`

### 7.4 Python 项目接入模板

```python
import httpx


class GraphitiRestClient:
    def __init__(self, base_url: str, group_id: str):
        self.base_url = base_url.rstrip('/')
        self.group_id = group_id

    async def add_note(self, content: str, name: str, source_description: str = 'project_note'):
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f'{self.base_url}/messages',
                json={
                    'group_id': self.group_id,
                    'messages': [
                        {
                            'content': content,
                            'name': name,
                            'role_type': 'system',
                            'role': 'app',
                            'source_description': source_description,
                        }
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def search(self, query: str, max_facts: int = 5):
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f'{self.base_url}/search',
                json={
                    'group_ids': [self.group_id],
                    'query': query,
                    'max_facts': max_facts,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def get_episodes(self, last_n: int = 10):
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f'{self.base_url}/episodes/{self.group_id}',
                params={'last_n': last_n},
            )
            resp.raise_for_status()
            return resp.json()
```

当前你自己的部署可以直接这样用：

```python
client = GraphitiRestClient(
    base_url='http://192.168.123.104:8010',
    group_id='repo_backend',
)
```

### 7.5 Node.js / TypeScript 项目接入模板

```ts
export class GraphitiRestClient {
  constructor(
    private readonly baseUrl: string,
    private readonly groupId: string,
  ) {}

  async addNote(content: string, name: string, sourceDescription = 'project_note') {
    const resp = await fetch(`${this.baseUrl}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        group_id: this.groupId,
        messages: [
          {
            content,
            name,
            role_type: 'system',
            role: 'app',
            source_description: sourceDescription,
          },
        ],
      }),
    });

    if (!resp.ok) throw new Error(await resp.text());
    return resp.json();
  }

  async search(query: string, maxFacts = 5) {
    const resp = await fetch(`${this.baseUrl}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        group_ids: [this.groupId],
        query,
        max_facts: maxFacts,
      }),
    });

    if (!resp.ok) throw new Error(await resp.text());
    return resp.json();
  }
}
```

### 7.6 其他项目接入时最容易忽略的点

- `/messages` 成功返回只代表消息进入队列，不代表已经完成抽取
- 同一个 `group_id` 的消息是串行处理的
- 刚写入立刻搜索，可能会因为队列延迟查不到
- 当前服务没有内建鉴权，最好只放在内网
- 如果你要对公网开放，前面至少加反向代理、鉴权和限流

### 7.7 建议补一层你自己的业务适配

对于多个项目长期使用，建议你再封一层轻量服务或 SDK，把这些约定固化下来：

- 自动补 `group_id`
- 自动补 `role_type=system`
- 自动标准化 `source_description`
- 增加重试、超时、日志和监控
- 对接你自己的配置中心

这样后续每个项目接入 Graphiti 的成本会低很多。

---

## 八、生产化建议

### 8.1 反向代理与访问控制

这两个服务都更适合放在内网或受控环境。

推荐加：

- Nginx / Traefik / Caddy
- Basic Auth、JWT 网关或零信任接入
- TLS 终止

### 8.2 监控

至少监控：

- 容器重启次数
- Neo4j 磁盘占用
- Graphiti 容器日志中的 `429`
- 写入队列堆积情况

### 8.3 并发调优

`SEMAPHORE_LIMIT` 不要盲目调大。

建议按 LLM 配额来：

- 低额度：`1-2`
- 中等：`5-10`
- 高额度：`10-15`

如果日志开始出现大量 `429`，先把它往下调。

### 8.4 数据隔离

无论是 MCP 还是 REST，最终都建议按项目、团队或环境区分 `group_id`。

示例：

- `repo_main`
- `repo_prod`
- `repo_docs`

---

## 九、部署完成后的核对清单

### MCP 方案

```bash
docker compose -f mcp_server/docker/docker-compose-neo4j.yml ps
curl http://localhost:8000/health
docker compose -f mcp_server/docker/docker-compose-neo4j.yml logs --tail=100 graphiti-mcp
```

确认：

- `graphiti-mcp` 容器是 `running`
- `/health` 返回 healthy
- 客户端能连 `http://localhost:8000/mcp/`

### REST 方案

```bash
docker compose ps
curl http://localhost:8000/healthcheck
curl http://localhost:8000/docs
docker compose logs --tail=100 graph
```

确认：

- `graph` 与 `neo4j` 都在运行
- `/healthcheck` 正常
- Swagger 能打开

### Neo4j

```bash
curl http://localhost:7474
```

确认：

- Browser 可访问
- Bolt 连接地址和部署拓扑一致

---

## 十、推荐结论

如果你的最终目标是“搭建 Neo4j + Graphiti 服务给编程助手用”，建议采用：

```text
mcp_server/docker/docker-compose-neo4j.yml
```

理由很直接：

- 它和“辅助编程”场景最贴近
- 代码里已经包含 HTTP MCP、队列、配置系统和 Neo4j compose
- 默认工具集已经覆盖写入、搜索、删除、状态检查

如果你要给业务系统提供 REST API，再补上根目录的 `graph_service` 部署。

如果你已经确定自己跑的是 REST 方式，并且需要逐接口接入说明，请继续看：

- `REST_API_REFERENCE.md`
