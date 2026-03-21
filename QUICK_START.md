# Graphiti 快速入门：让编程助手记住你的代码库

> 本文档面向“用 Graphiti 辅助编程”的场景。
> 如果你的目标是让 Claude、Cursor、Codex 等 MCP 客户端记住项目结构、约定、历史决策和排障经验，优先看本文。

---

## 一、先选对入口

这个仓库里实际上有 3 种使用方式：

| 场景 | 推荐入口 | 适合谁 |
|------|-----------|--------|
| 给编程助手提供长期记忆 | `mcp_server/` | Claude、Cursor、Codex、支持 MCP 的 AI IDE |
| 给你自己的应用提供 HTTP API | `server/graph_service/` | 你要自己写客户端或服务编排 |
| 直接在 Python 代码里调用 Graphiti | `graphiti_core/` | 你要做深度定制、嵌入现有 Python 系统 |

如果你的目标是“辅助编程”，推荐顺序是：

1. `mcp_server/` + Neo4j
2. 需要自定义业务接口时，再考虑 `server/graph_service/`
3. 只有在你要直接写 Python 集成时，才直接使用 `graphiti_core`

### 1.1 先判断你当前跑的是哪一种 Graphiti 服务

部署完成后，最先要分清楚自己跑的是 `MCP` 还是 `REST API`。

| 判断项 | MCP 服务 | REST 服务 |
|------|------|------|
| 常见容器名 | `graphiti-mcp` | `graphiti-api` |
| 健康检查 | `/health` | `/healthcheck` |
| 主要入口 | `/mcp/` | `/docs`、`/messages`、`/search` |
| 常见环境变量 | `GRAPHITI_GROUP_ID`、`CONFIG_PATH` | `OPENAI_BASE_URL`、`MODEL_NAME` |
| 典型用途 | 给 Claude / Cursor / Codex 接入 MCP | 给其他业务项目直接走 HTTP 调用 |

按你当前提供的线上配置：

- Neo4j 容器名：`neo4j-graph-db`
- Graphiti 容器名：`graphiti-api`
- 外部端口：`8010 -> 8000`
- 健康检查：`http://localhost:8000/healthcheck`

结论就是：

```text
你当前运行的是 Graphiti REST 服务，不是 MCP 服务。
```

如果你已经在 NAS 上跑起来了这一套，可以直接跳到后面的“其他项目如何通过当前 REST 服务接入”。

### 1.2 本仓库目前能不能本地提供 MCP 服务

可以。

仓库当前已经包含完整的本地 MCP 服务实现，入口在：

- `mcp_server/main.py`
- `mcp_server/src/graphiti_mcp_server.py`

当前代码明确支持 3 种 transport：

- `http`
- `stdio`
- `sse`（已标注 deprecated）

也就是说，以本仓库当前状态，你本地可以这样提供 MCP：

```bash
cd /opt/claude/graphiti/mcp_server
uv sync
uv run main.py --transport http --config config/config-docker-neo4j.yaml
```

或者提供 `stdio` 给本地 MCP 客户端：

```bash
cd /opt/claude/graphiti/mcp_server
uv sync
uv run main.py --transport stdio --config config/config-docker-neo4j.yaml
```

如果你已经有 Neo4j，本地提供 MCP 没有结构性阻碍。真正需要满足的是：

- 可用的数据库连接
- 可用的 LLM / embedder 配置
- `mcp_server` 依赖安装完成

如果你后面想把“当前 REST 服务”和“本地 MCP 服务”并行使用，最常见做法是：

1. 保留 `graphiti-api` 给业务项目调用
2. 额外在本机或开发机运行 `mcp_server` 给 AI 助手接入

如果你希望把 `mcp_server` 也放到当前 NAS 上，仓库里现在也有现成模板：

- `mcp_server/docker/docker-compose-neo4j-external.yml`
- `mcp_server/config/config-docker-neo4j-external.yaml`
- `mcp_server/.env.nas.example`

它的设计目标就是：

- 复用你当前 NAS 上已有的 `neo4j-graph-db`
- 和现有 `graphiti-api` 并存
- 通过单独端口暴露 MCP，例如 `8011 -> 8000`
- 把数据库、LLM、embedder 的切换入口统一收敛到 `.env.nas`

---

## 二、Graphiti 在辅助编程里能做什么

和普通向量记忆不同，Graphiti 会把信息组织成 `Episode`、`Entity`、`Fact` 这类图结构，并保留时间语义。

在编码场景里，最有价值的用法通常是：

- 记住代码库入口、模块关系、调用链、部署拓扑
- 记住团队约定，比如“权限判断统一放在什么中间件里”
- 记住排障过程和历史结论，而不是只存一段原始对话
- 在大仓库里把“事实”搜出来，而不是只靠关键词 grep
- 随着项目变化，保留“现在是什么”和“之前是什么”的差异

一个很实用的心智模型：

- `Episode`：原始材料，比如一次 code review 结论、一段架构说明、一份发布事故复盘
- `Entity`：项目里的对象，比如 `AuthService`、Neo4j、订单服务、权限中间件
- `Fact`：它们之间的关系，比如“`AuthService` 调用 `TokenProvider`”
- `group_id`：隔离不同项目、不同团队、不同环境的数据命名空间

---

## 三、5 分钟跑起来：MCP + Neo4j

这一节走仓库里现成的 Neo4j 部署路径：

- Compose 文件：`mcp_server/docker/docker-compose-neo4j.yml`
- 配置文件：`mcp_server/config/config-docker-neo4j.yaml`
- MCP HTTP 入口：`http://localhost:8000/mcp/`
- 健康检查：`http://localhost:8000/health`

### 3.1 前置条件

- Docker / Docker Compose
- 一个可用的 LLM API Key
- Neo4j 5.26+
- 建议准备一个稳定的 `group_id`

`group_id` 在 Graphiti 里很重要。根据 `graphiti_core/helpers.py` 的校验规则，它只能包含：

- 字母
- 数字
- `_`
- `-`

例如：

```text
graphiti_repo
graphiti-main
backend_v2
```

不要用空格、斜杠、中文、冒号。

### 3.2 配置环境变量

在 `mcp_server/` 目录下准备 `.env`：

```bash
cd /opt/claude/graphiti/mcp_server
cp .env.example .env
```

最小可用配置可以写成：

```bash
OPENAI_API_KEY=your_api_key

NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=demodemo
NEO4J_DATABASE=neo4j

GRAPHITI_GROUP_ID=graphiti_repo
SEMAPHORE_LIMIT=10
```

说明：

- `GRAPHITI_GROUP_ID` 是默认命名空间；如果客户端没显式传 `group_id`，就会落到这里
- `SEMAPHORE_LIMIT` 控制并发提取/写入的上限；`mcp_server/src/graphiti_mcp_server.py` 默认是 `10`
- 模型和 embedder 的默认值不在 `.env` 里，而在 `config/config-docker-neo4j.yaml` 里

### 3.3 启动服务

```bash
docker compose -f docker/docker-compose-neo4j.yml up -d --build
```

为什么这里建议 `--build`：

- `mcp_server/docker/docker-compose-neo4j.yml` 既支持直接拉镜像，也支持本地构建
- 仓库里的注释已经说明，Docker Hub 镜像可能落后于最新 `graphiti-core`
- 你如果是基于当前仓库代码写文档或做联调，本地构建更稳妥

### 3.4 验证服务

```bash
# MCP 服务健康检查
curl http://localhost:8000/health

# 查看容器状态
docker compose -f docker/docker-compose-neo4j.yml ps

# 查看服务日志
docker compose -f docker/docker-compose-neo4j.yml logs -f graphiti-mcp
```

预期：

- `http://localhost:8000/health` 返回类似 `{"status":"healthy","service":"graphiti-mcp"}`
- Neo4j Browser 可访问：`http://localhost:7474`
- 默认账号密码来自 compose：`neo4j / demodemo`

### 3.5 把 MCP 客户端指到 Graphiti

如果你的客户端支持 HTTP MCP，直接把服务地址配置为：

```text
http://localhost:8000/mcp/
```

注意这里不是 `/health`，也不是根路径 `/`，而是 `/mcp/`。

---

## 四、如果你的客户端只支持 `stdio`

有些客户端只能通过本地进程接入 MCP。这时直接运行 `mcp_server/main.py`。

### 4.1 安装依赖

```bash
cd /opt/claude/graphiti/mcp_server
uv sync
```

### 4.2 启动方式

仓库里已经给了示例配置文件：`mcp_server/config/mcp_config_stdio_example.json`

你可以按这个模板改成自己的绝对路径：

```json
{
  "mcpServers": {
    "graphiti": {
      "transport": "stdio",
      "command": "uv",
      "args": [
        "run",
        "/opt/claude/graphiti/mcp_server/main.py",
        "--transport",
        "stdio",
        "--config",
        "/opt/claude/graphiti/mcp_server/config/config-docker-neo4j.yaml"
      ],
      "env": {
        "OPENAI_API_KEY": "your_api_key",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "demodemo",
        "GRAPHITI_GROUP_ID": "graphiti_repo",
        "SEMAPHORE_LIMIT": "10"
      }
    }
  }
}
```

几点说明：

- `main.py` 会优先读取 `mcp_server/.env`
- 配置优先级按 `mcp_server/src/config/schema.py` 的实现是：
  `CLI 参数 > 环境变量 > YAML 配置 > 默认值`
- HTTP 是默认 transport；显式写 `--transport stdio` 才会走标准输入输出模式

---

## 五、编程助手拿到哪些能力

`mcp_server/src/graphiti_mcp_server.py` 暴露的核心工具有：

| 工具 | 用途 | 典型场景 |
|------|------|----------|
| `add_memory` | 写入一条 episode | 记录架构说明、评审结论、排障经验 |
| `get_ingest_status` | 查看一条写入的摄入状态 | 等待异步处理完成 |
| `search_nodes` | 搜索实体节点 | 找模块、服务、表、概念 |
| `search_memory_facts` | 搜索事实关系 | 找“谁依赖谁”“谁负责什么” |
| `get_episodes` | 查看原始记录 | 复查之前喂给 Graphiti 的文本 |
| `get_entity_edge` | 查看某条事实详情 | 调试事实抽取结果 |
| `delete_episode` | 删除原始记录 | 清理错误写入 |
| `delete_entity_edge` | 删除事实边 | 修正错误关系 |
| `clear_graph` | 清空某个 `group_id` 的图数据 | 重建某个项目的记忆 |
| `get_status` | 检查服务状态 | 联调和排障 |

两个很关键的实现细节：

1. `add_memory` 是异步排队的，不是同步立即入图。
2. 队列按 `group_id` 串行处理，同一个项目内连续写入更安全。
3. `add_memory` 会回传 `episode_uuid`，可以配合 `get_ingest_status` 显式等待完成。

这意味着：

- 你刚写完一批 memory，立刻搜索可能会有短暂延迟
- 如果你要给一个仓库持续补充记忆，尽量固定使用同一个 `group_id`
- 更稳的顺序是 `add_memory -> get_ingest_status -> search_nodes/search_memory_facts`

---

## 六、怎么把 Graphiti 真正用于辅助编程

### 6.1 先建“项目记忆”，再问具体问题

推荐先把这些高价值信息喂进去：

- 仓库入口和核心目录
- 服务拓扑和依赖关系
- 测试/构建/部署命令
- 认证、权限、事务、缓存等横切规则
- 团队约定和禁忌
- 最近一段时间的关键改动和事故复盘

这类信息比“把 README 整份塞进去”更值钱，因为它更接近“工程事实”。

### 6.2 一个仓库一个 `group_id`

推荐做法：

- `graphiti_repo`：仓库级长期记忆
- `graphiti_repo_main`：主干稳定知识
- `graphiti_repo_feature_x`：大分支或专项改造

不推荐：

- 把所有项目都塞进同一个 `main`
- 用随机字符串当 `group_id`

### 6.3 先搜事实，再改代码

最常见的高收益工作流：

1. 先让助手搜索现有记忆
2. 再结合代码搜索和静态分析确认
3. 修改完成后把关键决策写回 Graphiti

你可以直接这样描述任务：

```text
先在 Graphiti 里搜索这个仓库里和认证流程相关的事实，再开始改代码。
```

```text
先查一下 Graphiti 里是否已经记录了 deploy、Neo4j、MCP 的约定。
```

```text
把这次排障结论写入 Graphiti，group_id 用 graphiti_repo。
```

### 6.4 善用 `center_node_uuid`

如果你已经通过 `search_nodes` 找到一个核心节点，再用它作为中心点搜索事实，结果通常更准。

例如：

1. 先搜索节点：`AuthService`
2. 取返回节点的 `uuid`
3. 再调用 `search_memory_facts(query=..., center_node_uuid=...)`

这对应的正是 `Graphiti.search(..., center_node_uuid=...)` 的能力。

---

## 七、其他项目如何通过当前 REST 服务接入

如果你现在运行的是你提供的这套配置：

- Neo4j：`neo4j-graph-db`
- Graphiti：`graphiti-api`
- 外部地址：`http://192.168.123.104:8010`

那么其他项目应当把它当成一个内部 REST 记忆服务来用。

### 7.1 当前可直接使用的入口

| 能力 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 健康检查 | `GET` | `/healthcheck` | 服务是否存活 |
| 写入消息 | `POST` | `/messages` | 异步入队，写入记忆 |
| 搜索事实 | `POST` | `/search` | 按 `group_ids` + `query` 搜索 |
| 查看最近记录 | `GET` | `/episodes/{group_id}?last_n=10` | 调试写入结果 |
| 删除分组 | `DELETE` | `/group/{group_id}` | 清理某个项目的图数据 |
| Swagger | `GET` | `/docs` | 在线调试接口 |

按你当前部署，常用地址是：

```text
REST API: http://192.168.123.104:8010
Swagger:  http://192.168.123.104:8010/docs
Health:   http://192.168.123.104:8010/healthcheck
Neo4j:    http://192.168.123.104:7474
```

对其他项目来说，这一点很重要：

- 它们只需要知道 `graphiti-api` 的 HTTP 地址
- 不需要直接持有 `OPENAI_API_KEY`
- 也不需要直接连接 Neo4j

也就是说，Graphiti 在这里承担的是“统一记忆服务”的角色。

如果你需要逐个接口查看请求体、响应体和常见注意事项，请继续看：

- `REST_API_REFERENCE.md`

### 7.2 推荐接入模式

最稳妥的做法是：

1. 每个项目固定一个 `group_id`
2. 所有非结构化说明都通过 `/messages` 写入
3. 所有检索都先走 `/search`
4. 调试时再看 `/episodes/{group_id}`

例如：

- `crm_prod`
- `ops_docs`
- `repo_backend`
- `repo_frontend`

这样多个项目可以共用一个 Graphiti 服务，但数据不会混在一起。

### 7.3 非聊天文本也可以写入

当前 REST 服务没有单独暴露“原始 episode 文本导入”接口，最直接的办法是把任意说明包装成一条 `message`。

例如这些内容都适合写入：

- 架构说明
- 部署步骤
- 代码 review 结论
- 排障记录
- 系统约定

推荐写法：

- `role_type`: 用 `system`
- `role`: 用项目名、服务名或 `app`
- `source_description`: 写清来源，例如 `architecture_note`、`deploy_note`、`runbook`

### 7.4 `curl` 示例

写入一条项目说明：

```bash
curl -X POST http://192.168.123.104:8010/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "group_id": "repo_backend",
    "messages": [
      {
        "content": "JWT scope validation is implemented in AuthMiddleware before the controller layer.",
        "name": "auth_rule_001",
        "role_type": "system",
        "role": "backend",
        "source_description": "architecture_note"
      }
    ]
  }'
```

搜索项目事实：

```bash
curl -X POST http://192.168.123.104:8010/search \
  -H 'Content-Type: application/json' \
  -d '{
    "group_ids": ["repo_backend"],
    "query": "Where is JWT scope validation implemented?",
    "max_facts": 5
  }'
```

查看最近写入：

```bash
curl 'http://192.168.123.104:8010/episodes/repo_backend?last_n=10'
```

清理一个项目的图数据：

```bash
curl -X DELETE http://192.168.123.104:8010/group/repo_backend
```

### 7.5 Python 项目接入示例

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

推荐初始化方式：

```python
client = GraphitiRestClient(
    base_url='http://192.168.123.104:8010',
    group_id='repo_backend',
)
```

### 7.6 Node.js / TypeScript 项目接入示例

```ts
type SearchResponse = {
  facts: Array<{
    uuid: string;
    name: string;
    fact: string;
    valid_at: string | null;
    invalid_at: string | null;
    created_at: string;
    expired_at: string | null;
  }>;
};

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

  async search(query: string, maxFacts = 5): Promise<SearchResponse> {
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

### 7.7 其他项目接入时要注意什么

- `/messages` 返回成功只代表“已入队”，不代表已经完成抽取和入图
- 如果一个项目刚写完就马上搜索，可能会有几秒延迟
- 当前服务没有鉴权层，建议只暴露在内网或反向代理后面
- 最好把调用封装成一个内部 SDK，而不是在每个项目里散落 `curl`
- `group_id` 一定要稳定，不要今天叫 `repo1`，明天叫 `backend_repo`

一个很实用的接入套路：

1. 项目启动或发版后，把架构说明、约束、运行手册写进 Graphiti
2. AI 助手或应用侧检索时，优先查对应 `group_id`
3. 每次重大变更后再补写一条说明，形成持续记忆

---

## 八、10 分钟端到端示例

下面用你当前已经部署好的 REST 服务，演示一次完整流程。

目标：

- 为 `repo_backend` 建立一组基础工程记忆
- 写入 3 条项目事实
- 搜索 2 次验证结果

### 8.1 第一步：写入 3 条工程知识

```bash
curl -X POST http://192.168.123.104:8010/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "group_id": "repo_backend",
    "messages": [
      {
        "content": "AuthMiddleware performs JWT scope validation before requests reach controllers.",
        "name": "auth_rule_001",
        "role_type": "system",
        "role": "backend",
        "source_description": "architecture_note"
      },
      {
        "content": "RefundService is responsible for approval checks before triggering payment rollback.",
        "name": "refund_rule_001",
        "role_type": "system",
        "role": "backend",
        "source_description": "service_note"
      },
      {
        "content": "Deployment requires Redis, PostgreSQL, and Graphiti to be available before worker startup.",
        "name": "deploy_rule_001",
        "role_type": "system",
        "role": "ops",
        "source_description": "deploy_note"
      }
    ]
  }'
```

预期：

- 返回 `202 Accepted`
- 返回体里会提示消息已进入处理队列

### 8.2 第二步：查看最近记录

```bash
curl 'http://192.168.123.104:8010/episodes/repo_backend?last_n=10'
```

这一步主要用来确认数据已经落到对应 `group_id`。

### 8.3 第三步：搜索认证相关事实

```bash
curl -X POST http://192.168.123.104:8010/search \
  -H 'Content-Type: application/json' \
  -d '{
    "group_ids": ["repo_backend"],
    "query": "Where is JWT scope validation implemented?",
    "max_facts": 5
  }'
```

### 8.4 第四步：搜索部署相关事实

```bash
curl -X POST http://192.168.123.104:8010/search \
  -H 'Content-Type: application/json' \
  -d '{
    "group_ids": ["repo_backend"],
    "query": "What dependencies are required before worker startup?",
    "max_facts": 5
  }'
```

### 8.5 这个示例说明了什么

只要其他项目知道一件事：

```text
Graphiti REST 地址 + 自己的固定 group_id
```

它就已经可以开始使用 Graphiti 作为共享工程记忆层。

---

## 九、AI 开发者工作流

如果使用 Graphiti 的是 AI coding agent、Copilot 类助手或内部 agent，推荐按这个节奏来。

### 9.1 什么时候写入

适合写入的时机：

- 看完项目 README / 架构文档之后
- 完成一次 code review 之后
- 解决一次线上故障之后
- 完成一次大范围重构之后
- 梳理出部署约束、环境变量约定之后

不适合写入的内容：

- 临时猜测
- 未确认的推断
- 大量重复日志
- 没有工程价值的闲聊

### 9.2 什么时候检索

在这些任务开始前，先检索 Graphiti 的收益很高：

- 要改已有模块，但还没弄清入口
- 要判断历史约定是否已经存在
- 要确认某个服务/中间件/规则由谁负责
- 要查以前处理过的部署坑或故障结论

### 9.3 推荐工作流

1. 先用 `group_id` 锁定当前项目
2. 用 `/search` 搜索已有工程事实
3. 再结合代码搜索、测试、运行日志做二次确认
4. 改动完成后，把新的工程结论通过 `/messages` 写回去

### 9.4 一条实用原则

Graphiti 适合存“工程事实”，不适合存“推测”。

好的写入内容通常像这样：

- “权限校验发生在 `AuthMiddleware`，不是 controller”
- “退款审批逻辑属于 `RefundService`”
- “worker 启动前依赖 Redis、PostgreSQL 和 Graphiti”

而不是：

- “我猜可能是这里处理的”
- “大概和认证有关”

---

## 十、默认配置里有哪些值得注意的点

从仓库当前配置可以提炼出这些经验：

### 10.1 Neo4j 不是默认数据库

`mcp_server/config/config.yaml` 的默认数据库提供方是 `falkordb`，不是 Neo4j。

如果你要走 Neo4j：

- 用 `config/config-docker-neo4j.yaml`
- 或者显式传 `--database-provider neo4j`

### 10.2 默认模型不在 `.env.example` 里

对 `mcp_server` 来说，默认模型配置来自 YAML：

- LLM：`gpt-4o-mini`
- Embedder：`text-embedding-3-small`

如果你要改模型，有两种方式：

1. 直接改 `config/config-docker-neo4j.yaml`
2. 运行时传 CLI 参数，比如 `--model`

但如果你走的是 NAS / 外部 Neo4j 那条部署路径，推荐做法已经不是直接改 YAML，而是：

1. 保持 `config-docker-neo4j-external.yaml` 作为稳定结构
2. 通过 `.env.nas` 切 `LLM_MODEL`
3. 通过 `.env.nas` 切 `OPENAI_API_URL`
4. 通过 `.env.nas` 切 `EMBEDDER_MODEL` 和 `EMBEDDER_OPENAI_API_URL`

### 10.3 OpenAI 兼容端点变量名是 `OPENAI_API_URL`

这点很容易和 `server/graph_service/` 混淆：

- `mcp_server` 的 YAML 里用的是 `OPENAI_API_URL`
- `server/graph_service` 的环境变量是 `OPENAI_BASE_URL`

两边不要混写。

如果你走的是当前验证通过的 NAS MCP 方案，默认推荐的 LLM 兼容端点应当是：

```bash
OPENAI_API_URL=https://open.bigmodel.cn/api/paas/v4/
```

而不是旧的：

```bash
https://open.bigmodel.cn/api/anthropic
```

原因很直接：

- `graphiti-mcp` 这条链路需要 OpenAI-compatible 的 `chat/completions`
- 当前已经验证通过的是 `paas/v4/`
- `anthropic` 端点适合 Claude 兼容调用，不适合作为这条 MCP LLM 链路的默认推荐

### 10.4 并发不是越高越好

`SEMAPHORE_LIMIT` 太高时，最常见的问题不是 CPU，而是 LLM 侧的 `429`。

仓库代码给出的经验值：

- OpenAI 低配额度：`1-2`
- 中等额度：`5-10`
- 较高额度：`10-15`
- 本地 Ollama：`1-5`

### 10.5 这次实测说明瓶颈主要在 LLM，不在 embedder

当前已经验证过的 MCP 组合是：

- LLM：`glm-5`
- LLM endpoint：`https://open.bigmodel.cn/api/paas/v4/`
- Embedder：`qwen3-embedding:0.6b`
- Embedder endpoint：`http://192.168.123.74:11434/v1`

实测现象是：

- `qwen3-embedding:0.6b` 的单次 embedding 往往在毫秒到一秒级
- 真正的大头在 Graphiti 的几个 LLM prompt
  - `extract_nodes.extract_text`
  - `extract_edges.edge`
  - `extract_nodes.extract_summaries_batch`

所以如果你想进一步提速：

- 优先换 LLM 端点或模型
- 不要先怀疑 embedder

---

## 十一、如果你不走 MCP：直接用 Python API

如果你只是想快速验证 `graphiti_core`，下面这个例子更贴近仓库里 `examples/quickstart/quickstart_neo4j.py` 的真实调用方式。

```python
import asyncio
import os
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType


async def main():
    graphiti = Graphiti(
        os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
        os.environ.get('NEO4J_USER', 'neo4j'),
        os.environ.get('NEO4J_PASSWORD', 'password'),
    )

    await graphiti.build_indices_and_constraints()

    await graphiti.add_episode(
        name='auth_design_note',
        episode_body='JWT scope validation is handled in middleware, not in the controller.',
        source_description='engineering note',
        reference_time=datetime.now(timezone.utc),
        source=EpisodeType.text,
        group_id='graphiti_repo',
    )

    results = await graphiti.search(
        query='Where is JWT scope validation implemented?',
        group_ids=['graphiti_repo'],
        num_results=5,
    )

    for edge in results:
        print(edge.fact)

    await graphiti.close()


asyncio.run(main())
```

这里有两个和旧文档很容易写错的点：

- `search()` 参数是 `num_results`，不是 `limit`
- `add_episode()` 需要显式传 `source_description` 和 `reference_time`

补充说明：

- `add_episode_bulk()` 适合批量导入，但按 `graphiti_core/graphiti.py` 的注释，它**不会**执行逐条 `add_episode()` 那套边失效与日期提取流程
- 对“持续对话式记忆”或“逐步积累工程事实”的场景，顺序调用 `add_episode()` 更稳

---

## 十二、排障指南

### 12.1 搜不到刚写入的内容

先确认不是队列还没处理完。

检查方法：

```bash
docker compose -f mcp_server/docker/docker-compose-neo4j.yml logs -f graphiti-mcp
```

`add_memory` 走的是后台队列，同一个 `group_id` 会串行消费。

### 12.2 Neo4j 能开网页，但 Graphiti 连不上

最常见原因是 `NEO4J_URI` 写错了宿主机名。

原则：

- 容器访问同 compose 里的 Neo4j：`bolt://neo4j:7687`
- 本机直连 Neo4j：`bolt://localhost:7687`
- 一个容器访问宿主机 Neo4j：不要写容器内的 `localhost`

### 12.3 `group_id` 报错

如果你看到类似“must contain only alphanumeric characters, dashes, or underscores”，说明 `group_id` 包含非法字符。

把这些字符去掉：

- 空格
- `/`
- `:`
- `@`
- 中文

### 12.4 HTTP 可访问，但 MCP 客户端连不上

先确认你填的是：

```text
http://localhost:8000/mcp/
```

不要填成：

- `http://localhost:8000/health`
- `http://localhost:8000`
- `http://localhost:8000/docs`

---

## 十三、下一步建议

跑通之后，建议按这个顺序继续：

1. 先为一个真实仓库建立基础工程记忆
2. 让助手先用 Graphiti 检索，再做代码修改
3. 把 code review 结论、部署坑、故障复盘写回 Graphiti
4. 再决定是否需要额外部署 `server/graph_service/` 供其他系统调用

如果你接下来要上服务器、NAS 或容器环境部署 Neo4j + Graphiti，请继续看：

- `DOCKER_DEPLOYMENT_GUIDE.md`
