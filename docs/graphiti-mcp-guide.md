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

> 注意：如果你在旧英文 README、旧博客或旧示例里看到 `add_episode`、`search_facts`，
> 把它们视为历史名称。当前 `graphiti-mcp` 实际暴露的是 `add_memory` 和
> `search_memory_facts`。

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

### 2.5 当前工具名与关键参数

当前最常用的工具签名可以按下面理解：

- `add_memory(name, episode_body, group_id?, source?, source_description?, uuid?)`
- `search_nodes(query, group_ids?, max_nodes?, entity_types?)`
- `search_memory_facts(query, group_ids?, max_facts?, center_node_uuid?)`
- `get_episodes(group_ids?, max_episodes?)`
- `clear_graph(group_ids?)`

这里有两个容易混淆的点：

- 检索类工具大多使用 `group_ids`，是列表，不是单个 `group_id`
- `get_episodes` 当前参数是 `max_episodes`，不是旧资料里常见的 `last_n`

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

### 10.5 直接用基准脚本做对比

仓库里现在提供了一个可复用脚本：

- `mcp_server/benchmark_mcp.py`

它会自动执行：

1. MCP 协议 `initialize` 握手
2. `add_memory`
3. 轮询 `search_nodes`
4. 轮询 `search_memory_facts`
5. 输出 `success_at_seconds`
6. 默认清理测试 `group_id`

典型用法：

```bash
cd mcp_server
python benchmark_mcp.py \
  --url http://127.0.0.1:8011/mcp \
  --group-id-prefix perfprobe \
  --sleep-seconds 20 \
  --max-attempts 18
```

如果你要保留测试数据方便排查：

```bash
python benchmark_mcp.py \
  --url http://127.0.0.1:8011/mcp \
  --keep-data
```

输出里最值得看的是：

- `group_id`
- `success_at_seconds`
- `nodes_found`
- `facts_found`

所以以后切换 `.env.nas` 中的：

- `LLM_MODEL`
- `OPENAI_API_URL`

后，直接重复跑这个脚本就能做 A/B 对比。

## 11. 用 `examples/quickstart` 理解 Graphiti 本体

虽然本文主要讲 `graphiti-mcp`，但如果你想真正理解它在底层封装了什么，仓库里最好的起点仍然是：

- `examples/quickstart/README.md`
- `examples/quickstart/quickstart_neo4j.py`
- `examples/quickstart/quickstart_falkordb.py`
- `examples/quickstart/quickstart_neptune.py`

这组示例直接调用的是 `graphiti_core`，没有经过 MCP 包装，所以它非常适合用来建立“Graphiti 本体到底怎么工作”的心智模型。

### 11.1 为什么这个示例值得先看

它基本把 Graphiti 的核心主线都串起来了：

1. 连接图数据库
2. 初始化索引和约束
3. 写入文本和 JSON episode
4. 搜索事实关系
5. 用图距离重排结果
6. 用预定义搜索配方搜索节点

把这条主线看懂之后，再回头看 `graphiti-mcp`，你会更容易理解：

- MCP 的 `add_memory` 本质上就是在服务端调用 `graphiti.add_episode(...)`
- MCP 的 `search_memory_facts` 本质上就是 Graphiti 的边搜索
- MCP 的 `search_nodes` 本质上就是 Graphiti 的节点搜索配方封装

### 11.2 Quickstart 和 MCP 的对应关系

如果你把 quickstart 当成“底层 API 版”，那和 MCP 的关系可以这样对照：

| Quickstart 直接 API | `graphiti-mcp` 对应能力 | 说明 |
|---|---|---|
| `Graphiti(...)` / `FalkorDriver(...)` / `NeptuneDriver(...)` | 服务启动时读取 YAML + `.env` 建 client | MCP 把连接配置搬到了服务端 |
| `graphiti.add_episode(...)` | `add_memory` | MCP 用工具调用替代直接 Python 调用 |
| `graphiti.search(...)` | `search_memory_facts` | 都是在找事实关系（边） |
| `graphiti.search_()` + `NODE_HYBRID_SEARCH_RRF` | `search_nodes` | MCP 已经帮你选好了一个常用节点检索配方 |
| `await graphiti.close()` | 服务端生命周期管理 | MCP 客户端不需要自己关连接 |

### 11.3 安装和配置要点

quickstart README 里给出的最小依赖是：

- `graphiti-core`
- `python-dotenv`

最小环境变量是：

- `OPENAI_API_KEY`

按不同后端，还要再配：

- Neo4j
  - `NEO4J_URI`
  - `NEO4J_USER`
  - `NEO4J_PASSWORD`
- FalkorDB
  - 当前脚本实际读取的是 `FALKORDB_HOST`、`FALKORDB_PORT`
  - 可选 `FALKORDB_USERNAME`、`FALKORDB_PASSWORD`
- Neptune
  - `NEPTUNE_HOST`
  - `NEPTUNE_PORT`
  - `AOSS_HOST`

有两个细节值得注意：

- `examples/quickstart/README.md` 里写了 `FALKORDB_URI`，但当前 `quickstart_falkordb.py` 实际是按 host/port 读取环境变量
- quickstart README 写的是 Python 3.9+，但当前仓库根 `pyproject.toml` 和 `mcp_server` 都按 Python 3.10+ 运行更稳妥

### 11.4 连接到 Neo4j 数据库

quickstart 的 Neo4j 接法非常直接：

```python
graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
```

也就是说：

- Graphiti 默认可以直接接 Neo4j
- 你只要把 URI、用户名、密码准备好，就能开始写入和检索

如果你用的是自定义 Neo4j database，而不是默认库，quickstart README 还特别提醒了一个常见坑：

- `Graph not found: default_db`

遇到这种情况时，应该显式构造 `Neo4jDriver(..., database="your_db_name")`，而不是只传 URI。

对于 `graphiti-mcp`，同样的连接信息不再写进代码，而是放在：

- `config.yaml`
- `.env`
- 或命令行参数

### 11.5 初始化 Graphiti 索引和约束

从概念上说，这是第一次使用某个图数据库时应当做的初始化动作：

```python
await graphiti.build_indices_and_constraints()
```

它的作用是：

- 建立必要的索引
- 建立必要的约束
- 让后续写入和检索更稳定、更快

这里要特别说明当前示例的实际情况：

- `quickstart_neptune.py` 明确调用了 `await graphiti.build_indices_and_constraints()`
- `quickstart_neo4j.py` 和 `quickstart_falkordb.py` 的注释和 README 都强调这一步重要
- 但这两个脚本当前本体没有显式调用这行代码

所以如果你是第一次在一套新的 Neo4j / FalkorDB 上跑 Graphiti，建议把这一步当成“应该显式执行一次”的初始化步骤。

而在 `graphiti-mcp` 里，这一步通常已经由服务启动流程自动处理了，所以 MCP 客户端不需要自己再调用单独工具。

### 11.6 向图谱中添加 Episodes

quickstart 里最关键的写入示例，是同时写入：

- 纯文本 episode
- 结构化 JSON episode

典型形态是：

```python
await graphiti.add_episode(
    name='Freakonomics Radio 0',
    episode_body='...',
    source=EpisodeType.text,
    source_description='podcast transcript',
    reference_time=datetime.now(timezone.utc),
)
```

如果内容是 JSON，示例会先做一层：

```python
json.dumps(episode['content'])
```

这意味着：

- 底层 `graphiti_core` 仍然接收字符串形式的 episode body
- `EpisodeType.json` 只是告诉 Graphiti“这是结构化内容”
- Graphiti 会据此抽实体、关系和摘要

示例还说明了 `reference_time` 的意义：

- 它不只是“写入时间”
- 它还是事实生效时间的参考点
- 这也是 Graphiti 能做时间语义检索的基础

对应到 MCP：

- 你不会直接调用 `add_episode`
- 而是用 `add_memory`
- `source` 仍然是 `text` / `json` / `message`
- 但 `reference_time` 当前由服务端在入队处理时自动取当前 UTC 时间

### 11.7 使用混合搜索查找关系（边）

quickstart 演示关系搜索时，直接调用：

```python
results = await graphiti.search('Who was the California Attorney General?')
```

这个调用背后做的是混合搜索，核心包括：

- 语义相似度
- 关键词 / BM25
- 图结构相关性

返回结果的重点字段包括：

- `fact`
- `valid_at`
- `invalid_at`
- `source_node_uuid`
- `target_node_uuid`

所以它不是在返回“最像的文档块”，而是在返回“图谱中最相关的事实关系”。

这就是 `graphiti-mcp` 里 `search_memory_facts` 的底层原型。

### 11.8 使用图距离重排搜索结果

quickstart 的第二个关键动作，是先搜一次，再把首条结果的源节点拿来做 anchor：

```python
center_node_uuid = results[0].source_node_uuid
reranked_results = await graphiti.search(
    'Who was the California Attorney General?',
    center_node_uuid=center_node_uuid,
)
```

它的思路是：

1. 先用普通混合搜索找到一个足够靠谱的相关事实
2. 再用这条事实关联的节点作为“中心节点”
3. 让第二轮结果向图上更近的事实倾斜

这对下面这些问题很有价值：

- 先找到正确人物，再找该人物的其他事实
- 先找到正确服务，再找与它强相关的依赖或配置
- 先找到正确文档实体，再找同一主题链路上的其他节点和关系

而 `graphiti-mcp` 的 `search_memory_facts` 也保留了 `center_node_uuid` 参数，所以这套思路可以直接迁移到 MCP 客户端。

### 11.9 使用预定义配方搜索节点

quickstart 的第六步演示了“不是搜边，而是直接搜节点”：

```python
node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
node_search_config.limit = 5

node_search_results = await graphiti._search(
    query='California Governor',
    config=node_search_config,
)
```

这段代码有三个重点：

1. 它用的是 `NODE_HYBRID_SEARCH_RRF`
   - 这是内置的节点搜索配方
   - 你不用手写复杂搜索参数
2. 它会先复制 recipe 再改 `limit`
   - 避免直接修改共享的全局配置对象
3. 示例里调用的是 `_search()`
   - 但在当前 `graphiti_core` 中，`_search()` 已经是兼容包装
   - 新代码更建议直接用 `search_()`

对应到 MCP，这一步其实已经被封装好了：

- `search_nodes` 内部就是通过 `search_()` + `NODE_HYBRID_SEARCH_RRF` 来做节点检索

也就是说，MCP 客户端天然就拿到了 quickstart 推荐的那套“节点搜索最佳实践”。

### 11.10 这个示例如何帮助你理解 `graphiti-mcp`

把 quickstart 看懂后，再看 MCP，你会发现 `graphiti-mcp` 做的事情主要是三类封装：

- 把 Python API 变成 MCP tools
- 把数据库 / LLM / embedder 配置搬到服务端
- 把写入流程改造成异步队列，避免客户端阻塞等待

所以一个很实用的理解路径是：

1. 先跑通 quickstart
2. 确认你理解 episode、fact、node、`reference_time`、`center_node_uuid`
3. 再接 `graphiti-mcp`
4. 把 direct API 心智模型映射成 MCP tools

这样你在 Cursor、Claude、Codex 里调工具时，会更清楚每一步到底在底层触发了什么。

### 11.11 跑完 quickstart 后下一步看什么

quickstart README 里给的后续建议基本是对的，推荐顺序也可以照着走：

1. 改 episode 内容，观察抽取结果如何变化
2. 改搜索问题，看看边搜索怎么变化
3. 改 `center_node_uuid`，观察图距离重排差异
4. 继续看 `graphiti_core.search.search_config_recipes`
5. 再看 `examples/quickstart/dense_vs_normal_ingestion.py`

最后这个 `dense_vs_normal_ingestion.py` 很适合作为第二站，因为它会继续解释：

- 什么内容会被当成普通 prose
- 什么内容会触发 dense-content chunking
- Graphiti 在实体非常密集的 JSON / 报表场景下会怎么分块处理

## 12. 常见问题

### 12.1 `/health` 正常，但 tools 不可用

优先检查：

- URL 是否指向 `/mcp/`
- transport 是否一致

### 12.2 写入成功，但搜索不到

先查：

1. `get_status`
2. `get_episodes`
3. 容器日志

不要只盯着检索接口。

### 12.3 为什么 OpenAI 兼容端点看起来能调，但写入还是慢或失败

因为 Graphiti 不只是简单聊天：

- 它会要求结构化 JSON
- 会有多轮 prompt
- 兼容端点如果返回 fenced JSON、`answer` 包装、列表包装，都会增加重试成本

### 12.4 为什么我改了端点却没生效

优先检查：

- MCP 侧是不是改了 `OPENAI_API_URL`
- 而不是误改成 API 侧的 `OPENAI_BASE_URL`
- compose 是否真的重新 build

### 12.5 如果我要给业务系统接入，不是给 AI 客户端接入

那就不要优先用 MCP，直接看 [graphiti-api-guide.md](./graphiti-api-guide.md)。
