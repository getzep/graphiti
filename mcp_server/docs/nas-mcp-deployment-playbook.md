# Graphiti MCP on Synology NAS Playbook

> 这份文档以当前已经验证可用的 NAS 部署为准。
> 目标不是重新设计部署方式，而是保留一个可回退的真实基线，并说明后续如何验证 `graphiti-mcp` 继续保持 `GLM-5 + Ollama embedder` 的组合。

---

## 1. 当前目标架构

当前 NAS 上有两套 Graphiti 相关服务：

- `graphiti-api`
  - 面向其他业务项目
  - REST API
  - 外部端口：`8010`
- `graphiti-mcp`
  - 面向 Claude / Cursor / Codex 等 MCP 客户端
  - Streamable HTTP MCP
  - 外部端口：`8011`
- `neo4j-graph-db`
  - 两者共用
  - Bolt：`7687`
  - Browser：`7474`

其中 `graphiti-mcp` 的运行策略是：

- 主 `llm` 保持 `GLM-5`
- `embedder` 单独走本地 Ollama
- 本地 embeddings 模型为 `qwen3-embedding:0.6b`

---

## 2. 已确认的回退基线

后续任何修改前，都应先把下面这套配置视为回退点。

当前已确认生效的关键事实：

- 容器名：`graphiti-mcp`
- 镜像名：`docker.1ms.run/zepai/knowledge-graph-mcp:latest`
- 启动命令：`uv run --no-sync main.py`
- 挂载配置文件：`/app/mcp/config/config.yaml`
- 实际来源配置：`config/config-docker-neo4j-external.yaml`
- LLM：
  - provider: `openai`
  - model: `glm-5`
  - api_url: `https://open.bigmodel.cn/api/paas/v4/`
- Embedder：
  - provider: `openai`
  - model: `qwen3-embedding:0.6b`
  - api_url: `http://192.168.123.74:11434/v1`
  - api_key: `ollama`
  - dimensions: `1024`
- Database：
  - provider: `neo4j`
  - uri: `bolt://192.168.123.104:7687`
  - database: `neo4j`
- Group ID：`nas_mcp`

如果后续修改导致服务异常，优先回退到这里，而不是回退到更早的“示例镜像名”或“默认 embedding 配置”。

---

## 3. 当前实际生效的 Compose 结构

群晖 Container Manager 或 `docker compose` 应使用与下面等价的配置。

```yaml
services:
  graphiti-mcp:
    image: docker.1ms.run/zepai/knowledge-graph-mcp:latest
    container_name: graphiti-mcp
    restart: always
    build:
      context: /volume5/docker5/graphiti/mcp_server
      dockerfile: docker/Dockerfile.standalone
    environment:
      - CONFIG_PATH=/app/mcp/config/config.yaml
      - MCP_ALLOWED_HOSTS=0.0.0.0:*,localhost:*,127.0.0.1:*,192.168.123.104:*
      - NEO4J_URI=bolt://192.168.123.104:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=your_neo4j_password
      - NEO4J_DATABASE=neo4j
      - GRAPHITI_GROUP_ID=nas_mcp
      - SEMAPHORE_LIMIT=10
      - LLM_MODEL=glm-5
      - OPENAI_API_KEY=your_glm_api_key
      - OPENAI_API_URL=https://open.bigmodel.cn/api/paas/v4/
      - EMBEDDER_PROVIDER=openai
      - EMBEDDER_OPENAI_API_KEY=ollama
      - EMBEDDER_OPENAI_API_URL=http://192.168.123.74:11434/v1
      - EMBEDDER_MODEL=qwen3-embedding:0.6b
      - EMBEDDER_DIMENSIONS=1024
    volumes:
      - /volume5/docker5/graphiti/mcp_server/config/config-docker-neo4j-external.yaml:/app/mcp/config/config.yaml:ro
    ports:
      - "8011:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://127.0.0.1:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 40s
    command: ["uv", "run", "--no-sync", "main.py"]
```

关键点：

- `image` 保持 `docker.1ms.run/zepai/knowledge-graph-mcp:latest`
  - 这是当前群晖项目实际使用并构建出来的镜像名
  - 不要再把旧的 `graphiti-mcp-local:hostfix` 当作主推荐
- `env_file: ../.env.nas`
  - 以后切换 `OPENAI_API_URL`、`LLM_MODEL`、`EMBEDDER_OPENAI_API_URL` 时，优先改 `.env.nas`
  - 不要再直接改 compose YAML
- `build.context` 使用 NAS 本机路径
  - 说明源码是先同步到 NAS，再在 NAS 上构建
- `command: ["uv", "run", "--no-sync", "main.py"]`
  - 避免容器启动时再次同步依赖
- `EMBEDDER_*` 与 `OPENAI_*` 分离
  - LLM 继续走 `GLM-5`
  - embedder 单独走本地 Ollama

---

## 4. 挂载的配置文件

当前挂载进去的配置文件是：

- [config-docker-neo4j-external.yaml](/opt/claude/graphiti/mcp_server/config/config-docker-neo4j-external.yaml)

这份 YAML 的职责是：

- 固定 `server.transport=http`
- 固定 `database.provider=neo4j`
- 固定 `llm.provider=openai`
- 固定 `embedder.provider=openai`
- 通过环境变量注入：
  - `LLM_MODEL`
  - `OPENAI_API_URL`
  - `EMBEDDER_OPENAI_API_URL`
  - `EMBEDDER_MODEL`
  - `EMBEDDER_DIMENSIONS`

重要说明：

- 不要为了“让 embedder 跑起来”去把 `llm` 一并切到本地 Ollama 生成模型
- 当前这套部署的目标就是：
  - `llm = glm-5`
  - `embedder = qwen3-embedding:0.6b @ Ollama`

---

## 5. 为什么镜像名不是之前提过的本地标签

之前曾经讨论过本地镜像名，但当前实际构建/运行结果不是它。

如果群晖构建日志里出现：

```text
naming to docker.1ms.run/zepai/knowledge-graph-mcp:latest
```

那就以这个镜像名为准。

这通常意味着：

- 当前参与构建的项目配置就是这个 `image:` 值
- 或者群晖 Container Manager 仍然沿用它保存的项目定义

这里不要再猜“理论上应该叫什么”，直接以当前运行配置为准。

---

## 6. 从 WSL2 同步文件到 NAS

当前开发环境在 Windows 的 WSL2 中，而容器运行在 NAS。

推荐做法仍然是：

1. 在 WSL2 中修改源码
2. 同步到 NAS
3. 在 NAS 本机 build / restart

推荐同步命令：

```bash
cd /opt/claude/graphiti

rsync -rv --relative \
  graphiti_core/ \
  mcp_server/main.py \
  mcp_server/benchmark_mcp.py \
  mcp_server/pyproject.toml \
  mcp_server/uv.lock \
  mcp_server/docker/Dockerfile.standalone \
  mcp_server/docker/docker-compose-neo4j-external.yml \
  mcp_server/config/ \
  mcp_server/src/ \
  mcp_server/docs/ \
  john@192.168.123.104:/volume5/docker5/graphiti/
```

注意：

- 不要用 `rsync -a`
- `rsync -rv --relative` 足够
- 如果权限报错但文件内容已同步，优先检查内容而不是纠结元数据
- standalone 镜像现在依赖 repo root build context；如果漏掉 `graphiti_core/`，容器内会回退到 PyPI 发布版逻辑
- 如果漏掉 `benchmark_mcp.py`，NAS 上的 probe 将看不到 `INGEST_STATUS[...]` 调试输出

---

## 7. 启动时应该看到什么

当前版本的 `mcp_server` 启动时应明确打印出运行摘要，并在启动阶段做一次 embedder 连通性检查。

重点日志应包括：

- `LLM: openai / glm-5`
- `base_url: https://open.bigmodel.cn/api/paas/v4/`
- `Embedder: openai / qwen3-embedding:0.6b`
- `base_url: http://192.168.123.74:11434/v1`
- `dimensions: 1024`
- `Embedder connectivity check passed`

如果这里没有打印出上述组合，就说明运行时配置没有按预期生效。

---

## 8. 部署后验证命令

### 8.1 查看容器健康状态

```bash
curl http://127.0.0.1:8011/health
```

期望：

```json
{"status":"healthy","service":"graphiti-mcp"}
```

### 8.2 查看 `graphiti-mcp` 启动日志

```bash
docker logs graphiti-mcp --tail 100
```

重点确认：

- `glm-5` 仍然是 LLM
- `qwen3-embedding:0.6b` 仍然是 embedder
- Ollama 端点是 `http://192.168.123.74:11434/v1`
- embedder 自检通过

### 8.3 查看 Ollama 模型是否已加载

```bash
curl http://192.168.123.74:11434/api/ps
```

期望返回里包含：

- `qwen3-embedding:0.6b`

### 8.4 直接测试 Ollama embedding

```bash
curl http://192.168.123.74:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding:0.6b","input":"hello from nas"}'
```

### 8.5 查看当前容器实际使用的镜像名

```bash
docker inspect graphiti-mcp --format '{{.Config.Image}}'
```

期望：

```text
docker.1ms.run/zepai/knowledge-graph-mcp:latest
```

---

## 9. 回退策略

如果某次修改后服务异常，回退时按下面顺序做：

1. 恢复当前这份文档中记录的 compose 结构
2. 恢复挂载配置到 `config-docker-neo4j-external.yaml`
3. 恢复下面这组关键环境变量：
   - `LLM_MODEL=glm-5`
   - `OPENAI_API_URL=https://open.bigmodel.cn/api/paas/v4/`
   - `EMBEDDER_OPENAI_API_URL=http://192.168.123.74:11434/v1`
   - `EMBEDDER_MODEL=qwen3-embedding:0.6b`
   - `EMBEDDER_DIMENSIONS=1024`
4. 确认 `command` 仍是 `uv run --no-sync main.py`
5. 重新查看启动日志，确认运行摘要恢复到基线

不要把“回退”理解成回到更早的示例镜像名或默认 embedding 配置。

---

## 10. 经验结论

以后再做同类部署时，优先记住这几条：

- 真实生效配置比历史聊天记录更重要
- `llm` 和 `embedder` 可以而且应该分开配置
- `GLM-5 + 本地 Ollama embedding` 是当前验证通过的组合
- 只看 `/health` 不够，还要看启动摘要和 embedder 自检
- 如果 Ollama 端点不通，先查网络和防火墙，不要先怀疑 Graphiti 主逻辑
