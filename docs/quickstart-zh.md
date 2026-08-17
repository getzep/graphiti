# Vaka Wiki（Graphiti 底座）中文快速开始

这套本地编排用于复现当前工作区代码，而不是安装 PyPI 上的旧版
graphiti-core。默认启动 Neo4j、Vaka Wiki 控制面 REST 服务和 Graphiti MCP
服务，支持本地文件、飞书、MeeGo 数据源及增量同步。

## 架构与端口

    浏览器
      │  http://localhost:8000
      ▼
    Vaka 知识中心 ── SQLite Wiki/计划/构建/发布状态 + 上传文件
      │
      ├── Graphiti Core ───────────────┐
      │                                ▼
    Working Agent ── MCP /wiki/{wiki_id}/mcp ── Graphiti Core ── Neo4j
                         :8001                    :7687

- Vaka 知识中心提供 Copilot 对话创建、计划查看、文件库、离线任务、实体目录、
  图谱、检索和 MCP 管理。
- REST 与 MCP 都从当前 checkout 构建，并连接同一个 Neo4j。
- graphiti_data 保存 SQLite 增量状态和上传文件；neo4j_data 保存图数据。
- 每个 Wiki 自动生成一个只读 MCP URL；Agent 只会读取该 Wiki 当前已发布的版本。

当前 Compose 默认只把端口绑定到 127.0.0.1，避免把无认证的管理接口直接
暴露到局域网或公网。

## 1. 准备配置

需要 Docker Engine 与 Docker Compose v2。在仓库根目录执行：

    cp .env.example .env

本指南使用独立的 docker-compose.studio.yml。根 docker-compose.yml 保留原有
Neo4j/FalkorDB 入口以兼容既有工作流，不要把两套编排同时启动。

至少修改以下三项：

    ARK_API_KEY=替换为新建或已轮换的方舟密钥
    ARK_CHAT_MODEL=ep-20260805000513-69bnd
    NEO4J_PASSWORD=替换为强密码

ARK_CHAT_MODEL 应填写可调用 Doubao Seed Chat Completions 的 Endpoint ID，
不要填写 embedding endpoint。ARK_BASE_URL 默认使用北京地域的
OpenAI-compatible API 地址。

官方参考：[Ark ChatCompletions API](https://api.volcengine.com/api-docs/view?action=ChatCompletions&serviceCode=ark&version=2024-01-01)。

不要提交 .env。密钥如果曾出现在聊天、日志或截图中，应先在控制台撤销并
重新生成；不要继续使用已暴露的值。本仓库的示例文件不保存真实密钥。

## 2. 启动与检查

先检查最终配置，再构建启动：

    docker compose -f docker-compose.studio.yml config --quiet
    docker compose -f docker-compose.studio.yml up --build -d
    docker compose -f docker-compose.studio.yml ps

常用入口：

- Vaka 知识中心：http://localhost:8000/
- OpenAPI：http://localhost:8000/docs
- REST 健康检查：http://localhost:8000/healthcheck
- Neo4j Browser：http://localhost:7474/
- MCP 健康检查：http://localhost:8001/health
- MCP Streamable HTTP：http://localhost:8001/mcp/

查看日志：

    docker compose -f docker-compose.studio.yml logs -f graph mcp

停止服务不会删除数据：

    docker compose -f docker-compose.studio.yml down

在上述 down 命令后增加 -v 会删除 Neo4j、同步状态和上传文件卷，属于不可恢复
的清理操作；执行前先备份。

## 3. Wiki、Candidate 与 Published

所有 Wiki 复用同一个 Neo4j database：

    NEO4J_DATABASE=neo4j

控制面为每次 Wiki 构建自动分配新的 Graphiti `group_id`。数据源只写入
Candidate namespace；构建成功后，系统自动原子切换 Wiki 的
`published_group_id`。搜索、图谱页和 Wiki MCP 始终解析这个发布指针，不会读到
构建中的部分数据。用户不需要、也不能在 Studio 中手工管理 `group_id`。

最小操作流程：

1. 在「我的 Wiki」右侧 Vaka Copilot 输入“帮我创建一个新的 Wiki：xxx”；
2. 按对话补充建库目标和数据范围，确认项目 Wiki 计划；
3. 在「文件库」添加一个或多个本地、飞书或 MeeGo 数据源；
4. 上传或关联数据后，系统自动提交离线构建；
5. 构建成功后自动发布，在 Wiki 左侧浏览实体目录或图谱；
6. 点击左下角 MCP 入口，复制当前 Wiki 的 MCP URL 给 Agent。

## 4. 数据源与增量同步

### 本地文件

在当前 Wiki 的「文件库」中创建“本地文件”数据源并上传文件后，系统会自动开始构建。当前支持 PDF、txt、Markdown、
CSV、TSV、JSON、JSONL、YAML、XML、HTML、日志和 DOCX。默认单文件上限为
25 MiB，可用 MAX_UPLOAD_BYTES 调整。

PDF 使用文档已有的文本层进行提取，适合可以选中文字或复制文字的 PDF。纯图片
扫描件暂不执行 OCR；这类文件没有可提取文本时，同步任务会给出错误提示，不会把
空内容写入图谱。请先用可信的 OCR 工具为文件生成文本层，再重新上传。对于包含
敏感信息的企业文档，应优先使用组织批准的本地或私有化 OCR 服务，避免将文件发送
到未经授权的第三方。

上传内容保存在 graphiti_data 卷的 /app/data/uploads。服务按来源、外部 ID 和
内容哈希记录状态：

- 内容未变化时跳过抽取；
- 内容变化时追加新 episode，并关联上一个版本；
- 本地或飞书条目消失时写入 tombstone，同步状态会保留，历史图事实不会直接删除；
- 同一 group_id 的同步串行执行，避免并发写入冲突。

“增量”表示重复同步只处理变化项，不代表实时订阅。当前 Studio 提供手动同步
与全量对账；需要定时任务或 webhook 时，应在受认证的内部调度层调用同步接口。

### 飞书

用户侧只有个人 OAuth 授权，不需要填写 App ID、App Secret、token、folder token，
也没有额外的角色配置步骤：

1. 在 Vaka 点击“添加数据 → 飞书 → 连接飞书”；
2. 在飞书官方页面使用自己的账号确认授权；
3. 回到 Vaka 浏览自己可见的空间，选择整个空间、文件夹或若干文档；
4. 添加数据并构建 Wiki。

OAuth 只代表当前用户，不会绕过飞书文档 ACL。Vaka 只会读取该用户原本有权限访问的
内容。

Vaka 使用标准 OAuth Authorization Code + PKCE。部署 Vaka 时需要注册一个网页应用，
登记回调地址并把 Client 凭据仅注入服务端；这是服务部署配置，终端用户不参与：

    http://localhost:8000/api/oauth/feishu/callback

    FEISHU_APP_ID=Vaka OAuth Client ID
    FEISHU_APP_SECRET=Vaka OAuth Client Secret
    OAUTH_PUBLIC_BASE_URL=http://localhost:8000

用户访问令牌与 OAuth 状态只在后端处理，并使用 AES-256-GCM 加密保存在 SQLite；
前端只收到连接 ID 和账号显示名。刷新令牌轮换也在后端原子更新，不进入浏览器存储。
本地 Compose 未显式配置 `OAUTH_TOKEN_ENCRYPTION_KEY` 时，会在持久卷中生成一个
仅供本实例使用的密钥；生产环境应由 secret manager 注入固定的 32 字节密钥。

官方参考：[获取 OAuth 授权码](https://open.feishu.cn/document/authentication-management/access-token/obtain-oauth-code)、
[获取 user_access_token v3](https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/authentication-management/access-token/get-user-access-token-v3)、
[刷新 user_access_token v3](https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/authentication-management/access-token/refresh-user-access-token-v3)、
[获取文件夹中的文件清单](https://open.feishu.cn/document/server-docs/docs/drive-v1/folder/list)、
[Docx 接入指南](https://open.feishu.cn/document/server-docs/docs/docs/docx-v1/guide?lang=zh-CN)、
[云空间权限常见问题](https://open.feishu.cn/document/server-docs/docs/drive-v1/faq?lang=zh-CN)。

OAuth 只代表当前授权用户，不会绕过飞书文档 ACL。建议：

- 使用专用、最小权限的内部应用；
- 只同步该用户本来有权读取的目录或文档；
- 不把用户 token、App Secret 或文件正文写入数据源名称和日志；
- 不同租户或安全域使用独立部署，至少使用独立数据库边界。

### MeeGo

新建数据源使用 MeeGo 官方的新 OAuth + MCP 数据面，不需要插件 ID、插件密钥、
User Key，也不需要用户复制 project_key。默认配置只有：

    MEEGO_HOST=meego.larkoffice.com
    OAUTH_PUBLIC_BASE_URL=http://localhost:8000

`MEEGO_HOST` 必须与用户实际 MeeGo URL 的域名一致。例如
`https://meego.larkoffice.com/ai_search_rec/...` 对应 `meego.larkoffice.com`。
不同域名是相互隔离的站点，OAuth 授权和视图数据不能复用。

使用步骤：

1. 在 Studio 点击“添加数据源 → MeeGo → 连接 MeeGo”；
2. 在 MeeGo 页面授权；
3. 回到 Studio，从当前账号可见范围中选择产品或项目；
4. 在“导入范围”中选择具体需求视图；视图较多时可按名称查找；
5. 点击“同步到文件库”。后端保存选择结果的项目 Key 和 View ID，并通过
   `get_view_detail` 精确读取该视图当前筛选出的需求，
   再补充每条需求的完整字段。视图筛选发生变化后，下一次同步会自动跟随。

终端用户不需要复制 MeeGo 链接，也不需要理解或填写 project_key、view_id。

后端按 OAuth Discovery 动态注册 PKCE 客户端，随后以 Bearer token 调用 MeeGo
MCP，并先执行 `tools/list` 再选择服务端实际提供的视图/工作项工具。最终可见范围
仍由当前 MeeGo 用户权限决定。部分企业租户可能限制网页回调或动态注册；若授权页
拒绝当前回调地址，需要调整对应租户策略，而不是让用户粘贴 token。

旧版本中 `MEEGO_PLUGIN_*` / `MEEGO_USER_KEY` 只用于已有、未绑定
`connection_id` 的数据源兼容；Studio 不再提供创建这种数据源的 UI。

官方参考：[Meegle CLI OAuth 与 MCP 实现](https://github.com/larksuite/meegle-cli)、
[Meegle CLI 数据访问说明](https://github.com/larksuite/meegle-cli/blob/main/skills/meegle/SKILL.md)。

## 5. Ark Chat 与 embedding

默认组合是：

- Ark Chat endpoint：执行实体、关系和事实抽取；
- local_hash：在本机生成确定性的 1024 维特征；
- lexical reranker：提供无需额外模型的本地重排。

local_hash 便于零依赖启动，也不会把正文发送给远程 embedding 服务，但它不是
语义 embedding，召回质量不适合作为生产基线。图谱抽取仍会把必要内容发送给
已配置的 Ark Chat endpoint。

生产环境应创建独立的文本 embedding endpoint，不能把 Doubao Seed Chat
endpoint 当成 embedding 模型。官方参数和模型要求见
[火山方舟文本向量化 API](https://www.volcengine.com/docs/82379/1302003?lang=zh)。
在根目录 .env 中同时配置：

    EMBEDDING_PROVIDER=openai
    EMBEDDING_API_KEY=替换为 embedding 凭证
    EMBEDDING_BASE_URL=替换为 OpenAI-compatible embedding API 地址
    EMBEDDING_MODEL_NAME=替换为 embedding Endpoint ID
    EMBEDDING_DIM=替换为该 endpoint 的实际维度

根 Compose 会把同一组值映射到 REST 和 MCP，确保共享 Neo4j 的两条写入路径
使用相同模型和维度。不要只修改其中一个服务。

切换 embedding 模型或维度不是无损热更新。旧向量与新向量不可混用，Neo4j
向量索引维度也必须一致。生产切换前应备份，并在干净的数据库实例中重建索引和
重新摄取数据；不要直接在已有图上改维度。

## 6. Working Agent 接入 Wiki MCP

支持 Streamable HTTP 的客户端可使用类似配置：

    {
      "mcpServers": {
        "vakaWiki": {
          "url": "http://localhost:8001/wiki/<wiki_id>/mcp"
        }
      }
    }

Wiki ID 和完整 URL 可直接从 Studio 复制。Wiki MCP 仅暴露 `item.search`、
`item.get` 和 `link.traverse` 三个只读工具。`/mcp/` 依然保留为开发管理端点，
包含写入和删除工具，不应交给 Wiki 消费方。

### MCP 安全边界

当前 MCP 没有内建的用户认证或租户授权。Wiki URL 会强制查询对应的
Published namespace，但 URL 隔离不等于身份认证；拿到 URL 的调用方可以读取该 Wiki。
`/mcp/` 管理端点还提供读、写及清理图数据的工具。

因此：

- 本机开发保持默认的 127.0.0.1 绑定；
- 不要直接把 8001 或 Neo4j 的 7474/7687 暴露到公网；
- 远程使用时，在端点前增加 TLS、强认证、授权、速率限制和审计代理；
- 为每个 Agent 配置最小工具权限，尤其限制 clear/delete 类操作；
- 不把 Feishu、MeeGo 或 Ark 凭证下发给 MCP 客户端；
- 对多租户场景使用真正的数据库或部署隔离，不能把 group_id 当作安全边界。

Studio REST API 当前同样不应被视为带认证的公网控制面。若需远程访问，应与 MCP
一起放在受保护的反向代理或内网网关后。

## 7. 生产化检查清单

- 所有曾暴露的 API key 已轮换，密钥由 secrets manager 注入。
- Neo4j、Studio 与 MCP 不直接监听公网，入口有认证、TLS 和审计。
- NEO4J_DATABASE 为 neo4j；Wiki namespace 由控制面自动生成。
- REST 与 MCP 使用同一 embedding endpoint 和维度。
- Feishu/MeeGo 应用只拥有目标数据的只读最小权限。
- 定期备份 neo4j_data 与 graphiti_data，并验证恢复流程。
- 对同步失败、MCP 队列、LLM 限流和数据库容量配置监控告警。
