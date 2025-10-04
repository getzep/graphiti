# Graphiti MCP Server 重启说明

## 问题现状

✓ **Bug已修复** - 代码已经更新到 `mcp_server/graphiti_mcp_server.py`
✗ **未生效** - MCP服务器进程还在运行旧代码
⚠️ **需要重启** - 必须重启MCP服务器才能加载修复

## 如何重启MCP服务器

### 方法1: 重启Claude Code（推荐）

1. **完全退出Claude Code**
   - 关闭所有Claude Code窗口
   - 确保进程完全退出（检查任务管理器）

2. **重新启动Claude Code**
   - MCP服务器会自动启动并加载新代码

3. **验证修复生效**
   ```python
   # 测试添加数据
   mcp__graphiti__add_memory(
       name="重启后测试",
       episode_body="测试MCP服务器重启后数据是否能正确存储到Neo4j",
       source="text"
   )

   # 等待5-10秒后搜索
   mcp__graphiti__search_memory_nodes(query="重启后测试")
   ```

### 方法2: 手动重启MCP进程

1. **找到graphiti MCP进程**
   ```bash
   # Windows
   tasklist | findstr python

   # 查找运行 graphiti_mcp_server.py 的进程
   ```

2. **结束进程**
   ```bash
   # Windows - 找到对应的PID后
   taskkill /PID <进程号> /F
   ```

3. **Claude Code会自动重启MCP服务器**

### 方法3: 检查MCP配置并手动启动

MCP配置通常在以下位置之一：
- `%APPDATA%\Claude\claude_desktop_config.json`
- `%USERPROFILE%\.config\claude\mcp.json`
- `.codex/config.toml` (Codex)

找到graphiti的配置，记下启动命令，然后手动重启。

## 如何验证修复已生效

### 步骤1: 检查日志

重启后，在添加episode时，应该能在服务器日志中看到：
```
Starting episode queue worker for group_id: <your_group_id>
```

### 步骤2: 添加测试数据

```python
mcp__graphiti__add_memory(
    name="Bug修复验证",
    episode_body="这是验证bug修复的测试数据。如果能搜索到这条数据，说明修复成功。",
    source="text",
    source_description="verification test"
)
```

### 步骤3: 等待处理

等待5-10秒，让后台worker处理episode

### 步骤4: 搜索验证

```python
mcp__graphiti__search_memory_nodes(query="bug修复验证")
```

**预期结果**: 应该能找到刚才添加的数据

### 步骤5: 检查Neo4j数据库

```bash
curl -X POST http://localhost:7474/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic bmVvNGo6Z3JhcGhpdGkxMjMh" \
  -d '{"statements":[{"statement":"MATCH (n) RETURN count(n) as count"}]}'
```

**预期结果**: count应该大于0

## 修复内容摘要

修复了3处代码（所有修复都在 `graphiti_mcp_server.py`）：

1. **第802行** - 添加任务存储字典
   ```python
   queue_tasks: dict[str, asyncio.Task] = {}
   ```

2. **第903行** - 更新全局变量声明
   ```python
   global graphiti_client, episode_queues, queue_workers, queue_tasks
   ```

3. **第964-971行** - 修复任务创建逻辑
   ```python
   if not queue_workers.get(group_id_str, False):
       queue_workers[group_id_str] = True  # 先设置状态，防止竞态条件
       task = asyncio.create_task(process_episode_queue(group_id_str))
       queue_tasks[group_id_str] = task  # 存储引用，防止被垃圾回收
       task.add_done_callback(...)  # 添加完成回调
   ```

## 问题诊断

如果重启后还是不工作，检查：

1. **代码是否真的重新加载了**
   - 在graphiti_mcp_server.py第969行添加临时打印
   - 看是否执行新代码

2. **Neo4j是否正常运行**
   ```bash
   netstat -ano | findstr 7687
   ```

3. **API密钥是否配置**
   - 检查 GOOGLE_API_KEY 或 OPENAI_API_KEY
   - 没有API密钥，LLM操作会失败

4. **查看错误日志**
   - stdio模式: 日志在stderr
   - 检查是否有异常信息

## 需要帮助？

如果重启后问题依然存在，请：
1. 提供服务器日志（特别是启动时的日志）
2. 运行测试脚本：`python mcp_server/test_episode_queue_fix.py`
3. 检查是否有其他错误信息

## 相关文档

- **详细bug分析**: `BUGFIX_REPORT.md`
- **完整代码审查**: `COMPREHENSIVE_REVIEW.md`
- **测试脚本**: `mcp_server/test_episode_queue_fix.py`

---

**重要提示**: 修复代码已经在文件中，只需要重启MCP服务器即可生效！
