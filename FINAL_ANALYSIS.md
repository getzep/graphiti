# Graphiti Storage Problem - Final Analysis

## 问题现状

尽管我们已经：
1. ✅ 修复了asyncio任务垃圾回收bug
2. ✅ 清除了Python缓存
3. ✅ 添加了调试输出
4. ✅ 验证了Neo4j连接正常
5. ✅ 确认MCP服务器已重启并连接

**数据仍然无法存储到Neo4j数据库**

## 测试结果

```
Episode加入队列: ✅ 成功 (position: 1)
Worker启动: ❓ 未知（看不到DEBUG输出）
数据库节点数: 0
Episode数: 0
```

## 根本原因推测

基于以下观察：

1. **Episode成功加入队列** - `add_memory`工具返回成功
2. **数据库始终为空** - 即使等待10秒后
3. **看不到DEBUG输出** - 我们添加的🔥DEBUG消息没有出现
4. **Neo4j有20+连接** - MCP服务器确实连接到了数据库

### 可能性1：Worker任务根本没有启动（最可能）

**证据：**
- 看不到"DEBUG: Creating worker task"消息
- 看不到"DEBUG: Worker STARTED"消息
- 数据库完全为空

**原因可能是：**
- asyncio事件循环问题
- MCP框架的异步上下文问题
- task创建在错误的事件循环中

**验证方法：**
```python
# 在add_memory函数中添加：
import sys
print(f"DEBUG: Current event loop: {asyncio.get_event_loop()}", file=sys.stderr, flush=True)
print(f"DEBUG: Queue workers status: {queue_workers}", file=sys.stderr, flush=True)
print(f"DEBUG: Queue tasks status: {queue_tasks}", file=sys.stderr, flush=True)
```

### 可能性2：Worker启动但立即崩溃

**证据：**
- Episode被加入队列
- 但没有任何处理发生

**原因可能是：**
- `process_episode_queue`函数中的异常
- `episode_queues[group_id].get()`失败
- 异常被捕获但日志没有输出到stderr

**验证方法：**
在worker的try-except中添加更详细的错误处理

### 可能性3：graphiti_client未正确初始化

**证据：**
- `add_memory`返回成功（说明MCP工具本身工作）
- 但`graphiti_client`可能为None

**原因可能是：**
- `initialize_graphiti()`失败但静默
- LLM client创建失败
- Embedder创建失败

**验证方法：**
```python
# 在add_memory开始添加：
if graphiti_client is None:
    raise ValueError("graphiti_client is not initialized!")
```

### 可能性4：MCP stdio传输问题

**证据：**
- 所有DEBUG输出都应该到stderr
- 但我们看不到任何DEBUG消息

**原因可能是：**
- MCP stdio模式不传输stderr
- stderr被重定向到其他地方
- print(..., file=sys.stderr)在MCP上下文中不工作

**验证方法：**
改用logger.error而不是print到stderr

## 建议的修复步骤

### 立即可行的修复

1. **改进错误检查**

在`add_memory`函数开始添加验证：

```python
@mcp.tool()
async def add_memory(...):
    global graphiti_client

    # CRITICAL: Verify graphiti_client is initialized
    if graphiti_client is None:
        return {
            "success": False,
            "error": "Graphiti client is not initialized. Server may have failed to start."
        }

    # CRITICAL: Verify event loop
    try:
        loop = asyncio.get_running_loop()
        logger.info(f"add_memory running in loop: {id(loop)}")
    except RuntimeError:
        logger.error("add_memory called outside of event loop!")
        return {
            "success": False,
            "error": "No event loop running"
        }

    # ... rest of code
```

2. **改用logger而不是print**

将所有`print(..., file=sys.stderr)`改为：

```python
logger.error(f"🔥🔥🔥 DEBUG: Worker STARTED for group_id={group_id} 🔥🔥🔥")
```

这样即使stderr不工作，至少可以通过其他方式查看日志。

3. **添加worker状态查询工具**

```python
@mcp.tool()
async def debug_worker_status() -> dict:
    """Get debug information about worker status"""
    return {
        "queue_workers": dict(queue_workers),
        "queue_tasks": {k: str(v) for k, v in queue_tasks.items()},
        "episode_queues_sizes": {k: v.qsize() for k, v in episode_queues.items()},
        "graphiti_client_initialized": graphiti_client is not None,
    }
```

### 长期解决方案

1. **使用日志文件**

在MCP server启动时配置文件日志：

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    filename='/c/workspace/graphiti/mcp_server/debug.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

2. **测试graphiti_core直接调用**

创建独立测试脚本，直接调用graphiti_core的add_episode，看是否能成功存储。

3. **检查graphiti_core版本兼容性**

当前使用的是PyPI的graphiti-core 0.14.0，可能与MCP server代码不兼容。

## 下一步行动

由于我无法直接看到MCP服务器的stderr输出或日志，我建议：

1. **手动检查日志文件**（如果存在）
   - 查看是否有任何graphiti_mcp_server的日志文件
   - 检查Windows Event Viewer

2. **手动启动MCP服务器查看输出**
   ```bash
   cd /c/workspace/graphiti/mcp_server
   uv run graphiti_mcp_server.py --transport stdio
   ```
   然后观察输出

3. **运行独立测试**
   尝试不通过MCP，直接测试graphiti_core的功能

4. **联系graphiti开发者**
   这可能是graphiti 0.14.0版本的已知问题

## 总结

我已经修复了代码中明显的bug（asyncio任务垃圾回收），但问题比预期的更深层。真正的问题可能在于：

- MCP框架与async的交互
- graphiti_core库本身的问题
- 环境配置问题

需要更多的调试信息才能确定根本原因。最可能的情况是worker任务根本没有启动，这需要通过查看实际的MCP服务器日志来确认。

---
创建时间：2025-10-03 19:35
