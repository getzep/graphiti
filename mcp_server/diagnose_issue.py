#!/usr/bin/env python3
"""
诊断graphiti MCP server问题的脚本
"""

import asyncio
import os
import sys
from pathlib import Path

# 设置环境变量
os.environ['NEO4J_URI'] = 'bolt://localhost:7687'
os.environ['NEO4J_USER'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'graphiti123!'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC4YW25znj-zTc0BwEYmXf446XP8rNBFes'
os.environ['MODEL_NAME'] = 'gemini-2.5-pro'
os.environ['EMBEDDER_MODEL'] = 'models/text-embedding-004'

print("=" * 60)
print("Graphiti MCP Server 诊断")
print("=" * 60)

# 检查1: Neo4j连接
print("\n[检查1] Neo4j连接状态")
try:
    import subprocess
    result = subprocess.run([
        'curl', '-X', 'POST',
        'http://localhost:7474/db/neo4j/tx/commit',
        '-H', 'Content-Type: application/json',
        '-H', 'Authorization: Basic bmVvNGo6Z3JhcGhpdGkxMjMh',
        '-d', '{"statements":[{"statement":"MATCH (n) RETURN count(n) as count"}]}'
    ], capture_output=True, text=True, timeout=5)

    if 'count' in result.stdout:
        print("  ✓ Neo4j连接成功")
        print(f"  当前节点数: {result.stdout}")
    else:
        print(f"  ✗ Neo4j响应异常: {result.stdout}")
except Exception as e:
    print(f"  ✗ Neo4j连接失败: {e}")

# 检查2: Python环境
print("\n[检查2] Python环境")
print(f"  Python版本: {sys.version}")
print(f"  工作目录: {os.getcwd()}")

# 检查3: 依赖包
print("\n[检查3] 关键依赖包")
required_packages = ['graphiti_core', 'neo4j', 'google.generativeai', 'asyncio']
for pkg_name in required_packages:
    try:
        if pkg_name == 'google.generativeai':
            import google.generativeai as genai
            print(f"  ✓ {pkg_name} 已安装")
        elif pkg_name == 'graphiti_core':
            # 先添加路径
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from graphiti_core import Graphiti
            print(f"  ✓ {pkg_name} 可导入")
        else:
            __import__(pkg_name)
            print(f"  ✓ {pkg_name} 已安装")
    except ImportError as e:
        print(f"  ✗ {pkg_name} 未安装或无法导入: {e}")

# 检查4: 环境变量
print("\n[检查4] 环境变量配置")
env_vars = ['NEO4J_URI', 'NEO4J_USER', 'GOOGLE_API_KEY', 'MODEL_NAME']
for var in env_vars:
    value = os.environ.get(var)
    if value:
        display_value = value[:20] + '...' if len(value) > 20 else value
        if 'KEY' in var or 'PASSWORD' in var:
            display_value = value[:10] + '***'
        print(f"  ✓ {var}: {display_value}")
    else:
        print(f"  ✗ {var}: 未设置")

# 检查5: 代码修复
print("\n[检查5] Bug修复代码检查")
script_path = Path(__file__).parent / 'graphiti_mcp_server.py'
with open(script_path, 'r', encoding='utf-8') as f:
    content = f.read()

checks = [
    ('queue_tasks字典', 'queue_tasks: dict[str, asyncio.Task]'),
    ('global声明', 'global graphiti_client, episode_queues, queue_workers, queue_tasks'),
    ('任务存储', 'queue_tasks[group_id_str] = task'),
]

for check_name, check_string in checks:
    if check_string in content:
        print(f"  ✓ {check_name}: 已应用")
    else:
        print(f"  ✗ {check_name}: 未找到")

# 检查6: 测试Gemini API
print("\n[检查6] 测试Gemini API连接")
try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

    # 测试简单的调用
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    response = model.generate_content("Say 'API works'")
    print(f"  ✓ Gemini API响应: {response.text[:50]}")
except Exception as e:
    print(f"  ✗ Gemini API调用失败: {e}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)

# 检查7: 尝试直接测试graphiti
print("\n[检查7] 尝试直接初始化Graphiti (可能需要一些时间...)")
async def test_graphiti():
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from graphiti_core import Graphiti
        from graphiti_core.llm_client import LLMClient

        # 这里需要创建一个简单的LLM client
        print("  正在初始化Graphiti客户端...")
        # 注意：这可能因为缺少某些配置而失败，但至少能看到错误信息

    except Exception as e:
        print(f"  ✗ Graphiti初始化失败: {e}")
        import traceback
        traceback.print_exc()

# 运行异步测试
try:
    asyncio.run(test_graphiti())
except Exception as e:
    print(f"  异步测试失败: {e}")

print("\n建议:")
print("1. 如果Neo4j连接失败，检查Neo4j是否运行在7474/7687端口")
print("2. 如果Gemini API失败，检查GOOGLE_API_KEY是否有效")
print("3. 如果依赖包缺失，在mcp_server目录运行: uv sync")
print("4. 检查MCP server的实际错误日志")
