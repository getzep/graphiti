#!/usr/bin/env python3
"""Simple diagnostic script for graphiti MCP server"""

import asyncio
import os
import sys
from pathlib import Path

# Set environment variables
os.environ['NEO4J_URI'] = 'bolt://localhost:7687'
os.environ['NEO4J_USER'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'graphiti123!'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC4YW25znj-zTc0BwEYmXf446XP8rNBFes'
os.environ['MODEL_NAME'] = 'gemini-2.5-pro'
os.environ['EMBEDDER_MODEL'] = 'models/text-embedding-004'

print("=" * 60)
print("Graphiti MCP Server Diagnostics")
print("=" * 60)

# Check 1: Neo4j connection
print("\n[Check 1] Neo4j Connection")
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
        print("  OK - Neo4j connection successful")
        print(f"  Current node count: {result.stdout}")
    else:
        print(f"  ERROR - Neo4j response abnormal: {result.stdout}")
except Exception as e:
    print(f"  ERROR - Neo4j connection failed: {e}")

# Check 2: Python environment
print("\n[Check 2] Python Environment")
print(f"  Python version: {sys.version}")
print(f"  Working directory: {os.getcwd()}")

# Check 3: Dependencies
print("\n[Check 3] Key Dependencies")
required_packages = ['graphiti_core', 'neo4j', 'google.generativeai', 'asyncio']
for pkg_name in required_packages:
    try:
        if pkg_name == 'google.generativeai':
            import google.generativeai as genai
            print(f"  OK - {pkg_name} installed")
        elif pkg_name == 'graphiti_core':
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from graphiti_core import Graphiti
            print(f"  OK - {pkg_name} importable")
        else:
            __import__(pkg_name)
            print(f"  OK - {pkg_name} installed")
    except ImportError as e:
        print(f"  ERROR - {pkg_name} not installed or cannot import: {e}")

# Check 4: Environment variables
print("\n[Check 4] Environment Variables")
env_vars = ['NEO4J_URI', 'NEO4J_USER', 'GOOGLE_API_KEY', 'MODEL_NAME']
for var in env_vars:
    value = os.environ.get(var)
    if value:
        display_value = value[:20] + '...' if len(value) > 20 else value
        if 'KEY' in var or 'PASSWORD' in var:
            display_value = value[:10] + '***'
        print(f"  OK - {var}: {display_value}")
    else:
        print(f"  ERROR - {var}: Not set")

# Check 5: Bug fix code
print("\n[Check 5] Bug Fix Code Check")
script_path = Path(__file__).parent / 'graphiti_mcp_server.py'
with open(script_path, 'r', encoding='utf-8') as f:
    content = f.read()

checks = [
    ('queue_tasks dict', 'queue_tasks: dict[str, asyncio.Task]'),
    ('global declaration', 'global graphiti_client, episode_queues, queue_workers, queue_tasks'),
    ('task storage', 'queue_tasks[group_id_str] = task'),
    ('DEBUG worker start', 'DEBUG: Worker STARTED'),
    ('DEBUG processing', 'DEBUG: Processing episode'),
    ('DEBUG task creation', 'DEBUG: Creating worker task'),
]

for check_name, check_string in checks:
    if check_string in content:
        print(f"  OK - {check_name}: Applied")
    else:
        print(f"  ERROR - {check_name}: Not found")

# Check 6: Test Gemini API
print("\n[Check 6] Test Gemini API Connection")
try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    response = model.generate_content("Say 'API works'")
    print(f"  OK - Gemini API response: {response.text[:50]}")
except Exception as e:
    print(f"  ERROR - Gemini API call failed: {e}")

print("\n" + "=" * 60)
print("Diagnostics Complete")
print("=" * 60)

print("\nRecommendations:")
print("1. If Neo4j connection fails, check if Neo4j is running on ports 7474/7687")
print("2. If Gemini API fails, check if GOOGLE_API_KEY is valid")
print("3. If dependencies are missing, run: uv sync in mcp_server directory")
print("4. Check MCP server actual error logs")
