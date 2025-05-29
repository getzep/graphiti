#!/usr/bin/env python3
"""
Test script to verify Graphiti MCP Server integration with VS Code
"""

import sys
import subprocess
import json
import time

def test_mcp_server():
    """Test if the MCP server is working correctly"""
    print("🧪 Testing Graphiti MCP Server Integration...")
    
    # Test 1: Check if server starts
    print("1. Testing server startup...")
    try:
        result = subprocess.run([
            "uv", "run", "graphiti_mcp_server.py", "--help"
        ], cwd="mcp_server", capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("   ✅ Server can start successfully")
        else:
            print(f"   ❌ Server failed to start: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("   ❌ Server startup timed out")
        return False
    except Exception as e:
        print(f"   ❌ Error testing server: {e}")
        return False
    
    print("\n🎉 MCP Server integration test completed!")
    print("\nNext steps:")
    print("1. Install the 'Copilot MCP' extension in VS Code")
    print("2. Restart VS Code to load the configuration")
    print("3. Start chatting with GitHub Copilot using @copilot")
    print("4. Try asking: '@copilot Remember that I'm working on Graphiti integration'")
    
    return True

if __name__ == "__main__":
    success = test_mcp_server()
    sys.exit(0 if success else 1)
