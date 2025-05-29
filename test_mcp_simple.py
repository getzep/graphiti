#!/usr/bin/env python3
"""
Simple test to verify MCP server is responding
"""
import subprocess
import sys
import os

def test_mcp_direct():
    """Test MCP server directly with stdio"""
    print("🔧 Testing MCP server directly...")
    
    # Change to MCP server directory
    os.chdir("mcp_server")
    
    # Test if the server starts and responds to initialization
    try:
        # Start the server
        process = subprocess.Popen([
            "uv", "run", "graphiti_mcp_server.py",
            "--transport", "stdio",
            "--group-id", "direct-test"
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Send MCP initialization message
        init_message = '''{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}'''
        
        # Send the message
        stdout, stderr = process.communicate(input=init_message + "\n", timeout=10)
        
        if "result" in stdout and "capabilities" in stdout:
            print("✅ MCP server responds correctly!")
            print(f"Response preview: {stdout[:200]}...")
            return True
        else:
            print(f"❌ Unexpected response: {stdout}")
            print(f"❌ Errors: {stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ MCP server timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_mcp_direct()
    if success:
        print("\n🎯 MCP server is working! Now check VS Code:")
        print("1. Restart VS Code completely")
        print("2. Look for 'MCP Servers' button in left activity bar")
        print("3. Open GitHub Copilot Chat and try: '@copilot help'")
    sys.exit(0 if success else 1)
